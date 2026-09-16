"""Independent chronological views over immutable G5 development image shards."""

from __future__ import annotations

import csv
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from cb_image_alpha.artifacts import read_json, sha256, write_json
from cb_image_alpha.data.encoding import decode_fixed_i128


def fold_roles(days: list[date], endpoints: list[date], year: int) -> dict[str, np.ndarray]:
    d = np.array([x.year for x in days])
    e = np.array([x.year for x in endpoints])
    return {
        name: np.flatnonzero(mask)
        for name, mask in {
            "inner_train": (d >= 2020) & (d <= year - 2) & (e <= year - 2),
            "inner_validation": (d == year - 1) & (e <= year - 1),
            "final_train": (d >= 2020) & (d <= year - 1) & (e <= year - 1),
            "outer": (d == year) & (e <= year),
        }.items()
    }


def numerical_features(prices: np.ndarray, volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if (
        prices.shape != (39, 4)
        or volume.shape != (20,)
        or not np.isfinite(prices).all()
        or not np.isfinite(volume).all()
    ):
        raise ValueError("Invalid feature input")
    if np.any(prices <= 0) or np.any(volume <= 0):
        raise ValueError("Nonpositive input")
    ma = np.array([prices[i - 19 : i + 1, 3].mean() for i in range(19, 39)])
    values = np.column_stack((prices[-20:], ma)) / prices[19, 3]
    low, high = values.min(), values.max()
    if high == low:
        raise ValueError("Degenerate window")
    u = np.column_stack(((values - low) / (high - low), volume / volume.max()))
    q = np.column_stack((np.rint(50 * u[:, :5]) / 50, np.maximum(1, np.rint(12 * u[:, 5])) / 12))
    return u.reshape(120), q.reshape(120)


class FoldImages:
    def __init__(self, run: Path, indices: np.ndarray, normalization: dict | None = None):
        self.original = Path(read_json(run / "plan.json")["original"])
        table = pq.read_table(
            run / "index.parquet", columns=["split", "shard_id", "shard_row", "label"]
        )
        self.split = np.array(table["split"].to_pylist())
        self.shard = table["shard_id"].to_numpy()
        self.row = table["shard_row"].to_numpy()
        self.labels = table["label"].to_numpy()
        self.indices = indices
        self.cache = {}
        if normalization is None:
            counts = np.load(run / "white-counts.npy", mmap_mode="r")
            mean = int(counts[indices].sum()) / (len(indices) * 3840)
            normalization = {
                "mean": mean,
                "std": float(np.sqrt(mean * (1 - mean))),
                "count": len(indices),
                "ddof": 0,
                "fit_indices_sha256": sha256_array(indices),
            }
        if normalization["std"] <= 0:
            raise ValueError("Invalid image normalization")
        self.normalization = normalization

    def batch(self, positions: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.indices[positions]
        pixels = np.empty((len(ids), 64, 60), dtype=np.uint8)
        for split, shard in set(zip(self.split[ids], self.shard[ids])):
            key = (str(split), int(shard))
            if key not in self.cache:
                self.cache[key] = np.load(
                    self.original / split / f"{shard:04d}.npy", mmap_mode="r", allow_pickle=False
                )
            mask = (self.split[ids] == split) & (self.shard[ids] == shard)
            pixels[mask] = self.cache[key][self.row[ids[mask]]]
        x = torch.from_numpy(pixels).float().unsqueeze(1) / 255
        x = (x - self.normalization["mean"]) / self.normalization["std"]
        return x, torch.from_numpy(self.labels[ids].copy()).long()


def sha256_array(array: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.asarray(array, dtype="<i8").tobytes()).hexdigest()


def prepare(run: Path) -> None:
    plan = read_json(run / "plan.json")
    original = Path(plan["original"])
    frozen = read_json(original / "frozen.json")
    source = Path(frozen["source"])
    tables = []
    for split, count in (("train", 339682), ("validation", 123747)):
        table = pq.read_table(original / split / "index.parquet")
        if len(table) != count or set(table["split"].to_pylist()) != {split}:
            raise ValueError("Development index scope mismatch")
        tables.append(table)
    index = pa.concat_tables(tables).sort_by([("image_row", "ascending")])
    if len(set(index["sample_id"].to_pylist())) != len(index):
        raise ValueError("Duplicate sample IDs")
    if len(np.unique(index["image_row"].to_numpy())) != len(index):
        raise ValueError("Duplicate image rows")
    pq.write_table(index, run / "index.parquet")
    columns = [
        "sample_id",
        "instrument_id",
        "context_start",
        "window_start",
        "window_end",
        "label_end",
        "split",
        "shard_id",
        "shard_row",
    ]
    rows = index.select(columns).to_pylist()
    calendar_path = source.parent / "download/base/calendar.csv"
    with calendar_path.open(encoding="utf-8-sig", newline="") as f:
        calendar = [date.fromisoformat(r["trading_date"]) for r in csv.DictReader(f)]
    positions = {day: i for i, day in enumerate(calendar)}
    u = np.lib.format.open_memmap(
        run / "U.npy", mode="w+", dtype=np.float64, shape=(len(rows), 120)
    )
    q = np.lib.format.open_memmap(
        run / "Q.npy", mode="w+", dtype=np.float64, shape=(len(rows), 120)
    )
    current = None
    lookup = {}
    sources = {}
    for i, row in enumerate(rows):
        if row["window_end"] > date(2024, 12, 31):
            raise ValueError("Outside development interval")
        if current != row["instrument_id"]:
            current = row["instrument_id"]
            candidates = list(
                (source / "day01/data/bar" / f"{current}-1-DAY-LAST-EXTERNAL").glob("*.parquet")
            )
            if len(candidates) != 1:
                raise ValueError("Source shape changed")
            path = candidates[0]
            expected = frozen["source_files"][str(path.relative_to(source))]
            if sha256(path) != expected:
                raise ValueError("Source changed")
            # Only dates that can occur in this instrument's authorized development contexts.
            instrument_rows = [r for r in rows[i:] if r["instrument_id"] == current]
            first = min(r["context_start"] for r in instrument_rows)
            last = max(r["window_end"] for r in instrument_rows)
            start = (
                int(datetime(first.year, first.month, first.day, 15, tzinfo=UTC).timestamp())
                * 10**9
            )
            end = int(datetime(last.year, last.month, last.day, 15, tzinfo=UTC).timestamp()) * 10**9
            raw = pq.read_table(
                path,
                filters=[("ts_event", ">=", start), ("ts_event", "<=", end)],
                use_threads=False,
            )
            if (raw.schema.metadata or {}).get(b"instrument_id", b"").decode() != current:
                raise ValueError("Source identity mismatch")
            lookup = {
                datetime.fromtimestamp(r["ts_event"] // 10**9, UTC).date(): [
                    float(decode_fixed_i128(r[f]))
                    for f in ("open", "high", "low", "close", "volume")
                ]
                for r in raw.to_pylist()
            }
            if sha256(path) != expected:
                raise ValueError("Source changed while reading")
            sources[str(path)] = expected
        pos = positions[row["window_end"]]
        days = calendar[pos - 38 : pos + 1]
        if len(days) != 39 or days[0] != row["context_start"] or days[19] != row["window_start"]:
            raise ValueError("Context mismatch")
        values = np.asarray([lookup[d] for d in days], dtype=np.float64)
        u[i], q[i] = numerical_features(values[:, :4], values[-20:, 4])
    u.flush()
    q.flush()
    del u, q
    whites = np.zeros(len(index), dtype=np.int64)
    splits = np.array(index["split"].to_pylist())
    shard = index["shard_id"].to_numpy()
    offsets = index["shard_row"].to_numpy()
    for split, part in sorted(set(zip(splits, shard))):
        array = np.load(original / split / f"{part:04d}.npy", mmap_mode="r", allow_pickle=False)
        if array.dtype != np.uint8 or array.shape[1:] != (64, 60):
            raise ValueError("Image shape mismatch")
        counts = np.count_nonzero(array, axis=(1, 2))
        mask = (splits == split) & (shard == part)
        whites[mask] = counts[offsets[mask]]
    np.save(run / "white-counts.npy", whites, allow_pickle=False)
    days = index["window_end"].to_pylist()
    ends = index["label_end"].to_pylist()
    fold_manifest = {}
    years = np.array([d.year for d in days])
    for year in plan["config"]["fold_years"]:
        folder = run / str(year)
        folder.mkdir()
        roles = fold_roles(days, ends, year)
        summary = {}
        for role, ids in roles.items():
            if not len(ids):
                raise ValueError("Empty fold role")
            np.save(folder / f"{role}.npy", ids, allow_pickle=False)
            candidate = (
                (years <= year - 2)
                if role == "inner_train"
                else (years <= year - 1)
                if role == "final_train"
                else (years == year - 1)
                if role == "inner_validation"
                else (years == year)
            )
            summary[role] = {
                "count": len(ids),
                "extra_purged": int(candidate.sum()) - len(ids),
                "indices_sha256": sha256_array(ids),
                "decision_min": str(min(days[i] for i in ids)),
                "decision_max": str(max(days[i] for i in ids)),
                "label_max": str(max(ends[i] for i in ids)),
            }
        if np.intersect1d(roles["final_train"], roles["outer"]).size or max(
            ends[i] for i in roles["final_train"]
        ) >= min(days[i] for i in roles["outer"]):
            raise ValueError("Chronological leakage")
        if max(ends[i] for i in roles["inner_train"]) >= min(
            days[i] for i in roles["inner_validation"]
        ):
            raise ValueError("Inner leakage")
        fold_manifest[str(year)] = summary
    write_json(run / "folds.json", fold_manifest)
    write_json(
        run / "preparation.json",
        {
            "status": "completed",
            "count": len(index),
            "sources": sources,
            "features": "U/Q 20x6 float64; no new labels or excluded-row reentry",
            "files": {
                str(p.relative_to(run)): sha256(p)
                for p in [
                    run / "index.parquet",
                    run / "U.npy",
                    run / "Q.npy",
                    run / "white-counts.npy",
                    run / "folds.json",
                    *run.glob("*/*.npy"),
                ]
            },
        },
    )
