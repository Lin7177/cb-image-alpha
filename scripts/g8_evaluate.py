"""One reveal after all G8 outer predictions exist; paired calendar-block intervals."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cb_image_alpha.artifacts import read_json, sha256, write_json
from cb_image_alpha.evaluation.metrics import binary_metrics

METHODS = ("Constant", "U", "Q", "CNN")
PAIRS = (("U", "Constant"), ("Q", "Constant"), ("CNN", "Constant"), ("U", "Q"), ("CNN", "Q"))


def draw_blocks(rng: np.random.Generator, size: int, length: int) -> np.ndarray:
    if size < length:
        raise ValueError("Too few calendar days for frozen block length")
    starts = rng.integers(0, size - length + 1, size=(size + length - 1) // length)
    return (starts[:, None] + np.arange(length)).ravel()[:size]


def evaluate(run: Path) -> None:
    plan = read_json(run / "plan.json")
    config = plan["config"]
    for year in config["fold_years"]:
        for method in METHODS:
            if not (run / str(year) / f"{method}-p.npy").exists():
                raise ValueError("All outer predictions must exist before evaluation")
    index = pq.read_table(run / "index.parquet")
    frozen = read_json(Path(plan["original"]) / "frozen.json")
    path = Path(frozen["source"]).parent / "download/base/calendar.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        calendar = [date.fromisoformat(r["trading_date"]) for r in csv.DictReader(f)]
    reconciliation = {}
    matrix = {}
    blocks = {}
    uncertainty = {}
    differences = {}
    for year in config["fold_years"]:
        folder = run / str(year)
        ids = np.load(folder / "outer.npy")
        table = index.take(pa.array(ids))
        y = table["label"].to_numpy()
        dates = table["window_end"].to_pylist()
        days = [d for d in calendar if d.year == year]
        mapping = {d: i for i, d in enumerate(days)}
        day_ids = np.array([mapping[d] for d in dates])
        counts = np.bincount(day_ids, minlength=len(days))
        if len(set(table["sample_id"].to_pylist())) != len(ids):
            raise ValueError("Duplicate outer ID")
        matrix[str(year)] = {}
        sums = {}
        for name in METHODS:
            p = np.load(folder / f"{name}-p.npy", allow_pickle=False)
            if p.shape != (len(ids),):
                raise ValueError("Probability count mismatch")
            metrics = binary_metrics(y, p)
            clipped = np.clip(p, 1e-12, 1 - 1e-12)
            losses = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
            sums[name] = np.bincount(day_ids, weights=losses, minlength=len(days))
            metrics["date_equal_log_loss"] = float(
                (sums[name][counts > 0] / counts[counts > 0]).mean()
            )
            matrix[str(year)][name] = metrics
            table = table.append_column(name + "_p", pa.array(p))
        rate = read_json(folder / "constant.json")["train_positive_rate"]
        matrix[str(year)]["train_majority"] = binary_metrics(y, np.full(len(y), float(rate > 0.5)))
        pq.write_table(table, folder / "outer-predictions.parquet")
        rng = np.random.default_rng(np.random.SeedSequence([config["bootstrap_seed"], year]))
        repetitions = config["bootstrap_replicates"]
        draws = []
        eligible = int((counts > 0).sum()) >= config["bootstrap_block_days"]
        for _ in range(repetitions if eligible else 0):
            draw = draw_blocks(rng, len(days), config["bootstrap_block_days"])
            draws.append(draw)
        blocks[str(year)] = np.stack(draws) if draws else np.empty((0, len(days)), dtype=np.int64)
        np.save(folder / "bootstrap-date-indices.npy", blocks[str(year)], allow_pickle=False)
        boot = {}
        diffs = {}
        for a, b in PAIRS:
            key = a + "-" + b
            delta = sums[a] - sums[b]
            row = []
            equal = []
            for draw in draws:
                mass = counts[draw].sum()
                if mass == 0:
                    raise ValueError("Empty bootstrap sample")
                row.append(float(delta[draw].sum() / mass))
                valid = draw[counts[draw] > 0]
                equal.append(float((delta[valid] / counts[valid]).mean()))
            boot[key] = {"row": row, "date_equal": equal}
            diffs[key] = {
                "row": matrix[str(year)][a]["log_loss"] - matrix[str(year)][b]["log_loss"],
                "date_equal": matrix[str(year)][a]["date_equal_log_loss"]
                - matrix[str(year)][b]["date_equal_log_loss"],
            }
        uncertainty[str(year)] = {
            key: {
                kind: np.quantile(values, config["bootstrap_quantiles"]).tolist()
                if len(values)
                else None
                for kind, values in item.items()
            }
            for key, item in boot.items()
        }
        differences[str(year)] = diffs
        reconciliation[str(year)] = {
            "count": len(ids),
            "unique_sample_ids": len(ids),
            "method_order_shared": True,
            "outer_indices_sha256": sha256(folder / "outer.npy"),
            "interval_undefined_reason": None if eligible else "fewer_than_20_effective_dates",
        }
        write_json(
            folder / "daily-losses.json",
            {
                "dates": [str(d) for d in days],
                "counts": counts.tolist(),
                "loss_sums": {key: value.tolist() for key, value in sums.items()},
                "bootstrap_draws_sha256": sha256(folder / "bootstrap-date-indices.npy"),
            },
        )
        write_json(folder / "bootstrap-replicates.json", boot)
    combined = {}
    for a, b in PAIRS:
        key = a + "-" + b
        combined[key] = {}
        for kind in ("row", "date_equal"):
            replicates = [
                read_json(run / str(year) / "bootstrap-replicates.json")[key][kind]
                for year in config["fold_years"]
            ]
            values = (
                np.mean(replicates, axis=0)
                if all(len(r) == config["bootstrap_replicates"] for r in replicates)
                else np.array([])
            )
            combined[key][kind] = {
                "mean_difference": float(
                    np.mean([differences[str(year)][key][kind] for year in config["fold_years"]])
                ),
                "interval": np.quantile(values, config["bootstrap_quantiles"]).tolist()
                if len(values)
                else None,
            }
    means = {}
    for method in METHODS:
        means[method] = {}
        for metric in ("log_loss", "accuracy", "balanced_accuracy", "auc", "date_equal_log_loss"):
            values = [matrix[str(year)][method][metric] for year in config["fold_years"]]
            means[method][metric] = (
                float(np.mean(values)) if all(v is not None for v in values) else None
            )
    write_json(run / "reconciliation.json", reconciliation)
    write_json(
        run / "results.json",
        {
            "scope": "development",
            "matrix": matrix,
            "three_fold_core_means": means,
            "differences": differences,
            "fold_intervals": uncertainty,
            "three_fold_equal_mean": combined,
            "uncertainty": "paired 20-calendar-trading-day blocks,500 draws,95% percentile descriptive intervals; no confirmation or causality",
            "test_used": False,
            "all_outer_results_revealed_together": True,
        },
    )
