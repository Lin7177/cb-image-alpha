"""Independently validate completed G2 artifacts and inspect actual source windows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path

import pyarrow.parquet as pq

from cb_image_alpha.data.catalog import read_instrument
from cb_image_alpha.samples.index import read_partitions


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path) -> dict[str, object]:
    if not __debug__:
        raise RuntimeError("Verification requires assertions; remove -O/PYTHONOPTIMIZE")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    for name, expected in manifest["artifacts_sha256"].items():
        assert sha(root / name) == expected, name
    source = Path(manifest["inputs"]["source"])
    project = Path(__file__).resolve().parents[1]
    parts = {p.name: p for p in read_partitions(project / "configs/splits.toml")}
    with (source.parent / "download/base/calendar.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        calendar = [date.fromisoformat(r["trading_date"]) for r in csv.DictReader(handle)]
    positions = {d: i for i, d in enumerate(calendar)}
    counts: Counter[str] = Counter()
    ids: set[str] = set()
    selected: dict[str, dict] = {}
    split_dates: dict[date, str] = {}
    end_times = {}
    start_times = {}
    n = 0
    with pq.ParquetFile(root / "samples.parquet") as reader:
        for batch in reader.iter_batches(batch_size=8192):
            for row in batch.to_pylist():
                split = row["split"]
                part = parts[split]
                day = row["window_end"]
                i = positions[day]
                assert part.start <= day <= part.end
                assert row["label_end"] == calendar[i + 5] <= part.end
                assert row["context_start"] == calendar[i - 38]
                assert row["window_start"] == calendar[i - 19]
                assert row["label_start"] == day
                assert row["decision_time"].date() == day
                assert row["label_available_at"].date() == row["label_end"]
                for key in ("decision_time", "label_available_at"):
                    assert row[key].strftime("%H:%M:%S%z") == "23:59:59+0800"
                assert row["image_row"] == n
                assert row["sample_id"] not in ids
                ids.add(row["sample_id"])
                assert split_dates.setdefault(day, split) == split
                assert row["label"] == int(Decimal(row["end_close"]) > Decimal(row["start_close"]))
                counts[split] += 1
                n += 1
                end_times[split] = max(end_times.get(split, row["label_available_at"]),
                                      row["label_available_at"])
                start_times[split] = min(start_times.get(split, row["decision_time"]),
                                        row["decision_time"])
                selected.setdefault(split + "_first", row)
                selected[split + "_last"] = row
    assert end_times["train"] < start_times["validation"]
    assert end_times["validation"] < start_times["test"]
    assert n == summary["retained"] and dict(counts) == summary["split_counts"]
    reasons: Counter[str] = Counter()
    retained = set()
    total = 0
    with pq.ParquetFile(root / "decisions.parquet") as reader:
        for batch in reader.iter_batches(batch_size=8192):
            for row in batch.to_pylist():
                reasons[f'{row["decision_date"].year}|{row["candidate_split"]}|{row["reason"]}'] += 1
                total += 1
                if row["reason"] == "retained":
                    key = f'g2-v1.2|{row["instrument_id"]}|{row["decision_date"].isoformat()}'
                    retained.add(hashlib.sha256(key.encode()).hexdigest())
    assert retained == ids
    assert total == summary["candidate_rows"]
    assert dict(reasons) == summary["year_split_reason"]
    inspected = []
    raw_cache = {}
    for name, row in selected.items():
        instrument = row["instrument_id"]
        if instrument not in raw_cache:
            raw = read_instrument(source / "day01/data/bar" / f"{instrument}-1-DAY-LAST-EXTERNAL")
            raw_cache[instrument] = {
                datetime.fromtimestamp(b.ts_event // 10**9, UTC).date(): b for b in raw}
        lookup = raw_cache[instrument]
        i = positions[row["window_end"]]
        assert all(d in lookup for d in calendar[i - 38:i + 1])
        assert all(lookup[d].volume > 0 for d in calendar[i - 19:i + 1])
        first = lookup[row["window_end"]].close
        last = lookup[row["label_end"]].close
        assert Decimal(row["start_close"]) == first and Decimal(row["end_close"]) == last
        with localcontext() as ctx:
            ctx.prec = 34
            ctx.rounding = ROUND_HALF_EVEN
            assert Decimal(row["forward_return"]) == last / first - 1
        inspected.append({"case": name, "instrument": instrument,
                          "decision": str(row["window_end"]), "endpoint": str(row["label_end"]),
                          "start_close": str(first), "end_close": str(last)})
    remaining = total
    stages = [{"stage": "candidates", "remaining": remaining}]
    for reason in ("outside_range", "history_insufficient", "history_gap",
                   "display_zero_volume", "label_missing", "purged_boundary"):
        removed = sum(v for k, v in reasons.items() if k.endswith("|" + reason))
        remaining -= removed
        stages.append({"stage": reason, "removed": removed, "remaining": remaining})
    assert remaining == n
    return {"status": "passed", "stage_counts": stages,
            "verifier_sha256": sha(Path(__file__)),"assertions_enabled": __debug__, "dataset": str(root), "rows": n,
            "candidates": total, "split_counts": dict(counts), "source_cases": inspected,
            "artifact_hashes_verified": manifest["artifacts_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.dataset), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()