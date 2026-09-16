"""Audit all identity-matched daily bars; write evidence only inside this project."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as pq

FIELDS = ("open", "high", "low", "close", "volume")
SCALE = 10**16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    project = Path(__file__).resolve().parents[1]
    out = (project / "artifacts" / "data_audit").resolve()
    if not out.is_relative_to(project) or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Unsafe output")
    base = source.parent / "download/base"
    with (base / "cb_info.csv").open(encoding="utf-8-sig", newline="") as f:
        identities = {r["债券代码"].zfill(6) for r in csv.DictReader(f)}
    with (base / "calendar.csv").open(encoding="utf-8-sig", newline="") as f:
        calendar = {r["trading_date"] for r in csv.DictReader(f)}
    closes: dict[tuple[str, str], int] = {}
    records = []
    all_hours: Counter[str] = Counter()
    all_flags: Counter[str] = Counter()
    meta_profiles: Counter[str] = Counter()
    source_files = []
    for directory in sorted((source / "day01/data/bar").iterdir()):
        code = directory.name.split(".")[0]
        if not directory.is_dir() or code not in identities:
            continue
        for path in sorted(directory.glob("*.parquet")):
            before = path.stat()
            raw = path.read_bytes()
            table = pq.read_table(path)
            meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
            meta_profiles[json.dumps(meta, sort_keys=True)] += 1
            flags: Counter[str] = Counter()
            days = []
            volumes = []
            previous = -1
            for r in table.to_pylist():
                if any(not isinstance(r[x], bytes) or len(r[x]) != 16 for x in FIELDS):
                    flags["wrong_binary_width_or_null"] += 1
                    continue
                values = {x: int.from_bytes(r[x], "little", signed=True) for x in FIELDS}
                ts = r["ts_event"]
                instant = datetime.fromtimestamp(ts // 10**9, UTC)
                day = instant.date().isoformat()
                all_hours[instant.strftime("%H:%M:%S")] += 1
                days.append(day)
                volumes.append(values["volume"])
                if ts <= previous:
                    flags["non_increasing_timestamp"] += 1
                previous = ts
                if day not in calendar:
                    flags["not_calendar_day"] += 1
                if r["ts_event"] != r["ts_init"]:
                    flags["event_init_differ"] += 1
                if any(values[x] <= 0 for x in FIELDS[:4]):
                    flags["nonpositive_price"] += 1
                if not values["low"] <= min(values["open"], values["close"]) <= max(
                    values["open"], values["close"]
                ) <= values["high"]:
                    flags["invalid_ohlc_order"] += 1
                if values["volume"] < 0:
                    flags["negative_volume"] += 1
                if values["volume"] == 0:
                    flags["zero_volume"] += 1
                if values["volume"] % SCALE:
                    flags["fractional_catalog_volume"] += 1
                if any(values[x] % 10**13 for x in FIELDS[:4]):
                    flags["price_precision_beyond_3"] += 1
                closes[(day, ".".join(directory.name.split(".")[:2]))] = values["close"]
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError("Source changed while auditing")
            all_flags.update(flags)
            source_files.append({
                "path": str(path.relative_to(source)), "size": before.st_size,
                "mtime_ns": before.st_mtime_ns, "sha256": hashlib.sha256(raw).hexdigest(),
            })
            records.append({
                "instrument": meta.get("instrument_id"), "rows": table.num_rows,
                "first": min(days) if days else None, "last": max(days) if days else None,
                "flags": dict(flags), "volume_min": str(min(volumes) // SCALE) if volumes else None,
                "volume_max": str(max(volumes) // SCALE) if volumes else None,
            })
    comparison: Counter[str] = Counter()
    mismatches = []
    close_path = base / "day01_close.csv"
    with close_path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            key = (r["time"][:10], r["tr_code"])
            if key not in closes:
                continue
            comparison["matched_rows"] += 1
            value = Decimal(r["close"])
            if value * SCALE != closes[key]:
                comparison["mismatch_rows"] += 1
                if len(mismatches) < 20:
                    mismatches.append({
                        "key": key, "csv_close": str(value),
                        "catalog_close": str(Decimal(closes[key]) / SCALE),
                    })
    summary = {
        "files": len(records), "rows": sum(x["rows"] for x in records),
        "first": min(x["first"] for x in records if x["first"]),
        "last": max(x["last"] for x in records if x["last"]),
        "time_of_day_as_encoded_utc": dict(all_hours), "flags": dict(all_flags),
        "close_comparison": dict(comparison), "close_mismatch_examples": mismatches,
    }
    # Per-instrument metadata differs by identity; summarize shared encoding separately.
    report = {
        "created_at": datetime.now(UTC).isoformat(), "status": "completed",
        "source": str(source), "summary": summary,
        "metadata_profiles": dict(meta_profiles), "records": records, "source_files": source_files,
        "close_reference": {"path": str(close_path), "sha256": hashlib.sha256(
            close_path.read_bytes()).hexdigest()},
        "limitations": "Decoded catalog units only; raw physical volume unit is not asserted",
    }
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = out / f"quality-{stamp}.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(target), **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
