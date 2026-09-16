"""Audit identity/calendar/catalog coverage using read-only standard-library I/O."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    project = Path(__file__).resolve().parents[1]
    destination = (project / "artifacts" / "data_audit").resolve()
    if (not destination.is_relative_to(project)
            or destination.is_relative_to(source)
            or source.is_relative_to(destination)):
        raise ValueError("Unsafe output directory")
    base = source.parent / "download" / "base"
    info_path = base / "cb_info.csv"
    calendar_path = base / "calendar.csv"
    with info_path.open(encoding="utf-8-sig", newline="") as handle:
        identities = list(csv.DictReader(handle))
    codes = [row["债券代码"].zfill(6) for row in identities]
    if len(set(codes)) != len(codes):
        raise ValueError("Duplicate convertible-bond identity codes")
    with calendar_path.open(encoding="utf-8-sig", newline="") as handle:
        days = [date.fromisoformat(row["trading_date"]) for row in csv.DictReader(handle)]
    if days != sorted(set(days)):
        raise ValueError("Calendar is not sorted/unique")
    code_set = set(codes)
    directories = sorted((source / "day01" / "data" / "bar").iterdir())
    selected = []
    all_count = 0
    for directory in directories:
        if not directory.is_dir():
            continue
        all_count += 1
        if directory.name.split(".")[0] not in code_set:
            continue
        files = [
            {"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(directory.glob("*.parquet"))
        ]
        selected.append({"instrument_directory": directory.name, "files": files})
    selected_codes = {x["instrument_directory"].split(".")[0] for x in selected}
    report = {
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "source": str(source),
        "identity": {
            "path": str(info_path), "sha256": sha256(info_path), "count": len(codes),
            "used_fields": ["债券代码"],
            "excluded_fields": "all current price, premium, credit and other snapshot fields",
        },
        "calendar": {
            "path": str(calendar_path), "sha256": sha256(calendar_path),
            "count": len(days), "first": str(days[0]), "last": str(days[-1]),
        },
        "catalog": {
            "all_day_directories": all_count,
            "identity_matched_directories": len(selected),
            "identity_codes_without_day_directory": sorted(code_set - selected_codes),
            "files_per_instrument": dict(Counter(len(x["files"]) for x in selected)),
            "selected": selected,
        },
        "limitations": [
            "Filename ranges are not validated row ranges",
            "Identity membership is not a point-in-time price or liquidity filter",
            "Historical universe completeness cannot be proven by this intersection alone",
            "OHLCV decoding and time semantics remain separate checks",
        ],
    }
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = destination / f"inventory-{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(path), "identity_count": len(codes), "calendar_count": len(days),
        "matched_directories": len(selected), "all_day_directories": all_count,
        "files_per_instrument": report["catalog"]["files_per_instrument"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
