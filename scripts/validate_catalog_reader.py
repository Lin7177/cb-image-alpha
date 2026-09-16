"""Run the G1 reader over the frozen inventory and verify source fingerprints."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from cb_image_alpha.config import ResearchPolicy
from cb_image_alpha.data.catalog import read_instrument
from cb_image_alpha.paths import validate_output_path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    source = args.source.resolve(strict=True)
    out = validate_output_path(
        Path("artifacts/data_audit"), project_root=project, source_root=source
    )
    policy_path = project / "configs/data_policy.toml"
    ResearchPolicy.read(policy_path)
    quality = json.loads(args.quality_report.read_text(encoding="utf-8"))
    if Path(quality["source"]).resolve() != source:
        raise ValueError("Quality evidence is for another source")
    base = source.parent / "download/base"
    identity_path = base / "cb_info.csv"
    calendar_path = base / "calendar.csv"
    auxiliary = {str(p): digest(p) for p in (identity_path, calendar_path)}
    with identity_path.open(encoding="utf-8-sig", newline="") as handle:
        codes = [r["债券代码"].zfill(6) for r in csv.DictReader(handle)]
    if len(codes) != len(set(codes)):
        raise ValueError("Duplicate identity codes")
    with calendar_path.open(encoding="utf-8-sig", newline="") as handle:
        days = [r["trading_date"] for r in csv.DictReader(handle)]
    if days != sorted(set(days)):
        raise ValueError("Calendar is not sorted/unique")
    allowed_days = set(days)
    available = sorted(
        d for d in (source / "day01/data/bar").iterdir()
        if d.is_dir() and d.name.split(".")[0] in set(codes)
    )
    expected = {str(Path(f["path"]).parent) for f in quality["source_files"]}
    actual = {str(d.relative_to(source)) for d in available}
    if actual != expected:
        raise ValueError("Identity/catalog selection changed since quality audit")
    file_hashes = {f["path"]: f["sha256"] for f in quality["source_files"]}
    records = []
    seen = set()
    for directory in available:
        paths = list(directory.glob("*.parquet"))
        if len(paths) != 1:
            raise ValueError("Instrument file count changed")
        path = paths[0]
        relative = str(path.relative_to(source))
        if digest(path) != file_hashes[relative]:
            raise ValueError(f"Source changed since quality audit: {relative}")
        bars = read_instrument(directory)
        for bar in bars:
            day = datetime.fromtimestamp(bar.ts_event // 10**9, UTC).date().isoformat()
            key = (bar.instrument_id, day)
            if key in seen or day not in allowed_days:
                raise ValueError(f"Duplicate or non-calendar key: {key}")
            seen.add(key)
        if digest(path) != file_hashes[relative]:
            raise ValueError(f"Source changed during reader validation: {relative}")
        records.append({"instrument": bars[0].instrument_id, "rows": len(bars)})
    for path, before in auxiliary.items():
        if digest(Path(path)) != before:
            raise ValueError(f"Auxiliary source changed: {path}")
    report = {
        "status": "passed", "created_at": datetime.now(UTC).isoformat(),
        "source": str(source), "instrument_count": len(records), "row_count": len(seen),
        "policy_sha256": digest(policy_path), "quality_report_sha256": digest(args.quality_report),
        "auxiliary_sha256": auxiliary, "source_content_unchanged_during_check": True,
        "quality_source_hashes_revalidated": len(file_hashes),
        "identity_count": len(codes), "calendar_count": len(days), "records": records,
    }
    out.mkdir(parents=True, exist_ok=True)
    target = out / ("reader-validation-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=False))
    print(target)


if __name__ == "__main__":
    main()
