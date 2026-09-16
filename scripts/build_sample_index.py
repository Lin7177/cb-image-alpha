"""Build or independently rebuild G2 indexes. Never trains or evaluates predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from cb_image_alpha.config import ResearchPolicy
from cb_image_alpha.data.catalog import read_instrument
from cb_image_alpha.paths import validate_output_path
from cb_image_alpha.samples.index import build_samples, exclusion_summary, read_partitions


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str,
                      separators=(",", ":")).encode()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(canonical(value))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reader-report", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--compare", type=Path, help="Completed prior dataset directory")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    source = args.source.resolve(strict=True)
    policy = project / "configs/data_policy.toml"
    split_path = project / "configs/splits.toml"
    ResearchPolicy.read(policy)
    partitions = read_partitions(split_path)
    evidence = json.loads(args.reader_report.read_text(encoding="utf-8"))
    quality = json.loads(args.quality_report.read_text(encoding="utf-8"))
    if (evidence["status"] != "passed"
            or Path(evidence["source"]).resolve() != source
            or Path(quality["source"]).resolve() != source
            or evidence["policy_sha256"] != digest(policy)
            or evidence["quality_report_sha256"] != digest(args.quality_report)):
        raise ValueError("Evidence/source/policy mismatch")
    for path, expected in evidence["auxiliary_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Auxiliary changed since G1: {path}")
    base = source.parent / "download/base"
    with (base / "calendar.csv").open(encoding="utf-8-sig", newline="") as handle:
        calendar = [date.fromisoformat(r["trading_date"]) for r in csv.DictReader(handle)]
    with (base / "cb_info.csv").open(encoding="utf-8-sig", newline="") as handle:
        identities = [r["债券代码"].zfill(6) for r in csv.DictReader(handle)]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate identities")
    dirs = sorted(d for d in (source / "day01/data/bar").iterdir()
                  if d.is_dir() and d.name.split(".")[0] in set(identities))
    files = quality["source_files"]
    expected_dirs = {str(Path(r["path"]).parent) for r in files}
    if {str(d.relative_to(source)) for d in dirs} != expected_dirs:
        raise ValueError("Universe changed since G1")
    hashes = {r["path"]: r["sha256"] for r in files}
    code = {str(p.relative_to(project)): digest(p) for p in sorted(
        [*project.glob("src/**/*.py"), Path(__file__).resolve(),
         project / "pyproject.toml", project / "uv.lock", policy, split_path])}
    input_manifest = {
        "source": str(source), "source_files": files,
        "auxiliary_sha256": evidence["auxiliary_sha256"], "code_sha256": code,
        "reader_report_sha256": digest(args.reader_report),
        "quality_report_sha256": digest(args.quality_report),
        "data_policy": policy.read_text(encoding="utf-8"), "split_config": split_path.read_text(encoding="utf-8"),
        "return_precision": "decimal 34 significant digits, ROUND_HALF_EVEN",
        "ordering": "instrument directory lexical order, then decision date",
    }
    out_root = validate_output_path(Path("artifacts/datasets"), project_root=project,
                                    source_root=source.parent)
    run_id = "g2-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = validate_output_path(out_root / run_id, project_root=project,
                               source_root=source.parent)
    if not out.is_relative_to((project / "artifacts").resolve()):
        raise ValueError("Output outside artifacts")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "manifest.json", {"status": "running", "inputs": input_manifest})
    try:
        counts: Counter[str] = Counter()
        samples_hash = hashlib.sha256()
        decisions_hash = hashlib.sha256()
        split_counts: Counter[str] = Counter()
        coverage = []
        sample_count = decision_count = 0
        seen_ids: set[str] = set()
        sample_schema = pa.schema([
            ("sample_id", pa.string()), ("instrument_id", pa.string()),
            ("context_start", pa.date32()), ("window_start", pa.date32()),
            ("window_end", pa.date32()),
            ("decision_time", pa.timestamp("us", tz="Asia/Shanghai")),
            ("label_start", pa.date32()), ("label_end", pa.date32()),
            ("label_available_at", pa.timestamp("us", tz="Asia/Shanghai")),
            ("horizon", pa.int64()), ("forward_return", pa.string()),
            ("label", pa.int64()), ("split", pa.string()),
            ("start_close", pa.string()), ("end_close", pa.string()),
            ("image_row", pa.int64()),
        ])
        decision_schema = pa.schema([
            ("instrument_id", pa.string()), ("decision_date", pa.date32()),
            ("candidate_split", pa.string()), ("reason", pa.string()),
        ])
        with pq.ParquetWriter(out / "samples.tmp", sample_schema) as sw, (
            pq.ParquetWriter(out / "decisions.tmp", decision_schema)
        ) as dw:
            for directory in dirs:
                candidates = list(directory.glob("*.parquet"))
                if len(candidates) != 1:
                    raise ValueError("File count changed")
                path = candidates[0]
                expected = hashes[str(path.relative_to(source))]
                if digest(path) != expected:
                    raise ValueError(f"Source changed: {path}")
                result = build_samples(read_instrument(directory), calendar, partitions)
                if digest(path) != expected:
                    raise ValueError(f"Source changed during read: {path}")
                rows = []
                for sample in result.samples:
                    if sample.sample_id in seen_ids:
                        raise ValueError("Duplicate sample ID")
                    seen_ids.add(sample.sample_id)
                    row = asdict(sample)
                    row["image_row"] = sample_count
                    sample_count += 1
                    split_counts[sample.split] += 1
                    samples_hash.update(canonical(row) + b"\n")
                    rows.append(row)
                if rows:
                    sw.write_table(pa.Table.from_pylist(rows, schema=sample_schema))
                decisions = [asdict(d) for d in result.decisions]
                for row in decisions:
                    decisions_hash.update(canonical(row) + b"\n")
                dw.write_table(pa.Table.from_pylist(decisions, schema=decision_schema))
                decision_count += len(decisions)
                counts.update(exclusion_summary(result.decisions))
                coverage.append({"instrument": directory.name,
                                 "missing_between_first_last": result.coverage_missing_sessions,
                                 "candidates": len(decisions), "retained": len(rows)})
        for path, expected in evidence["auxiliary_sha256"].items():
            if digest(Path(path)) != expected:
                raise ValueError("Auxiliary changed during build")
        if any(digest(project / p) != h for p, h in code.items()):
            raise ValueError("Code/config changed during build")
        if set(split_counts) != {"train", "validation", "test"}:
            raise ValueError("Empty split: protocol review required")
        if sum(counts.values()) != decision_count or (
            sum(n for k, n in counts.items() if k.endswith("|retained")) != sample_count
        ):
            raise ValueError("Accounting failed")
        summary = {"candidate_rows": decision_count, "retained": sample_count,
                   "split_counts": dict(split_counts), "year_split_reason": dict(sorted(counts.items())),
                   "coverage": coverage, "samples_logical_sha256": samples_hash.hexdigest(),
                   "decisions_logical_sha256": decisions_hash.hexdigest()}
        comparison = None
        if args.compare:
            old = json.loads((args.compare / "manifest.json").read_text(encoding="utf-8"))
            old_summary = json.loads((args.compare / "summary.json").read_text(encoding="utf-8"))
            if any(digest(args.compare / name) != expected for name, expected in old["artifacts_sha256"].items()):
                raise ValueError("Prior artifact hash mismatch")
            if old["status"] != "completed" or old["inputs"] != input_manifest:
                raise ValueError("Rebuild requires identical inputs/code/config")
            if old_summary != summary:
                raise ValueError("Rebuild differs")
            comparison = {"against": str(args.compare.resolve()), "exact_match": True}
        for name in ("samples", "decisions"):
            (out / f"{name}.tmp").replace(out / f"{name}.parquet")
        write_json(out / "summary.json", summary)
        write_json(out / "manifest.json", {
            "status": "completed", "created_at": datetime.now(UTC).isoformat(),
            "inputs": input_manifest, "rebuild": comparison,
            "environment": {"python": sys.version, "platform": platform.platform(),
                            "pyarrow": pa.__version__},
            "artifacts_sha256": {name: digest(out / name) for name in (
                "samples.parquet", "decisions.parquet", "summary.json")},
            "scope": "G2 sample index only; no fitting, images, training or test evaluation",
        })
        print(json.dumps({"output": str(out), "retained": sample_count,
                          "splits": dict(split_counts), "rebuild": comparison}))
    except Exception as exc:
        write_json(out / "manifest.json", {"status": "failed", "inputs": input_manifest,
                                          "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()