"""Read a bounded catalog sample without assigning unverified financial semantics."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq


def encode(value: object) -> object:
    if isinstance(value, bytes):
        raw = int.from_bytes(value, byteorder="little", signed=True)
        return {"hex": value.hex(), "bytes": len(value), "little_signed_int": str(raw),
                "candidate_div_1e16": raw / 10**16, "candidate_div_1e9": raw / 10**9}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    source = args.source.resolve(strict=True)
    output = project / "artifacts" / "data_audit"
    resolved_output = output.resolve()
    if (not resolved_output.is_relative_to(project)
            or resolved_output.is_relative_to(source)
            or source.is_relative_to(resolved_output)):
        raise ValueError("Unsafe audit output path")
    report: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(), "source": str(source),
        "interpretation": "candidate scales only; not a normalized dataset",
    }
    files = []
    for code in ("110031", "113001", "118000", "118001", "123001", "127001"):
        for directory in sorted((source / "day01/data/bar").glob(f"{code}.*")):
            for path in sorted(directory.glob("*.parquet")):
                stat_before = path.stat()
                data = path.read_bytes()
                parquet = pq.ParquetFile(path)
                table = parquet.read()
                selected = sorted({0, table.num_rows // 2, table.num_rows - 1})
                rows = []
                for i in selected:
                    if i < 0:
                        continue
                    row = table.slice(i, 1).to_pylist()[0]
                    converted = {key: encode(value) for key, value in row.items()}
                    for key in ("ts_event", "ts_init"):
                        if isinstance(row.get(key), int):
                            converted[key + "_as_utc"] = datetime.fromtimestamp(
                                row[key] / 10**9, tz=UTC
                            ).isoformat()
                    rows.append({"row": i, "values": converted})
                metadata = {
                    key.decode("utf-8", errors="replace"): value.decode("utf-8", errors="replace")
                    for key, value in (parquet.schema_arrow.metadata or {}).items()
                }
                after = path.stat()
                if (stat_before.st_size, stat_before.st_mtime_ns) != (
                    after.st_size, after.st_mtime_ns
                ):
                    raise RuntimeError(f"Source changed during inspection: {path}")
                files.append({
                    "path": str(path.relative_to(source)), "sha256": hashlib.sha256(data).hexdigest(),
                    "rows": parquet.metadata.num_rows, "schema": str(parquet.schema_arrow),
                    "metadata": metadata, "samples": rows,
                })
    report["files"] = files
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = output / f"catalog-sample-{stamp}.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(target), "file_count": len(files)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
