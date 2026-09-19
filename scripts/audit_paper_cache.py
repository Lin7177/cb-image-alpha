"""Inventory the capped official CACHE_DIR without executing reference code.

The publisher's README says CACHE_DIR keeps only the first and last 50 rows.
This script records what configurations are present for sanity checks while
making it impossible to mistake those files for a complete rerun input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path


MODEL_RE = re.compile(r"(?:^|/)(cnn1d|linear)_i(5|20|60)r(5|20|60)_(.+)\.csv$")


def _count_rows(zf: zipfile.ZipFile, name: str) -> tuple[int, list[str]]:
    with zf.open(name, "r") as fh:
        raw = fh.read()
    text = raw.decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return 0, []
    return max(0, len(rows) - 1), rows[0]


def audit(cache: Path) -> dict:
    cache = cache.resolve()
    cache_sha = hashlib.sha256(cache.read_bytes()).hexdigest()
    models: list[dict] = []
    other_members: list[str] = []
    with zipfile.ZipFile(cache) as zf:
        for info in sorted(zf.infolist(), key=lambda item: item.filename):
            if info.is_dir():
                continue
            match = MODEL_RE.search(info.filename.lower())
            if not match:
                other_members.append(info.filename)
                continue
            rows, columns = _count_rows(zf, info.filename)
            models.append(
                {
                    "member": info.filename,
                    "model": match.group(1),
                    "config": f"I{match.group(2)}/R{match.group(3)}",
                    "signal": match.group(4),
                    "rows_in_capped_member": rows,
                    "columns": columns,
                    "file_size": info.file_size,
                    "compressed_size": info.compress_size,
                }
            )

    configs = sorted({item["config"] for item in models})
    return {
        "schema_version": "paper-cache-audit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache": str(cache),
        "cache_sha256": cache_sha,
        "central_directory_only_for_inventory": True,
        "payload_read_for_csv_headers_and_row_counts": True,
        "reference_code_executed": False,
        "official_readme_scope": "CACHE_DIR rows are capped to first 50 and last 50",
        "full_reproduction_input": False,
        "configs_with_cached_model_outputs": configs,
        "model_member_count": len(models),
        "model_members": models,
        "other_member_count": len(other_members),
        "other_members": other_members,
        "interpretation": (
            "Cached files may support formula and naming sanity checks only. "
            "They cannot reconstruct full-sample tables, OOS metrics, or a "
            "complete portfolio history without uncapped inputs."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="official CACHE_DIR.zip")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/paper-v1/paper-cache-audit.json"),
        help="manifest output path",
    )
    args = parser.parse_args()
    result = audit(args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model_member_count": result["model_member_count"],
                "configs": result["configs_with_cached_model_outputs"],
                "full_reproduction_input": result["full_reproduction_input"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
