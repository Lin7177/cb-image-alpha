"""Audit which official image members are available for the paper's I/R grid.

This audit intentionally reads only the ZIP central directory.  It never
extracts, decodes, or executes reference-code payloads.  The output is a
scope manifest that makes data availability explicit before any new run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


CONFIGS = [(i, r) for i in (5, 20, 60) for r in (5, 20, 60)]


def _normalise(name: str) -> str:
    return name.replace("\\", "/").lower()


def _expected_tokens(i: int, r: int) -> tuple[str, ...]:
    # R determines the sampling frequency in the official implementation.
    # Keep multiple spellings because the public archive has changed folder
    # prefixes between releases; matching is still based on the complete
    # central-directory name, never on decoded content.
    frequency = {5: ("week", "5d_week"), 20: ("month", "20d_month"), 60: ("quarter", "60d_quarter")}[r]
    return (f"{i}d", *frequency)


def _matches(name: str, i: int, r: int) -> bool:
    n = _normalise(name)
    tokens = _expected_tokens(i, r)
    # Images and their label sidecars normally carry the same I/R/frequency
    # stem.  Requiring all tokens prevents Ret_5d/Ret_60d labels on a monthly
    # image from being misclassified as another official configuration.
    return all(token in n for token in tokens)


def audit(archive: Path) -> dict:
    archive = archive.resolve()
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as zf:
        names = sorted(info.filename for info in zf.infolist() if not info.is_dir())

    configs = []
    for i, r in CONFIGS:
        matches = [name for name in names if _matches(name, i, r)]
        configs.append(
            {
                "config": f"I{i}/R{r}",
                "i_days": i,
                "r_days": r,
                "expected_frequency": {5: "week", 20: "month", 60: "quarter"}[r],
                "member_count": len(matches),
                "members": matches[:200],
                "members_truncated": len(matches) > 200,
                "available": bool(matches),
                "classification": "available" if matches else "data_blocked",
            }
        )

    return {
        "schema_version": "paper-scope-audit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive),
        "archive_sha256": sha,
        "central_directory_only": True,
        "payload_decoded": False,
        "reference_code_executed": False,
        "configs": configs,
        "available_configs": [x["config"] for x in configs if x["available"]],
        "blocked_configs": [x["config"] for x in configs if not x["available"]],
        "interpretation": (
            "A configuration is available only when the central directory contains "
            "members carrying its image length and R-driven frequency tokens. "
            "Alternative return-label fields on another image frequency do not "
            "create an official I/R configuration."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="official image ZIP")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/paper-v1/paper-scope-audit.json"),
        help="manifest output path",
    )
    args = parser.parse_args()
    result = audit(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "available": result["available_configs"], "blocked": result["blocked_configs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
