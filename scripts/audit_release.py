"""Release checks without training, inference, or access to the external catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = {
    "G5": "artifacts/runs/formal-20260912T083408279881Z",
    "G8": "artifacts/research/G8/g8-20260915T032338487327Z",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-evidence", action="store_true")
    args = parser.parse_args()
    result = {"status": "passed", "checks": {}}
    if args.local_evidence:
        for gate, relative in FROZEN.items():
            run = ROOT / relative
            manifest = read(run / "manifest.json")
            entries = manifest["artifacts_sha256"]
            for name, digest in entries.items():
                path = run / name.replace("\\", "/")
                if not path.resolve().is_relative_to(run.resolve()) or sha(path) != digest:
                    raise ValueError(f"Artifact mismatch: {gate}/{name}")
            plan = read(run / ("frozen.json" if gate == "G5" else "plan.json"))
            for name, digest in plan["code_sha256"].items():
                if sha(ROOT / name.replace("\\", "/")) != digest:
                    raise ValueError(f"Frozen code mismatch: {name}")
            result["checks"][gate] = {
                "artifacts": len(entries),
                "frozen_code": len(plan["code_sha256"]),
            }
    public = read(ROOT / "reports/cb-v1/results.json")
    assert public["scope"] == "convertible_bond_method_transfer"
    files = (
        subprocess.check_output(["git", "ls-files", "--cached", "-z"], cwd=ROOT)
        .decode()
        .split("\0")
    )
    files = [name for name in files if name]
    if not files:
        raise ValueError("Stage publication files before release audit")
    forbidden = {
        "artifacts",
        ".venv",
        ".uv-cache",
        ".tools",
        "Reference_repository",
        ".git",
        "__pycache__",
    }
    secret = re.compile(
        r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
    )
    for name in files:
        path = ROOT / name
        if (
            forbidden.intersection(path.relative_to(ROOT).parts)
            or path.suffix.lower()
            in {".pdf", ".pt", ".parquet", ".npy", ".npz", ".zip", ".exe", ".pyc"}
            or path.name.startswith(".env")
        ):
            raise ValueError(f"Forbidden publication file: {name}")
        if path.stat().st_size > 5 * 1024 * 1024:
            raise ValueError(f"Unexpected large publication file: {name}")
        if secret.search(path.read_text(encoding="utf-8-sig")):
            raise ValueError(f"Potential secret in {name}; content suppressed")
    for name in [n for n in files if n.endswith(".md")]:
        path = ROOT / name
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8-sig")):
            target = target.split("#")[0].strip("<>")
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (path.parent / target).resolve()
            if (
                not resolved.is_relative_to(ROOT)
                or resolved.relative_to(ROOT).as_posix() not in files
            ):
                raise ValueError(f"Nonpublished Markdown link: {name} -> {target}")
    result["checks"]["publication_files"] = len(files)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
