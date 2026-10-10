"""Check publishable Git files, paper snapshot arithmetic and frozen source hashes."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from paper_demo import PUBLIC, ROOT, digest, read, recalculate


def audit(root: Path = ROOT) -> dict:
    files = subprocess.check_output(["git", "ls-files", "--cached", "-z"], cwd=root)
    names = [n for n in files.decode("utf-8").split("\0") if n]
    forbidden = {"artifacts", ".venv", ".uv-cache", ".tools", "Reference_repository",
                 ".git", "__pycache__"}
    allowed_suffixes = {".py", ".toml", ".lock", ".md", ".json", ".csv", ".html", ".png", ""}
    secret = re.compile(
        r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|"
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
    )
    total = 0
    for name in names:
        path = root / name
        if (forbidden.intersection(Path(name).parts) or path.name.startswith(".env")
                or path.suffix.lower() not in allowed_suffixes):
            raise ValueError(f"Forbidden publication file: {name}")
        size = path.stat().st_size
        if size > 2 * 1024**2:
            raise ValueError(f"Unexpected large file: {name}")
        total += size
        if path.suffix.lower() == ".png":
            if not name.startswith("reports/paper-i20r20/figures/"):
                raise ValueError(f"PNG outside owned figure directory: {name}")
            continue
        text = path.read_text(encoding="utf-8-sig")
        if secret.search(text):
            raise ValueError(f"Potential credential in {name}; content suppressed")
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^)]+)\)", text):
                target = target.split("#")[0].strip("<>")
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                resolved = (path.parent / target).resolve()
                if (not resolved.is_relative_to(root.resolve())
                        or resolved.relative_to(root.resolve()).as_posix() not in names):
                    raise ValueError(f"Nonpublished Markdown link: {name} -> {target}")
    if "scripts/paper_p4_evaluate.py" in names:
        raise ValueError("Pre-existing untracked evaluator must not be included by this release")
    cfg = root / "configs/paper/p4_reval_v1.toml"
    import tomllib
    frozen = tomllib.loads(cfg.read_text(encoding="utf-8"))
    for path, expected in [(root / "src/cb_image_alpha/models/paper_cnn.py", frozen["model_source_sha256"]),
                           (root / "uv.lock", frozen["environment_lock_sha256"])]:
        if digest(path) != expected:
            raise ValueError(f"Frozen source changed: {path.name}")
    published_sources = read(PUBLIC / "source_fingerprints.json")
    for name, expected in published_sources["sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f"Published core source changed: {name}")
    acceptance = read(PUBLIC / "ANALYSIS_ACCEPTANCE.json")
    for name, expected in acceptance["accepted_file_hashes"].items():
        if digest(PUBLIC / name) != expected:
            raise ValueError(f"Accepted analysis changed: {name}")
    value = recalculate(PUBLIC)
    return {"status": "PASS_PUBLICATION_CHECKS_ONLY", "tracked_files": len(names),
            "working_tree_file_bytes": total, "public_months": value["months"],
            "accepted_analysis_hashes": len(acceptance["accepted_file_hashes"]),
            "core_source_fingerprints_verified": len(published_sources["sha256"]),
            "frozen_model_and_lock_unchanged": True, "new_training": False,
            "raw_oos_reads": 0, "note": "Checks current Git-index file list and worktree bytes; no scientific acceptance change."}


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
