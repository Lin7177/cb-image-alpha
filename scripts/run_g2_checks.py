"""Mechanical G2 build/rebuild/verification runner with persistent exit codes."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    source = args.source.resolve(strict=True)
    out = (project / "artifacts" / (
        "g2-check-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )).resolve()
    if (not out.is_relative_to(project) or out.is_relative_to(source.parent)
            or source.parent.is_relative_to(out)):
        raise ValueError("Unsafe evidence output")
    out.mkdir(exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(project / "src"), PYTHONIOENCODING="utf-8")
    env.pop("PYTHONOPTIMIZE", None)
    base = [sys.executable, "scripts/build_sample_index.py", "--source", str(source),
            "--reader-report", "artifacts/data_audit/reader-validation-20260911T062403608042Z.json",
            "--quality-report", "artifacts/data_audit/quality-20260911T053105796931Z.json"]
    steps = []

    def run(name: str, command: list[str]) -> dict:
        result = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True,
                                encoding="utf-8", check=False)
        (out / f"{name}.stdout.log").write_text(result.stdout, encoding="utf-8")
        (out / f"{name}.stderr.log").write_text(result.stderr, encoding="utf-8")
        steps.append({"name": name, "command": command, "exit_code": result.returncode})
        (out / "steps.json").write_text(json.dumps(steps, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{name} failed: {result.stderr}")
        return json.loads(result.stdout)

    first = run("build", base)
    second = run("rebuild", [*base, "--compare", first["output"]])
    for name, record in (("verify-first", first), ("verify-rebuild", second)):
        run(name, [sys.executable, "scripts/verify_sample_index.py", record["output"]])
    print(json.dumps({"status": "passed", "evidence": str(out),
                      "first": first, "second": second}, ensure_ascii=False))


if __name__ == "__main__":
    main()