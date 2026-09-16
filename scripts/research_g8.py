"""Run the frozen G8 panel once; abort on failure, no retries or intermediate reveal."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from cb_image_alpha.artifacts import read_json, sha256, verify_files, write_json

PROJECT = Path(__file__).resolve().parents[1]
ORIGINAL = PROJECT / "artifacts/runs/formal-20260912T083408279881Z"


def validate(run: Path) -> None:
    plan = read_json(run / "plan.json")
    verify_files(PROJECT, plan["code_sha256"])
    verify_files(ORIGINAL, plan["inputs_sha256"])
    frozen = read_json(ORIGINAL / "frozen.json")
    for path, expected in frozen["auxiliary_sha256"].items():
        if sha256(Path(path)) != expected:
            raise ValueError("Auxiliary input changed")
    if (run / "preparation.json").exists():
        info = read_json(run / "preparation.json")
        verify_files(run, info["files"])
        for path, expected in info["sources"].items():
            if sha256(Path(path)) != expected:
                raise ValueError("Source changed")


def worker(run: Path, stage: str) -> None:
    import numpy as np
    import psutil
    import pyarrow as pa
    import torch

    config = read_json(run / "plan.json")["config"]
    psutil.Process().cpu_affinity(psutil.Process().cpu_affinity()[: config["cpu_threads"]])
    torch.set_num_threads(config["cpu_threads"])
    pa.set_cpu_count(config["cpu_threads"])
    pa.set_io_thread_count(config["cpu_threads"])
    validate(run)
    if stage == "prepare":
        from g8_data import prepare

        write_json(
            run / "environment.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pyarrow": pa.__version__,
                "psutil": psutil.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "lock_sha256": sha256(PROJECT / "uv.lock"),
            },
        )
        prepare(run)
    elif stage == "logistic":
        from g8_models import logistic_stage

        logistic_stage(run)
    elif stage == "finalize":
        pass
    elif stage == "evaluate":
        from g8_evaluate import evaluate

        evaluate(run)
    else:
        from g8_models import cnn_stage

        _, year, mode = stage.split("-")
        cnn_stage(run, int(year), mode)
    validate(run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--stage")
    args = parser.parse_args()
    if args.worker:
        run = args.worker.resolve(strict=True)
        if not run.is_relative_to((PROJECT / "artifacts/research/G8").resolve()):
            raise ValueError("Worker output boundary")
        worker(run, args.stage)
        return
    gate = PROJECT / "artifacts/research/G8"
    gate.mkdir(parents=True, exist_ok=True)
    if list(gate.glob("*/plan.json")):
        raise ValueError("G8 attempt exists; no automatic retry allowed")
    config = tomllib.loads((PROJECT / "configs/research/g8.toml").read_text(encoding="utf-8-sig"))
    run = gate / datetime.now(UTC).strftime("g8-%Y%m%dT%H%M%S%fZ")
    run.mkdir()
    frozen = read_json(ORIGINAL / "frozen.json")
    original = read_json(ORIGINAL / "manifest.json")
    canonical = {
        name.replace(chr(92), "/"): value for name, value in original["artifacts_sha256"].items()
    }
    inputs = {
        name: value
        for name, value in canonical.items()
        if name.startswith(("train/", "validation/"))
        or name in ("frozen.json", "config.json", "environment.json")
    }
    code = {
        **frozen["code_sha256"],
        **{
            str(path.relative_to(PROJECT)): sha256(path)
            for path in [
                *PROJECT.glob("scripts/g8_*.py"),
                PROJECT / "scripts/research_g8.py",
                PROJECT / "configs/research/g8.toml",
            ]
        },
    }
    write_json(
        run / "plan.json",
        {
            "scope": "development",
            "protocol": "research-v2/G8",
            "original": str(ORIGINAL),
            "config": config,
            "code_sha256": code,
            "inputs_sha256": inputs,
            "maximum_logistic_fits": 6,
            "maximum_cnn_fits": 6,
            "outer_reveal": "after all3folds; no intermediate model selection",
            "comparisons": ["U-Constant", "Q-Constant", "CNN-Constant", "U-Q", "CNN-Q"],
            "calendar_blocks": "full outer-year market calendar, zero-eligible days retained in blocks; date-equal loss excludes zero-count days",
        },
    )
    write_json(run / "manifest.json", {"status": "running", "gate": "G8"})
    ledger = []
    used = {"cpu": 0.0, "gpu": 0.0}
    stages = (
        [("prepare", "cpu"), ("logistic", "cpu")]
        + [
            (f"cnn-{year}-{mode}", "gpu")
            for year in config["fold_years"]
            for mode in ("inner", "final")
        ]
        + [("evaluate", "cpu"), ("finalize", "cpu")]
    )
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(config["cpu_threads"])
    os.environ["PYTHONPATH"] = str(PROJECT / "src")
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from run_formal import directory_bytes, supervise

    print({"event": "started", "run": str(run)}, flush=True)
    try:
        for stage, kind in stages:
            remaining = config[kind + "_seconds"] - used[kind]
            seconds = min(remaining, config["single_cnn_seconds"]) if kind == "gpu" else remaining
            if seconds <= 0:
                raise RuntimeError("Cumulative resource time exhausted")
            entry = {
                "stage": stage,
                "kind": kind,
                "status": "running",
                "allowed_seconds": seconds,
                "started_at": datetime.now(UTC).isoformat(),
            }
            ledger.append(entry)
            write_json(run / "attempt-ledger.json", ledger)
            phase_start = time.monotonic()
            try:
                report = supervise(
                    run,
                    stage,
                    seconds,
                    {
                        "ram_limit_bytes": config["ram_bytes"],
                        "artifact_limit_bytes": config["artifact_bytes"],
                    },
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        str(run),
                        "--stage",
                        stage,
                    ],
                )
            except BaseException:
                entry["status"] = "failed"
                report_path = run / f"{stage}-supervision.json"
                if report_path.exists():
                    report = read_json(report_path)
                    used[kind] += report["seconds"]
                    entry["report"] = report
                else:
                    elapsed = time.monotonic() - phase_start
                    used[kind] += elapsed
                    entry["unreported_failure_seconds"] = elapsed
                write_json(run / "attempt-ledger.json", ledger)
                raise
            used[kind] += report["seconds"]
            entry.update({"status": "completed", "report": report})
            write_json(run / "attempt-ledger.json", ledger)
            print(
                {"event": "stage_completed", "stage": stage, "seconds": report["seconds"]},
                flush=True,
            )
        finalize_start = time.monotonic()
        hashes = {}
        for path in run.rglob("*"):
            if path.is_file() and path != run / "manifest.json":
                hashes[str(path.relative_to(run))] = sha256(path)
            if used["cpu"] + time.monotonic() - finalize_start > config["cpu_seconds"]:
                used["cpu"] += time.monotonic() - finalize_start
                raise RuntimeError("Final fingerprint CPU budget exhausted")
        used["cpu"] += time.monotonic() - finalize_start
        if directory_bytes(run) > config["artifact_bytes"]:
            raise RuntimeError("Final artifact cap")
        write_json(
            run / "manifest.json",
            {
                "status": "completed",
                "scope": "development",
                "gate": "G8",
                "acceptance": "pending independent audit",
                "used_seconds": used,
                "artifacts_sha256": hashes,
            },
        )
        print({"event": "completed", "run": str(run)}, flush=True)
    except BaseException as exc:
        write_json(
            run / "manifest.json",
            {
                "status": "failed",
                "gate": "G8",
                "error": str(exc),
                "used_seconds": used,
                "outer_results_not_revealed": not (run / "results.json").exists(),
            },
        )
        raise


if __name__ == "__main__":
    main()
