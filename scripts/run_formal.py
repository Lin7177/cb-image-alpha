"""Supervise one frozen formal experiment. No automatic retries or model search."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import psutil

from cb_image_alpha.artifacts import read_json, sha256, verify_files, write_json
from cb_image_alpha.paths import validate_output_path


def directory_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass  # Atomic checkpoint/shard rename between enumeration and stat.
    return total


def stop_tree(process: subprocess.Popen) -> None:
    try:
        root = psutil.Process(process.pid)
        for child in root.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        root.kill()
    except psutil.NoSuchProcess:
        pass
    process.wait(timeout=30)


def supervise(run: Path, stage: str, seconds: int, config: dict,
              command: list[str] | None = None) -> dict:
    command = command or [sys.executable, str(Path(__file__).resolve()), "--run", str(run),
                          "--stage", stage]
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONIOENCODING="utf-8")
    start = time.monotonic()
    peak = 0
    reason = None
    with (run / f"{stage}.stdout.log").open("w", encoding="utf-8") as stdout, (
        run / f"{stage}.stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            while process.poll() is None:
                working_set = 0
                try:
                    parent = psutil.Process(process.pid)
                    for member in [parent, *parent.children(recursive=True)]:
                        try:
                            working_set += member.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass
                peak = max(peak, working_set)
                size = directory_bytes(run)
                elapsed = time.monotonic() - start
                if working_set > config["ram_limit_bytes"]:
                    reason = "ram_limit"
                elif size > config["artifact_limit_bytes"]:
                    reason = "artifact_limit"
                elif elapsed > seconds:
                    reason = "wall_time_limit"
                if reason:
                    stop_tree(process)
                    break
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            stop_tree(process)
            raise
    if directory_bytes(run) > config["artifact_limit_bytes"]:
        reason = reason or "artifact_limit"
    if time.monotonic() - start > seconds:
        reason = reason or "wall_time_limit"
    report = {"stage": stage, "command": command, "exit_code": process.returncode,
              "seconds": time.monotonic() - start, "peak_working_set_bytes": peak,
              "artifact_bytes": directory_bytes(run), "failure": reason}
    write_json(run / f"{stage}-supervision.json", report)
    if process.returncode or reason:
        raise RuntimeError(f"{stage} failed: {report}")
    return report


def validate_frozen(run: Path) -> None:
    frozen = read_json(run / "frozen.json")
    project = Path(__file__).resolve().parents[1]
    source = Path(frozen["source"])
    validate_output_path(run, project_root=project, source_root=source.parent)
    if not run.resolve().is_relative_to((project / "artifacts/runs").resolve()):
        raise ValueError("Formal run outside artifacts")
    verify_files(project, frozen["code_sha256"])
    verify_files(Path(frozen["parent"]), frozen["parent_files"])
    for path, expected in frozen["auxiliary_sha256"].items():
        if sha256(Path(path)) != expected:
            raise ValueError("Auxiliary changed")
    if sha256(run / "config.json") != frozen["config_sha256"]:
        raise ValueError("Frozen config changed")


def worker(run: Path, stage: str) -> None:
    validate_frozen(run)
    if stage == "prepare":
        import platform

        import numpy as np
        import torch

        from cb_image_alpha.training.formal_data import prepare_split
        config = read_json(run / "config.json")
        torch.cuda.set_per_process_memory_fraction(config["gpu_memory_fraction"])
        write_json(run / "environment.json", {
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "numpy": np.__version__, "psutil": psutil.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(),
            "lock_sha256": sha256(Path(__file__).resolve().parents[1] / "uv.lock"),
            "deterministic_algorithms": True, "cublas": os.environ.get("CUBLAS_WORKSPACE_CONFIG")})
        prepare_split(run, "train")
        prepare_split(run, "validation")
    elif stage == "train":
        from cb_image_alpha.training.formal_engine import train
        train(run)
    elif stage == "evaluate":
        from cb_image_alpha.training.formal_engine import evaluate
        evaluate(run)
    else:
        raise ValueError("Unknown stage")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--stage", choices=["prepare", "train", "evaluate"])
    args = parser.parse_args()
    if args.stage:
        if args.run is None:
            raise ValueError("--run required")
        worker(args.run, args.stage)
        return
    if args.dataset is None:
        raise ValueError("--dataset required")
    project = Path(__file__).resolve().parents[1]
    parent = args.dataset.resolve(strict=True)
    previous = read_json(parent / "manifest.json")
    if previous["status"] != "completed":
        raise ValueError("Incomplete parent dataset")
    if sha256(parent / "samples.parquet") != "8c5afdf43c6951354dde91614219446232466f6d768f824adf652102f87ad357":
        raise ValueError("Not the approved G2 sample index")
    config_path = project / "configs/formal.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if config["scope"] != "formal" or config["version"] != "g5-v1":
        raise ValueError("Unsupported formal configuration")
    source = Path(previous["inputs"]["source"])
    run = validate_output_path(
        Path("artifacts/runs") / datetime.now(UTC).strftime("formal-%Y%m%dT%H%M%S%fZ"),
        project_root=project, source_root=source.parent)
    if not run.is_relative_to((project / "artifacts/runs").resolve()):
        raise ValueError("Run output boundary")
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "manifest.json", {"scope": "formal", "status": "running"})
    write_json(run / "config.json", config)
    frozen = {
        "scope": "formal", "parent": str(parent), "source": str(source),
        "config_sha256": sha256(run / "config.json"),
        "parent_files": {**previous["artifacts_sha256"], "manifest.json": sha256(parent / "manifest.json")},
        "source_files": {r["path"]: r["sha256"] for r in previous["inputs"]["source_files"]},
        "auxiliary_sha256": previous["inputs"]["auxiliary_sha256"],
        "code_sha256": {str(p.relative_to(project)): sha256(p) for p in sorted([
            *project.glob("src/**/*.py"), Path(__file__).resolve(), config_path,
            project / "pyproject.toml", project / "uv.lock"])},
    }
    write_json(run / "frozen.json", frozen)
    try:
        validate_frozen(run)
        print(json.dumps({"scope": "formal", "run": str(run), "event": "frozen"}), flush=True)
        for stage, seconds in (("prepare", config["prepare_seconds"]),
                               ("train", config["train_seconds"]),
                               ("evaluate", config["evaluate_seconds"])):
            result = supervise(run, stage, seconds, config)
            print(json.dumps({"event": "stage_completed", **result}), flush=True)
        validate_frozen(run)
        for relative, expected in frozen["source_files"].items():
            if sha256(source / relative) != expected:
                raise ValueError("Source changed during formal run")
        if directory_bytes(run) > config["artifact_limit_bytes"]:
            raise ValueError("Final artifact budget exceeded")
        hashes = {str(p.relative_to(run)): sha256(p) for p in run.rglob("*")
                  if p.is_file() and p.name != "manifest.json"}
        # Include split manifests too; exclude only this root manifest.
        hashes.update({f"{split}/manifest.json": sha256(run / split / "manifest.json")
                       for split in ("train", "validation", "test")})
        write_json(run / "manifest.json", {
            "scope": "formal", "status": "completed", "artifacts_sha256": hashes,
            "completed_at": datetime.now(UTC).isoformat(),
            "acceptance": "pending independent G5 audit"})
        print(json.dumps({"event": "completed", "run": str(run)}), flush=True)
    except BaseException as exc:
        write_json(run / "manifest.json", {"scope": "formal", "status": "failed",
             "error": f"{type(exc).__name__}: {exc}", "test_accessed": (run / "test-access.json").exists()})
        raise


if __name__ == "__main__":
    main()