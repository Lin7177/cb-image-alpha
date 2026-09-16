"""One user-authorized G8 continuation; original protocol/code remain frozen."""

import os
import sys
import time
from pathlib import Path

from research_g8 import PROJECT, validate, worker
from run_formal import supervise

from cb_image_alpha.artifacts import read_json, sha256, verify_files, write_json

RUN = PROJECT / "artifacts/research/G8/g8-20260915T032338487327Z"


def main():
    r = RUN
    if len(sys.argv) > 1:
        stage = sys.argv[1]
        recovery = read_json(r / "recovery-plan.json")
        verify_files(PROJECT, recovery["code_sha256"])
        verify_files(r, recovery["snapshot_sha256"])
        if sha256(r / "plan.json") != recovery["original_plan_sha256"]:
            raise ValueError("Original plan changed")
        if stage == "resume-2023-inner":
            import psutil
            import pyarrow as pa
            import torch
            from g8_resume_model import resume_cnn_stage

            psutil.Process().cpu_affinity(psutil.Process().cpu_affinity()[:4])
            torch.set_num_threads(4)
            pa.set_cpu_count(4)
            pa.set_io_thread_count(4)
            for name in ("inner-latest.pt", "inner-best.pt", "inner-history.json"):
                if sha256(r / "2023" / name) != sha256(r / "interruption-snapshot/2023" / name):
                    raise ValueError("Checkpoint changed after recovery snapshot")
            checkpoint = torch.load(
                r / "2023/inner-latest.pt", map_location="cpu", weights_only=True
            )
            if checkpoint["epoch"] + 1 != recovery["resume_epoch"]:
                raise ValueError("Unexpected recovery epoch")
            del checkpoint
            validate(r)
            resume_cnn_stage(r, 2023, "inner")
            validate(r)
        else:
            worker(r, stage)
        return
    if (r / "recovery-plan.json").exists():
        raise RuntimeError("Recovery already attempted; no automatic retry")
    cfg = read_json(r / "plan.json")["config"]
    snapshot = r / "interruption-snapshot"
    snapshot.mkdir()
    import shutil

    for name in (
        "manifest.json",
        "attempt-ledger.json",
        "2023/inner-latest.pt",
        "2023/inner-best.pt",
        "2023/inner-history.json",
    ):
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(r / name, dest)
    write_json(
        r / "recovery-plan.json",
        {
            "authorization": "User confirmed deliberate termination and authorized continuation",
            "original_plan_sha256": sha256(r / "plan.json"),
            "code_sha256": {
                str(p.relative_to(PROJECT)): sha256(p)
                for p in (Path(__file__).resolve(), PROJECT / "scripts/g8_resume_model.py")
            },
            "snapshot_sha256": {
                str(p.relative_to(r)): sha256(p) for p in snapshot.rglob("*") if p.is_file()
            },
            "unknown_stage_gpu_budget_charge": 2700,
            "charge_is_measured_time": False,
            "exception": "User-authorized continuation segment capped at 2700s; combined logical fit not claimed <=2700s; global 14400s cap retained",
            "resume_epoch": 8,
        },
    )
    original_ledger = read_json(r / "attempt-ledger.json")
    used = {
        kind: sum(
            e.get("report", {}).get("seconds", 0) for e in original_ledger if e["kind"] == kind
        )
        for kind in ("cpu", "gpu")
    }
    used["gpu"] += 2700
    ledger = []
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "4"
    os.environ["PYTHONPATH"] = str(PROJECT / "src")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    stages = [
        ("resume-2023-inner", "gpu"),
        ("cnn-2023-final", "gpu"),
        ("cnn-2024-inner", "gpu"),
        ("cnn-2024-final", "gpu"),
        ("evaluate", "cpu"),
        ("finalize", "cpu"),
    ]
    write_json(r / "manifest.json", {"status": "running", "gate": "G8", "recovery": "authorized"})
    try:
        for stage, kind in stages:
            limit = cfg[kind + "_seconds"] - used[kind]
            if kind == "gpu":
                limit = min(limit, cfg["single_cnn_seconds"])
            if limit <= 0:
                raise RuntimeError("Cumulative budget exhausted")
            entry = {"stage": stage, "kind": kind, "status": "running", "limit": limit}
            ledger.append(entry)
            write_json(r / "recovery-ledger.json", ledger)
            start = time.monotonic()
            try:
                report = supervise(
                    r,
                    "recovery-" + stage,
                    limit,
                    {
                        "ram_limit_bytes": cfg["ram_bytes"],
                        "artifact_limit_bytes": cfg["artifact_bytes"],
                    },
                    [sys.executable, str(Path(__file__).resolve()), stage],
                )
            except BaseException:
                used[kind] += time.monotonic() - start
                entry["status"] = "failed"
                write_json(r / "recovery-ledger.json", ledger)
                raise
            used[kind] += report["seconds"]
            entry.update(status="completed", report=report)
            write_json(r / "recovery-ledger.json", ledger)
            print({"stage_completed": stage, "seconds": report["seconds"]}, flush=True)
        start = time.monotonic()
        hashes = {}
        try:
            for p in r.rglob("*"):
                if p.is_file() and p != r / "manifest.json":
                    hashes[str(p.relative_to(r))] = sha256(p)
                if used["cpu"] + time.monotonic() - start > cfg["cpu_seconds"]:
                    raise RuntimeError("Fingerprint CPU budget exhausted")
        finally:
            used["cpu"] += time.monotonic() - start
        write_json(
            r / "manifest.json",
            {
                "status": "completed",
                "gate": "G8",
                "scope": "development",
                "acceptance": "pending independent audit",
                "budget_charged_seconds": used,
                "artifacts_sha256": hashes,
            },
        )
        print("completed", flush=True)
    except BaseException as exc:
        write_json(
            r / "manifest.json",
            {"status": "blocked", "gate": "G8", "error": str(exc), "budget_charged_seconds": used},
        )
        raise


if __name__ == "__main__":
    main()
