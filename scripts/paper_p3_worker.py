"""P3 worker entry point. Called by the supervised CLI, never by reference code."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_replication.p3_checkpoint import (
    GateError,
    committed,
    durable_json,
    exact,
    read_json,
)
from paper_replication.p3_data import CONFIG, code_fingerprints, environment, real, synthetic
from paper_replication.p3_engine import configure, probe, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    output = Path(request["output"])
    result_path = args.request.with_suffix(".result.json")
    try:
        if request["code"] != code_fingerprints():
            raise GateError("Worker code fingerprint mismatch")
        synthetic_mode = request["synthetic"]
        if not synthetic_mode:
            authorization = read_json(Path(request["artifact_root"]) / "ready.json")
            if (
                authorization["status"] != "READY_TO_TRAIN"
                or authorization["code"] != request["code"]
            ):
                raise GateError("Real training readiness gate absent")
        if not synthetic_mode:
            from paper_replication.p3_checkpoint import digest
            from paper_replication.p3_data import verify_inputs

            root = Path(request["artifact_root"])
            preflight = root / "preflight/acceptance.json"
            if (
                authorization["preflight_sha256"] != digest(preflight)
                or read_json(preflight)["status"] != "PASS"
                or authorization["independent_then_cross"] is not True
                or set(authorization["reviewers"])
                != {"paper_research_audit", "paper_engineering_audit"}
            ):
                raise GateError("Audited preflight gate absent")
            if (
                request["run"] != "paper-p3-training-v1"
                or request["member"] not in range(5)
                or output.resolve() != (root / "batch" / f"member-{request['member']}").resolve()
                or request["device"] != "cuda"
                or request.get("resume", False)
            ):
                raise GateError("Unapproved real run/member/device or continuation")
            if verify_inputs() != request["input"]:
                raise GateError("Worker input fingerprints changed")
        train, validation, meta = synthetic() if synthetic_mode else real(request["member"])
        device = request["device"]
        configure(device, meta["model_seed"])
        identity = {
            "run": request["run"],
            "member": request["member"],
            "device": device,
            "code": request["code"],
            "environment": environment(device),
            "input": request["input"],
            "protocol": str(CONFIG.name),
        }
        if request.get("action") == "probe":
            import torch

            from cb_image_alpha.models.paper_cnn import PaperCNN

            paths = committed(output, identity)
            last = torch.load(paths[-1] / "state.pt", weights_only=False, map_location="cpu")
            best = paths[last["best_epoch"]]
            state = torch.load(best / "state.pt", weights_only=False, map_location="cpu")
            predictions = torch.load(
                best / "predictions.pt", weights_only=False, map_location="cpu"
            )
            from paper_replication.p3_engine import coverage

            coverage(predictions["sample_id"].numpy(), validation.ids)
            exact(predictions["sample_id"], torch.from_numpy(validation.ids.copy()))
            exact(predictions["label"], torch.from_numpy(validation.labels.copy()))
            if not torch.isfinite(predictions["probability"]).all():
                raise GateError("Nonfinite sealed probability")
            model = PaperCNN().to(device)
            model.load_state_dict(state["model"])
            actual = probe(model, validation, device, torch.Generator(), torch.Generator())
            exact(actual, predictions["probability"][:128], "fresh_process_best_probe")
            result = {
                "status": "passed",
                "best_epoch": last["best_epoch"],
                "best_loss": last["best_loss"],
                "rows": len(validation),
                "probe_rows": 128,
                "best_directory": str(best),
            }
        else:
            result = run(
                output,
                Path(request["artifact_root"]),
                train,
                validation,
                meta,
                identity,
                device,
                epochs=3 if synthetic_mode else 50,
                resume=request.get("resume", False),
                interrupt_after=request.get("interrupt_after"),
                fault=request.get("fault", ""),
                synthetic=synthetic_mode,
            )
        durable_json(result_path, result)
        return 0
    except BaseException as error:  # noqa: BLE001 -- terminal boundary preserves failure evidence
        durable_json(
            result_path,
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
