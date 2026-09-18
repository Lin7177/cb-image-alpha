"""Single-attempt P3 preflight and real-batch orchestration. No retry commands."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_replication.p3_checkpoint import GateError, digest, durable_json, exact, read_json
from paper_replication.p3_data import CONFIG, P2, ROOT, code_fingerprints, verify_inputs
from paper_replication.p3_supervision import supervise

OUTPUT = ROOT / "artifacts/paper-v1/p3-paper-training-v1"
WORKER = ROOT / "scripts/paper_p3_worker.py"


def request(
    root,
    name,
    output,
    *,
    synthetic=True,
    device="cpu",
    member=0,
    run_id="synthetic-fixed",
    inputs=None,
    **extra,
):
    value = {
        "run": run_id,
        "member": member,
        "synthetic": synthetic,
        "device": device,
        "output": str(output),
        "artifact_root": str(OUTPUT),
        "code": code_fingerprints(),
        "input": inputs or {"fixture": "binary-63017-256-128"},
        **extra,
    }
    path = root / (name + ".json")
    durable_json(path, value)
    return path


def invoke(path, *, expected_error=None):
    result = subprocess.run([sys.executable, str(WORKER), "--request", str(path)], check=False)
    record = read_json(path.with_suffix(".result.json"))
    if expected_error:
        if result.returncode != 1 or expected_error not in record.get("error", ""):
            raise GateError(f"Fault did not stop correctly: {path.name}: {record}")
    elif result.returncode != 0:
        raise GateError(f"Worker failure: {path.name}: {record}")
    return record


def compare_runs(a, b):
    import torch

    pa, pb = sorted(a.glob("epoch-*/COMMIT.json")), sorted(b.glob("epoch-*/COMMIT.json"))
    if len(pa) != 4 or len(pb) != 4:
        raise GateError("Preflight must complete exactly three committed epochs")
    for left, right in zip(pa, pb, strict=True):
        x = torch.load(left.parent / "state.pt", map_location="cpu", weights_only=False)
        y = torch.load(right.parent / "state.pt", map_location="cpu", weights_only=False)
        x.pop("ledger")
        y.pop("ledger")
        exact(x, y, f"epoch_{left.parent.name}")
        exact(
            torch.load(left.parent / "predictions.pt", weights_only=False),
            torch.load(right.parent / "predictions.pt", weights_only=False),
            "predictions",
        )


def preflight_worker():
    """The only formal preflight attempt; any unexpected failure ends execution."""
    import torch

    from paper_replication.p3_checkpoint import committed, load_latest
    from paper_replication.p3_data import authorized_image
    from paper_replication.p3_engine import coverage
    from paper_replication.state import ensemble_probabilities

    root = OUTPUT / "preflight"
    root.mkdir()
    checks = []
    try:
        for device in ("cpu", "cuda"):
            base, resumed = root / (device + "-continuous"), root / (device + "-resumed")
            invoke(request(root, device + "-continuous", base, device=device))
            invoke(request(root, device + "-interrupt", resumed, device=device, interrupt_after=1))
            invoke(request(root, device + "-resume", resumed, device=device, resume=True))
            compare_runs(base, resumed)
            checks.append(device + "_exact_model_BN_Adam_RNG_loader_history_probabilities")
        # A real partial training epoch is replayed, in a fresh process, from epoch 1.
        partial = root / "cpu-mid-epoch"
        invoke(request(root, "mid-interrupt", partial, fault="interrupt_during_epoch"))
        invoke(request(root, "mid-resume", partial, resume=True))
        compare_runs(root / "cpu-continuous", partial)
        checks.append("partial_epoch_exact_replay")
        for fault, message in [
            ("nonfinite_loss", "Nonfinite loss"),
            ("nonfinite_gradient", "Nonfinite gradient"),
            ("incomplete_commit", "injected incomplete_commit"),
        ]:
            target = root / fault
            invoke(request(root, fault, target, fault=fault), expected_error=message)
            if (target / "epoch-001/COMMIT.json").exists():
                raise GateError("Failed epoch was committed")
            if fault == "incomplete_commit":
                marker = read_json(target / "epoch-000/COMMIT.json")
                if len(committed(target, marker["identity"])) != 1:
                    raise GateError("Partial commit incorrectly authoritative")
            checks.append(fault + "_correctly_blocked")
        # Exercise the production loader of the *latest* committed epoch, with no fallback.
        corrupted = root / "corrupt-copy"
        shutil.copytree(root / "cpu-continuous", corrupted)
        marker = read_json(corrupted / "epoch-003/COMMIT.json")
        with (corrupted / "epoch-003/state.pt").open("ab") as stream:
            stream.write(b"corrupt")
        expect_gate(lambda: load_latest(corrupted, marker["identity"]), "fingerprint mismatch")
        checks.append("latest_corrupt_commit_no_fallback")
        base = root / "cpu-continuous"
        identity = read_json(base / "epoch-003/COMMIT.json")["identity"]
        expect_gate(lambda: load_latest(base, {**identity, "input": {"changed": True}}), "identity")
        import numpy as np

        expect_gate(lambda: coverage([1, 1], np.array([1, 2])), "Duplicate/missing")
        expect_gate(lambda: coverage([1], np.array([1, 2])), "Duplicate/missing")
        expect_gate(lambda: authorized_image(ROOT / "2001_images.dat"), "Forbidden OOS")
        checks.extend(["fingerprint_mismatch", "duplicate_and_missing_ids", "forbidden_OOS_path"])
        # The same supervisor must terminate a child on an injected zero resource cap.
        limited = supervise(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            root / "resource-injection",
            OUTPUT,
            30,
            rss_limit=0,
        )
        if limited["status"] != "failed" or limited["reason"] != "rss_tree_limit":
            raise GateError("Resource injection did not terminate child")
        checks.append("resource_cap_terminated_child")
        logits = [torch.tensor([[float(i), -float(i)], [0.0, 1.0]]) for i in range(5)]
        exact(ensemble_probabilities(logits), torch.stack([x.softmax(1) for x in logits]).mean(0))
        checks.append("five_equal_softmax_mean")
        durable_json(
            root / "acceptance.json",
            {
                "status": "PASS",
                "checks": checks,
                "code": code_fingerprints(),
                "real_fits": 0,
                "oos_sealed": True,
                "physical_power_loss_durability_proven": False,
            },
        )
        return 0
    except BaseException as error:  # noqa: BLE001 -- terminal boundary preserves failure evidence
        durable_json(
            root / "acceptance.json",
            {
                "status": "BLOCKED",
                "checks_completed": checks,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "code": code_fingerprints(),
                "real_fits": 0,
                "oos_sealed": True,
            },
        )
        traceback.print_exc()
        return 1


def expect_gate(action, contains):
    try:
        action()
    except GateError as error:
        if contains.lower() not in str(error).lower():
            raise
    else:
        raise GateError("Expected injected gate failure was not raised")


def preflight():
    OUTPUT.mkdir(exist_ok=False)
    if shutil.disk_usage(OUTPUT).free < 10 * 1024**3:
        raise GateError("Minimum free disk gate failed")
    durable_json(
        OUTPUT / "protocol-binding.json",
        {
            "protocol_sha256": digest(CONFIG),
            "authorization_sha256": digest(
                ROOT / "artifacts/paper-v1/p3-execution-authorization.json"
            ),
            "code": code_fingerprints(),
            "real_fits": 0,
        },
    )
    result = supervise(
        [sys.executable, str(Path(__file__).resolve()), "preflight-worker"],
        OUTPUT / "preflight-supervision",
        OUTPUT,
        1800,
    )
    if result["status"] != "passed":
        raise GateError(f"Preflight BLOCKED: {result}")


def train_batch():
    """Called only after externally recorded independent/cross acceptance."""
    ready = read_json(OUTPUT / "ready.json")
    if ready["status"] != "READY_TO_TRAIN" or ready["code"] != code_fingerprints():
        raise GateError("Implementation audit/READY gate failed")
    if (
        ready["preflight_sha256"] != digest(OUTPUT / "preflight/acceptance.json")
        or read_json(OUTPUT / "preflight/acceptance.json")["status"] != "PASS"
        or read_json(OUTPUT / "preflight-supervision/result.json")["status"] != "passed"
        or ready["independent_then_cross"] is not True
        or set(ready["reviewers"]) != {"paper_research_audit", "paper_engineering_audit"}
    ):
        raise GateError("Independent/cross preflight gate failed")
    if ready["protocol_sha256"] != digest(CONFIG):
        raise GateError("Protocol changed after review")
    authorization_path = ROOT / "artifacts/paper-v1/p3-execution-authorization.json"
    authorization = read_json(authorization_path)
    if not authorization["conditional_real_training_authorized"] or ready[
        "authorization_sha256"
    ] != digest(authorization_path):
        raise GateError("User authorization gate failed")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise GateError("Real start requires clean code snapshot")
    if shutil.disk_usage(OUTPUT).free < 10 * 1024**3:
        raise GateError("Minimum free disk gate failed")
    batch = OUTPUT / "batch"
    batch.mkdir()
    start = time.monotonic()
    results = []
    try:
        inputs = verify_inputs()
        snapshot = batch / "p2-input-snapshot"
        snapshot.mkdir()
        for path in P2.iterdir():
            if path.name in {
                "pool.npz",
                "members.json",
                "manifest.json",
                "exclusions.json",
                "config.json",
                "p2-acceptance.json",
            } or path.name.startswith("member-"):
                shutil.copy2(path, snapshot / path.name)
        code_snapshot = batch / "code"
        for relative, expected in code_fingerprints().items():
            target = code_snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
            if digest(target) != expected:
                raise GateError("Code snapshot mismatch")
        shutil.copy2(ROOT / "uv.lock", batch / "uv.lock")
        shutil.copy2(CONFIG, batch / CONFIG.name)
        durable_json(
            batch / "manifest.json",
            {
                "status": "started",
                "inputs": inputs,
                "code": code_fingerprints(),
                "commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "protocol_sha256": digest(CONFIG),
                "oos_sealed": True,
            },
        )
        for member in range(5):
            remaining = 36000 - (time.monotonic() - start)
            if remaining <= 0:
                raise GateError("Total member time exhausted")
            path = request(
                batch,
                f"member-{member}-request",
                batch / f"member-{member}",
                synthetic=False,
                device="cuda",
                member=member,
                run_id="paper-p3-training-v1",
                inputs=inputs,
            )
            result = supervise(
                [sys.executable, str(WORKER), "--request", str(path)],
                batch / f"member-{member}-supervision",
                OUTPUT,
                min(7200, remaining),
            )
            results.append(result)
            if result["status"] != "passed":
                raise GateError(f"Member {member} failed; subsequent members not started")
            terminal = read_json(batch / f"member-{member}" / "terminal.json")
            if terminal["status"] not in ("early_stopped", "capped"):
                raise GateError("Invalid member terminal status")
        seal = supervise(
            [sys.executable, str(Path(__file__).resolve()), "seal-worker"],
            OUTPUT / "seal-supervision",
            OUTPUT,
            1800,
        )
        if seal["status"] != "passed":
            raise GateError("Seal failed")
        durable_json(
            batch / "execution-result.json",
            {
                "status": "AWAITING_FINAL_AUDIT",
                "members": results,
                "elapsed": time.monotonic() - start,
                "oos_sealed": True,
                "p4_authorized": False,
            },
        )
    except BaseException as error:
        durable_json(
            batch / "execution-result.json",
            {
                "status": "BLOCKED",
                "members": results,
                "elapsed": time.monotonic() - start,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "oos_sealed": True,
            },
        )
        raise


def seal_worker():
    batch = OUTPUT / "batch"
    seal = OUTPUT / "seal"
    seal.mkdir()
    source = read_json(batch / "manifest.json")
    if verify_inputs() != source["inputs"] or code_fingerprints() != source["code"]:
        raise GateError("Final frozen input/code reconciliation failed")
    records = []
    for member in range(5):
        path = request(
            seal,
            f"probe-{member}",
            batch / f"member-{member}",
            synthetic=False,
            device="cuda",
            member=member,
            run_id="paper-p3-training-v1",
            inputs=source["inputs"],
            action="probe",
        )
        result = invoke(path)
        best = Path(result["best_directory"])
        target = seal / f"member-{member}"
        target.mkdir()
        for name in ("state.pt", "predictions.pt", "COMMIT.json"):
            shutil.copy2(best / name, target / name)
        meta = read_json(P2 / "members.json")["members"][member]
        durable_json(target / "normalization.json", meta["normalization"])
        records.append(
            {
                "member": member,
                **result,
                "terminal": read_json(batch / f"member-{member}/terminal.json"),
                "files": {p.name: digest(p) for p in target.iterdir()},
            }
        )
    durable_json(
        seal / "manifest.json",
        {
            "status": "SEALED_AWAITING_AUDIT",
            "members": records,
            "ensemble_score": "not_computed_different_validation_sets",
            "endpoint_verified": False,
            "strict_leakage_free_claim": False,
            "oos_sealed": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "preflight-worker", "train", "seal-worker"])
    action = parser.parse_args().action
    if action == "preflight-worker":
        raise SystemExit(preflight_worker())
    {"preflight": preflight, "train": train_batch, "seal-worker": seal_worker}[action]()
