"""Classroom entry: recalculate public aggregates or run a synthetic P3 workflow."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "reports/paper-i20r20"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def csv_rows(folder: Path, name: str) -> list[dict[str, str]]:
    with (folder / "tables" / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_public(folder: Path) -> dict:
    manifest = read(folder / "manifest.json")
    if manifest["scope"] != "paper_I20R20_five_member_limited_replication":
        raise ValueError("Unexpected publication scope")
    for item in manifest["files"]:
        path = (folder / item["path"]).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError("Manifest path escapes public report directory")
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Public file mismatch: {item['path']}")
    return manifest


def close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"Aggregate reconciliation failed: {name}")


def recalculate(folder: Path = PUBLIC) -> dict:
    """Independent arithmetic on published monthly series; no row-level predictions."""
    manifest = verify_public(folder)
    monthly = csv_rows(folder, "monthly_portfolios.csv")
    dates = [r["date"] for r in monthly]
    if len(dates) != 227 or dates != sorted(set(dates)):
        raise ValueError("Expected 227 unique chronological months")
    summary = read(folder / "tables/summary.json")
    signals = summary["portfolio"]["signals"]
    metrics = {}
    for weight in ("ew", "vw"):
        values = [float(r[f"{weight}_H-L"]) for r in monthly]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Nonfinite monthly return")
        for row in monthly:
            close(float(row[f"{weight}_H-L"]),
                  float(row[f"{weight}_high"]) - float(row[f"{weight}_low"]),
                  f"{weight} D10-D1")
        mean = statistics.mean(values)
        ann = mean * 12
        vol = statistics.stdev(values) * math.sqrt(12)
        metrics[weight] = {"annualized_return": ann, "annualized_volatility": vol,
                           "annualized_sharpe": ann / vol}
        for name, value in metrics[weight].items():
            close(value, signals["ensemble"][weight]["H-L"][name], f"{weight}/{name}")
        running = 0.0
        for row, value in zip(monthly, values, strict=True):
            running += value
            close(running, float(row[f"{weight}_cumulative_spread_sum"]), "spread sum")
    classes = csv_rows(folder, "classification.csv")
    ensemble = next(r for r in classes if r["signal"] == "ensemble")
    members = [r for r in classes if r["signal"].startswith("member-")]
    if len(members) != 5 or len({r["signal"] for r in members}) != 5:
        raise ValueError("Exactly five distinct members required")
    for r in classes:
        tn, fp, fn, tp = (int(r[k]) for k in ("TN", "FP", "FN", "TP"))
        n = int(r["rows"])
        if tn + fp + fn + tp != n or n != 1387781:
            raise ValueError("Classification denominator mismatch")
        close((tp + tn) / n, float(r["accuracy"]), "accuracy from confusion matrix")
    accuracy = float(ensemble["accuracy"])
    baseline = float(ensemble["always_up_accuracy"])
    ew_members = [signals[f"member-{m}"]["ew"]["H-L"]["annualized_sharpe"]
                  for m in range(5)]
    dispersion = {"mean": statistics.mean(ew_members),
                  "sample_std_ddof1": statistics.stdev(ew_members),
                  "min": min(ew_members), "max": max(ew_members)}
    reported = next(r for r in csv_rows(folder, "member_dispersion.csv")
                    if r["metric"] == "ew_hl_sharpe")
    for key, source in [("mean", "member_mean"), ("sample_std_ddof1", "member_sample_std_ddof1"),
                        ("min", "member_min"), ("max", "member_max")]:
        close(dispersion[key], float(reported[source]), key)
    return {"scope": manifest["scope"], "months": len(monthly), "rows": int(ensemble["rows"]),
            "date_min": dates[0], "date_max": dates[-1], "portfolio": metrics,
            "accuracy": accuracy, "always_up_accuracy": baseline,
            "accuracy_gain_percentage_points": (accuracy - baseline) * 100,
            "balanced_accuracy": float(ensemble["balanced_accuracy"]),
            "roc_auc": float(ensemble["roc_auc"]), "member_ew_sharpe": dispersion,
            "checks": ["manifest bytes/SHA256", "227 chronological monthly H-L D10-D1",
                       "arithmetic annualization and sample standard deviation",
                       "cumulative sum, not NAV", "classification confusion denominators",
                       "five-member descriptive variability"],
            "limits": ["Ret_month proxy", "endpoint unverified", "already-seen evaluation",
                       "no trading costs", "single configuration", "dispersion is not CI"],
            "new_training": False, "new_inference": False, "raw_oos_reads": 0}


def results(run: Path) -> dict:
    print("[1/3] Verify published files and recalculate 227 monthly observations", flush=True)
    payload = recalculate()
    write(run / "results.json", payload)
    print("[2/3] Reconcile classification baselines and all five members", flush=True)
    print(f"Accuracy {payload['accuracy']:.4%} | always-up {payload['always_up_accuracy']:.4%} "
          f"| gain {payload['accuracy_gain_percentage_points']:.4f} percentage points")
    for weight, values in payload["portfolio"].items():
        print(f"{weight.upper()} H-L: annual arithmetic return {values['annualized_return']:.4%} "
              f"| Sharpe {values['annualized_sharpe']:.4f}")
    source = (PUBLIC / "report.html").read_text(encoding="utf-8")
    # Embed report images and remove its local hyperlink for a single portable output.
    import base64
    import re
    source = re.sub(r'<a href="[^"]+">([^<]+)</a>', r'\1', source)
    for path in (PUBLIC / "figures").glob("*.png"):
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        source = source.replace(f'figures/{path.name}', f'data:image/png;base64,{encoded}')
    banner = '<p class="notice">现场执行：公开聚合表重新核算通过。不是重新训练或重跑原始图像评价。</p>'
    source = source.replace("<main>", "<main>" + banner, 1)
    (run / "report.html").write_text(source, encoding="utf-8")
    print("[3/3] Wrote results.json and self-contained report.html", flush=True)
    return payload


def workflow_worker(run: Path) -> None:
    """Uses production engine unchanged; only the input fixture is artificial."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    import torch

    from cb_image_alpha.models.paper_cnn import PaperCNN
    from paper_replication.p3_checkpoint import committed, exact, load_latest
    from paper_replication.p3_data import synthetic
    from paper_replication.p3_engine import run as train
    from paper_replication.state import ensemble_probabilities
    print("[1/5] SYNTHETIC ONLY: 256 train + 128 validation binary images; random labels", flush=True)
    training, validation, meta = synthetic()
    identity = {"run_id": run.name, "purpose": "synthetic_classroom_demo",
                "input": "binary-63017-256-128", "code": read(run / "config.json")["code"]}
    environment = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                   "device": "cpu", "meta": meta, "input_is_synthetic": True}
    write(run / "environment.json", environment)
    folder = run / "model"
    print("[2/5] Production PaperCNN + Adam: train/validate epoch 1, commit, stop", flush=True)
    interrupted = train(folder, run, training, validation, meta, identity, "cpu",
                        epochs=2, interrupt_after=1, synthetic=True)
    if interrupted != {"status": "injected_interruption", "epoch": 1}:
        raise ValueError("Demo interruption was not observed")
    print("[3/5] Restore model/Adam/RNG/shuffle/early-stop state, finish epoch 2", flush=True)
    terminal = train(folder, run, training, validation, meta, identity, "cpu",
                     epochs=2, resume=True, synthetic=True)
    state, _ = load_latest(folder, identity)
    chain = committed(folder, identity)
    if state["epoch"] != 2 or len(chain) != 3:
        raise ValueError("Demo checkpoint recovery mismatch")
    model = PaperCNN(); model.load_state_dict(state["model"]); model.eval()
    images = torch.stack([validation[i][0] for i in range(128)])
    with torch.no_grad():
        probability = model(images).softmax(1)
    saved = torch.load(chain[-1] / "predictions.pt", map_location="cpu", weights_only=False)
    exact(probability, saved["probability"], "restored_demo_predictions")
    print("[4/5] Reloaded checkpoint probabilities match saved validation output exactly", flush=True)
    a = torch.tensor([[0.0, 0.0]])
    b = torch.tensor([[0.0, math.log(3.0)]])
    combined = ensemble_probabilities([a, a, a, a, b])
    if not torch.allclose(combined, torch.tensor([[0.45, 0.55]])):
        raise ValueError("Probability ensemble fixture mismatch")
    print("[5/5] Hand example: average five softmax probabilities -> p(up)=0.55", flush=True)
    payload = {"status": "PASS_SYNTHETIC_WORKFLOW_ONLY", "synthetic": True, "terminal": terminal,
               "checkpoint_commits": len(chain), "restore_exact": True,
               "ensemble_hand_fixture": combined.tolist(), "formal_training_runs": 0,
               "raw_oos_reads": 0,
               "note": "Two epochs on synthetic images; not five trained members or market accuracy."}
    write(run / "workflow.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["results", "workflow"], default="results")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        target = args.worker.resolve()
        if not target.is_relative_to((ROOT / "artifacts/course-demo").resolve()):
            raise ValueError("Worker output must stay inside artifacts/course-demo")
        workflow_worker(target)
        return 0
    run = ROOT / "artifacts/course-demo" / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + args.mode + "-" + uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    code_paths = [Path(__file__), *sorted((ROOT / "src/paper_replication").glob("*.py")),
                  ROOT / "src/cb_image_alpha/models/paper_cnn.py", ROOT / "uv.lock"]
    write(run / "config.json", {"run_id": run.name, "mode": args.mode, "argv": sys.argv,
                                 "python": sys.version, "platform": platform.platform(),
                                 "code": {p.relative_to(ROOT).as_posix(): digest(p) for p in code_paths},
                                 "public_manifest_sha256": digest(PUBLIC / "manifest.json"),
                                 "new_formal_training": False, "raw_oos_reads": 0})
    started = time.perf_counter()
    exit_code = 0
    try:
        if args.mode == "results":
            import contextlib
            with (run / "stdout.log").open("x", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                results(run)
            print((run / "stdout.log").read_text(encoding="utf-8"), end="")
        else:
            sys.path.insert(0, str(ROOT / "src"))
            from paper_replication.p3_supervision import supervise
            print("Running synthetic CPU workflow (2 epochs); logs: " + str(run / "supervision"), flush=True)
            result = supervise([sys.executable, str(Path(__file__)), "--worker", str(run)],
                               run / "supervision", run, 180,
                               rss_limit=2 * 1024**3, disk_limit=128 * 1024**2)
            log = run / "supervision/stdout.log"
            if log.exists():
                print(log.read_text(encoding="utf-8"), end="")
            if result["status"] != "passed" or not (run / "workflow.json").exists():
                raise RuntimeError(f"Synthetic demo failed; see {run / 'supervision'}")
    except Exception:  # noqa: BLE001 -- terminal boundary preserves failure evidence
        exit_code = 1
        (run / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
    write(run / "execution.json", {"run_id": run.name, "status": "passed" if exit_code == 0 else "failed",
                                    "exit_code": exit_code, "elapsed_seconds": time.perf_counter() - started,
                                    "new_formal_training": False, "raw_oos_reads": 0})
    write(run / "manifest.json", {"files": [
        {"path": p.relative_to(run).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(run.rglob("*")) if p.is_file()]})
    print(f"Run output: {run}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
