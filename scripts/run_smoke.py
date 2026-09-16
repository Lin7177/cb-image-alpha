"""One bounded real smoke run, followed by a fresh-process prediction verification."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tomllib
from datetime import UTC, date, datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from cb_image_alpha.data.catalog import read_instrument
from cb_image_alpha.imaging.renderer import RENDER_VERSION, render_i20
from cb_image_alpha.models.paper_cnn import MODEL_VERSION, PaperCNN
from cb_image_alpha.paths import validate_output_path
from cb_image_alpha.samples.index import session
from cb_image_alpha.training.engine import Normalization, normalize, probabilities, train_smoke


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, default=str, indent=2), encoding="utf-8")
    temp.replace(path)


def verify(run: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    run = run.resolve(strict=True)
    if not run.is_relative_to(project) or not run.is_relative_to((project / "artifacts/runs").resolve()):
        raise ValueError("Verification output must stay in project smoke artifacts")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    inputs = json.loads((run / "inputs.json").read_text(encoding="utf-8"))
    validate_output_path(run, project_root=project, source_root=Path(inputs["source_root"]).parent)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if config["device"] == "cuda":
        torch.cuda.set_per_process_memory_fraction(config["gpu_memory_fraction"])
    for name in ("images", "labels"):
        if sha(run / f"{name}.npy") != inputs[f"{name}_sha256"]:
            raise ValueError(f"{name} input hash mismatch")
    checkpoint = torch.load(run / "best.pt", map_location=config["device"], weights_only=True)
    if checkpoint["input_fingerprint"] != sha(run / "inputs.json"):
        raise ValueError("Checkpoint input fingerprint mismatch")
    history = metrics["history"]
    if checkpoint["epoch"] != min(history, key=lambda r: r["validation_loss"])["epoch"]:
        raise ValueError("Did not restore best validation checkpoint")
    images = np.load(run / "images.npy", allow_pickle=False)
    train_count = config["train_count"]
    data = images[:train_count].astype(np.float64) / 255
    stats = checkpoint["normalization"]
    if stats != {"mean": float(data.mean()), "std": float(data.std(ddof=0))}:
        raise ValueError("Normalization not fitted exclusively to training subset")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    model = PaperCNN().to(config["device"])
    model.load_state_dict(checkpoint["model"])
    result = probabilities(model, normalize(images, Normalization(**stats)),
                           config["batch_size"], config["device"]).numpy()
    prediction = pq.read_table(run / "predictions.parquet").to_pydict()
    ids = [r["sample_id"] for r in inputs["selected"]]
    if len(ids) != config["train_count"] + config["validation_count"]:
        raise ValueError("Smoke count mismatch")
    if prediction["label"] != [r["label"] for r in inputs["selected"]]:
        raise ValueError("Prediction labels mismatch")
    if not np.array_equal(np.load(run / "labels.npy", allow_pickle=False),
                          [r["label"] for r in inputs["selected"]]):
        raise ValueError("Training labels mismatch")
    if prediction["sample_id"] != ids or len(ids) != len(set(ids)):
        raise ValueError("Prediction IDs missing/duplicated/reordered")
    stored = np.column_stack([prediction["p_down"], prediction["p_up"]])
    if not np.isfinite(stored).all() or not np.allclose(
        result, stored, atol=config["inference_atol"], rtol=config["inference_rtol"]
    ):
        raise ValueError("Fresh process inference mismatch")
    if any(r["split"] not in ("train", "validation") for r in inputs["selected"]):
        raise ValueError("Test sample in smoke run")
    save(run / "verification.json", {"scope": "smoke", "status": "passed",
         "count": len(ids), "fresh_process_inference_match": True,
         "normalization_train_only": True, "best_epoch": checkpoint["epoch"],
         "max_probability_difference": float(np.max(np.abs(result - stored))),
         "atol": config["inference_atol"], "rtol": config["inference_rtol"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--verify-run", type=Path)
    args = parser.parse_args()
    if args.verify_run:
        verify(args.verify_run)
        return
    if args.dataset is None:
        raise ValueError("--dataset required")
    project = Path(__file__).resolve().parents[1]
    config_path = project / "configs/smoke.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if config["scope"] != "smoke" or config["selection"] != "sample_id_ascending":
        raise ValueError("Unsupported smoke protocol")
    parent = args.dataset.resolve(strict=True)
    previous = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    source = Path(previous["inputs"]["source"])
    output = validate_output_path(
        Path("artifacts/runs") / datetime.now(UTC).strftime("smoke-%Y%m%dT%H%M%S%fZ"),
        project_root=project, source_root=source.parent)
    if not output.is_relative_to((project / "artifacts").resolve()):
        raise ValueError("Artifacts boundary required")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "manifest.json", {"scope": "smoke", "status": "running"})
    save(output / "config.json", config)
    try:
        if previous["status"] != "completed":
            raise ValueError("Parent dataset incomplete")
        for name, expected in previous["artifacts_sha256"].items():
            if sha(parent / name) != expected:
                raise ValueError("Parent artifact changed")
        auxiliary = previous["inputs"]["auxiliary_sha256"]
        for path, expected in auxiliary.items():
            if sha(Path(path)) != expected:
                raise ValueError("Auxiliary changed")
        columns = ["sample_id", "instrument_id", "context_start", "window_start", "window_end",
                   "image_row", "label", "split"]
        selected, accounting = [], {}
        for split in ("train", "validation"):
            table = pq.read_table(parent / "samples.parquet", columns=columns,
                                  filters=[("split", "=", split)])
            count = config[f"{split}_count"]
            if len(table) < count:
                raise ValueError("Insufficient smoke subset")
            chosen = table.sort_by([("sample_id", "ascending")]).slice(0, count).to_pylist()
            selected.extend(chosen)
            accounting[split] = {"eligible": len(table), "selected": count,
                                 "not_selected": len(table) - count, "render_excluded": 0}
        with (source.parent / "download/base/calendar.csv").open(
            encoding="utf-8-sig", newline=""
        ) as f:
            calendar = [date.fromisoformat(r["trading_date"]) for r in csv.DictReader(f)]
        position = {d: i for i, d in enumerate(calendar)}
        file_hashes = {r["path"]: r["sha256"] for r in previous["inputs"]["source_files"]}
        arrays = [None] * len(selected)
        actual_sources = {}
        for instrument in sorted({r["instrument_id"] for r in selected}):
            directory = source / "day01/data/bar" / f"{instrument}-1-DAY-LAST-EXTERNAL"
            files = list(directory.glob("*.parquet"))
            if len(files) != 1:
                raise ValueError("Changed file count")
            path = files[0]
            expected = file_hashes[str(path.relative_to(source))]
            if sha(path) != expected:
                raise ValueError("Changed source")
            lookup = {session(b): b for b in read_instrument(directory)}
            for j, record in enumerate(selected):
                if record["instrument_id"] != instrument:
                    continue
                i = position[record["window_end"]]
                days = calendar[i - 38:i + 1]
                if days[0] != record["context_start"] or days[19] != record["window_start"]:
                    raise ValueError("Context mismatch")
                arrays[j] = render_i20([lookup[d] for d in days]).pixels
            if sha(path) != expected:
                raise ValueError("Source changed during read")
            actual_sources[str(path)] = expected
        images = np.stack(arrays)
        np.save(output / "images.npy", images, allow_pickle=False)
        labels = np.array([r["label"] for r in selected], dtype=np.int64)
        np.save(output / "labels.npy", labels, allow_pickle=False)
        fingerprints = {str(p.relative_to(project)): sha(p) for p in sorted([
            *project.glob("src/**/*.py"), Path(__file__).resolve(), config_path, project / "uv.lock"])}
        inputs = {"scope": "smoke", "source_root": str(source), "selected": selected, "selection_accounting": accounting,
                  "parent_manifest_sha256": sha(parent / "manifest.json"),
                  "parent_samples_sha256": sha(parent / "samples.parquet"),
                  "parent_quality_sha256": sha(parent / "summary.json"),
                  "sources": actual_sources, "auxiliary_sha256": auxiliary,
                  "code_sha256": fingerprints, "images_sha256": sha(output / "images.npy"),
                  "labels_sha256": sha(output / "labels.npy"),
                  "render_version": RENDER_VERSION, "model_version": MODEL_VERSION}
        save(output / "inputs.json", inputs)
        save(output / "selection-statistics.json", accounting)
        torch.set_num_threads(2)
        if config["device"] == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Configured CUDA unavailable; no silent fallback")
            torch.cuda.set_per_process_memory_fraction(config["gpu_memory_fraction"])
            torch.cuda.reset_peak_memory_stats()
        save(output / "environment.json", {"python": sys.version, "platform": platform.platform(),
             "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
             "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
             "lock_sha256": sha(project / "uv.lock"), "scope": "smoke"})
        probs, metrics = train_smoke(
            images, labels, config["train_count"], output, seed=config["seed"],
            batch_size=config["batch_size"], learning_rate=config["learning_rate"],
            max_epochs=config["max_epochs"], patience=config["patience"],
            seconds_limit=config["training_seconds_limit"], device=config["device"],
            input_fingerprint=sha(output / "inputs.json"))
        metrics["peak_gpu_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        save(output / "normalization.json", metrics["normalization"])
        save(output / "metrics.json", metrics)
        rows = [{**r, "scope": "smoke", "p_down": float(p[0]), "p_up": float(p[1])}
                for r, p in zip(selected, probs)]
        pq.write_table(pa.Table.from_pylist(rows), output / "predictions.tmp")
        (output / "predictions.tmp").replace(output / "predictions.parquet")
        # Independent process: disk checkpoint and pixels are the only model inputs.
        command = [sys.executable, str(Path(__file__).resolve()), "--verify-run", str(output)]
        try:
            result = subprocess.run(command, cwd=project, capture_output=True, text=True,
                                    encoding="utf-8", check=False, timeout=120)
        except subprocess.TimeoutExpired as exc:
            for name, value in (("stdout", exc.stdout), ("stderr", exc.stderr)):
                text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
                (output / f"verify.{name}.log").write_text(text or "", encoding="utf-8")
            save(output / "verify-command.json",
                 {"command": command, "status": "timed_out", "timeout_seconds": 120})
            raise
        (output / "verify.stdout.log").write_text(result.stdout, encoding="utf-8")
        (output / "verify.stderr.log").write_text(result.stderr, encoding="utf-8")
        save(output / "verify-command.json", {"command": command, "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError("Independent verification failed; see stderr log")
        for path, expected in auxiliary.items():
            if sha(Path(path)) != expected:
                raise ValueError("Auxiliary changed during smoke")
        for path, expected in actual_sources.items():
            if sha(Path(path)) != expected:
                raise ValueError("Source changed during smoke")
        if sha(parent / "manifest.json") != inputs["parent_manifest_sha256"]:
            raise ValueError("Parent manifest changed during smoke")
        for name, expected in previous["artifacts_sha256"].items():
            if sha(parent / name) != expected:
                raise ValueError("Parent artifacts changed during smoke")
        if any(sha(project / path) != expected for path, expected in fingerprints.items()):
            raise ValueError("Code/config changed during smoke")
        save(output / "manifest.json", {"scope": "smoke", "status": "completed",
             "inputs_sha256": sha(output / "inputs.json"),
             "config_sha256": sha(output / "config.json"),
             "artifacts_sha256": {p.name: sha(p) for p in output.iterdir()
                                   if p.is_file() and p.name != "manifest.json"}})
        print(json.dumps({"scope": "smoke", "status": "completed", "output": str(output),
                          "best_epoch": metrics["best_epoch"], "samples": len(selected)}))
    except Exception as exc:
        save(output / "manifest.json", {"scope": "smoke", "status": "failed",
                                       "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()