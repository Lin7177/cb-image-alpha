"""Formal training and one sealed evaluation; no automatic model-search loop."""
from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

from cb_image_alpha.artifacts import read_json, sha256, write_json
from cb_image_alpha.evaluation.metrics import compare_baselines
from cb_image_alpha.models.paper_cnn import PaperCNN
from cb_image_alpha.training.engine import EarlyStop
from cb_image_alpha.training.formal_data import ImageBatches, prepare_split, verify_split


def setup(config: dict) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen CUDA device unavailable")
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(config["gpu_memory_fraction"])
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def validation_loss(model: nn.Module, data: ImageBatches, batch_size: int) -> float:
    model.eval()
    total = 0.0
    for start in range(0, data.length, batch_size):
        x, y = data.batch(np.arange(start, min(start + batch_size, data.length)))
        loss = nn.functional.cross_entropy(model(x.cuda()), y.cuda(), reduction="sum")
        total += float(loss.cpu())
    value = total / data.length
    if not np.isfinite(value):
        raise ValueError("Nonfinite validation loss")
    return value


def train(run: Path) -> None:
    if (run / "seal.json").exists() or (run / "test-access.json").exists():
        raise ValueError("Cannot retrain a sealed experiment")
    config = read_json(run / "config.json")
    setup(config)
    train_data, valid_data = ImageBatches(run, "train"), ImageBatches(run, "validation")
    model = PaperCNN().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    generator = torch.Generator().manual_seed(config["seed"])
    stopper = EarlyStop(config["patience"])
    history = []
    started = time.monotonic()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        order = torch.randperm(train_data.length, generator=generator).numpy()
        loss_sum = 0.0
        for start in range(0, len(order), config["batch_size"]):
            x, y = train_data.batch(order[start:start + config["batch_size"]])
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x.cuda()), y.cuda())
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            gradients = [torch.isfinite(p.grad).all() for p in model.parameters()
                         if p.grad is not None]
            if len(gradients) != len(list(model.parameters())) or not torch.stack(gradients).all():
                raise ValueError("Missing or nonfinite gradients")
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(y)
        val = validation_loss(model, valid_data, config["batch_size"])
        improved = stopper.observe(val, epoch)
        history.append({"epoch": epoch, "train_loss": loss_sum / train_data.length,
                        "validation_loss": val, "improved": improved,
                        "elapsed_seconds": time.monotonic() - started})
        write_json(run / "history.json", history)
        checkpoint = {
            "scope": "formal", "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(), "shuffle_rng": generator.get_state(),
            "best_epoch": stopper.epoch, "best_loss": stopper.best, "stale": stopper.stale,
            "config_sha256": sha256(run / "config.json"),
            "normalization_sha256": sha256(run / "normalization.json"),
            "frozen_sha256": sha256(run / "frozen.json")}
        for filename in (["latest.pt", "best.pt"] if improved else ["latest.pt"]):
            torch.save(checkpoint, run / (filename + ".tmp"))
            (run / (filename + ".tmp")).replace(run / filename)
        if stopper.stopped:
            break
    summary = {"scope": "formal", "best_epoch": stopper.epoch, "final_epoch": epoch,
               "stop_reason": "early_stopping" if stopper.stopped else "max_epochs",
               "training_seconds": time.monotonic() - started,
               "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
    write_json(run / "training-summary.json", summary)
    verify_split(run, "train")
    verify_split(run, "validation")
    seal = {
        "scope": "formal", "sealed_at": datetime.now(UTC).isoformat(), **summary,
        "files": {p: sha256(run / p) for p in (
            "best.pt", "config.json", "normalization.json", "frozen.json",
            "train/manifest.json", "validation/manifest.json", "history.json")}}
    with (run / "seal.json").open("x", encoding="utf-8") as handle:
        json.dump(seal, handle, sort_keys=True, indent=2)


def evaluate(run: Path) -> None:
    seal = read_json(run / "seal.json")
    for name, expected in seal["files"].items():
        if sha256(run / name) != expected:
            raise ValueError("Sealed input changed")
    verify_split(run, "train")
    verify_split(run, "validation")
    config = read_json(run / "config.json")
    setup(config)
    checkpoint = torch.load(run / "best.pt", map_location="cuda", weights_only=True)
    if checkpoint["epoch"] != seal["best_epoch"]:
        raise ValueError("Best epoch mismatch")
    model = PaperCNN().cuda()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for key, filename in (("config_sha256", "config.json"),
                          ("normalization_sha256", "normalization.json"),
                          ("frozen_sha256", "frozen.json")):
        if checkpoint[key] != sha256(run / filename):
            raise ValueError("Checkpoint metadata mismatch")
    # The exclusive event precedes the first consumption of the Final Test sample partition.
    with (run / "test-access.json").open("x", encoding="utf-8") as handle:
        json.dump({"scope": "formal", "first_access": datetime.now(UTC).isoformat(),
                   "seal_sha256": sha256(run / "seal.json")}, handle)
    prepare_split(run, "test")
    rate = read_json(run / "normalization.json")["positive_rate"]
    metrics = {}
    for split in ("train", "validation", "test"):
        data = ImageBatches(run, split)
        probabilities = np.empty(data.length, dtype=np.float64)
        with torch.no_grad():
            for start in range(0, data.length, config["batch_size"]):
                indices = np.arange(start, min(start + config["batch_size"], data.length))
                x, _ = data.batch(indices)
                probs = model(x.cuda()).softmax(dim=1).cpu().numpy()
                if not np.isfinite(probs).all() or not np.allclose(probs.sum(1), 1, atol=1e-6):
                    raise ValueError("Invalid probabilities")
                probabilities[indices] = probs[:, 1]
        table = pq.read_table(run / split / "index.parquet")
        table = table.append_column("p_up", pa.array(probabilities))
        table = table.append_column("run_id", pa.array([run.name] * len(table)))
        table = table.append_column("scope", pa.array(["formal"] * len(table)))
        pq.write_table(table, run / (split + "-predictions.tmp"))
        (run / (split + "-predictions.tmp")).replace(run / (split + "-predictions.parquet"))
        metrics[split] = compare_baselines(data.labels, probabilities, rate)
        del data
    write_json(run / "metrics.json", {"scope": "formal", "train_rate": rate,
               "splits": metrics, "peak_gpu_bytes": torch.cuda.max_memory_allocated()})
    from cb_image_alpha.evaluation.verify import verify_results
    verify_results(run)
    for name, expected in seal["files"].items():
        if sha256(run / name) != expected:
            raise ValueError("Seal changed during evaluation")
