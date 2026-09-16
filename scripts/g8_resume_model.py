"""Fixed G8 estimators. No hyperparameter search or outer-set selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from g8_data import FoldImages, sha256_array
from g8_models import loss_on
from torch import nn

from cb_image_alpha.artifacts import read_json, sha256, write_json
from cb_image_alpha.models.paper_cnn import PaperCNN
from cb_image_alpha.training.engine import EarlyStop


def resume_cnn_stage(run: Path, year: int, mode: str) -> None:
    config = read_json(run / "plan.json")["config"]
    folder = run / str(year)
    torch.cuda.set_per_process_memory_fraction(config["gpu_memory_fraction"])
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    role = "inner_train" if mode == "inner" else "final_train"
    indices = np.load(folder / f"{role}.npy")
    data = FoldImages(run, indices)
    write_json(folder / f"{mode}-normalization.json", data.normalization)
    valid = (
        FoldImages(run, np.load(folder / "inner_validation.npy"), data.normalization)
        if mode == "inner"
        else None
    )
    epochs = (
        config["max_epochs"]
        if mode == "inner"
        else read_json(folder / "inner-summary.json")["best_epoch"]
    )
    model = PaperCNN().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    generator = torch.Generator().manual_seed(config["seed"])
    stopper = EarlyStop(config["patience"])
    checkpoint = torch.load(folder / "inner-latest.pt", map_location="cpu", weights_only=True)
    if (
        checkpoint["fold"],
        checkpoint["mode"],
        checkpoint["plan_sha256"],
        checkpoint["training_indices_sha256"],
    ) != (year, mode, sha256(run / "plan.json"), sha256_array(indices)):
        raise ValueError("Resume identity mismatch")
    if checkpoint["normalization"] != data.normalization:
        raise ValueError("Resume normalization mismatch")
    history = read_json(folder / "inner-history.json")
    epoch = checkpoint["epoch"]
    if [h["epoch"] for h in history] != list(range(1, epoch + 1)):
        raise ValueError("History/checkpoint mismatch")
    for h in history:
        stopper.observe(h["validation_loss"], h["epoch"])
    best = torch.load(folder / "inner-best.pt", map_location="cpu", weights_only=True)
    if best["epoch"] != stopper.epoch:
        raise ValueError("Best checkpoint/history mismatch")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    generator.set_state(checkpoint["shuffle_rng"])
    next_epoch = epoch + 1
    end_epoch = epoch if stopper.stopped else epochs
    for epoch in range(next_epoch, end_epoch + 1):
        model.train()
        total = 0.0
        order = torch.randperm(len(indices), generator=generator).numpy()
        for start in range(0, len(order), config["batch_size"]):
            x, y = data.batch(order[start : start + config["batch_size"]])
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x.cuda()), y.cuda())
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            gradients = [
                torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
            ]
            if len(gradients) != len(list(model.parameters())) or not torch.stack(gradients).all():
                raise ValueError("Invalid gradients")
            optimizer.step()
            total += float(loss.detach().cpu()) * len(y)
        record = {"epoch": epoch, "train_loss": total / len(indices)}
        improved = False
        if valid is not None:
            value = loss_on(model, valid, config["batch_size"])
            improved = stopper.observe(value, epoch)
            record["validation_loss"] = value
        history.append(record)
        write_json(folder / f"{mode}-history.json", history)
        checkpoint = {
            "scope": "development",
            "fold": year,
            "mode": mode,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "shuffle_rng": generator.get_state(),
            "plan_sha256": sha256(run / "plan.json"),
            "training_indices_sha256": sha256_array(indices),
            "normalization": data.normalization,
        }
        for name in [f"{mode}-latest.pt", f"{mode}-best.pt"] if improved else [f"{mode}-latest.pt"]:
            torch.save(checkpoint, folder / (name + ".tmp"))
            (folder / (name + ".tmp")).replace(folder / name)
        if mode == "inner" and stopper.stopped:
            break
    write_json(
        folder / f"{mode}-summary.json",
        {
            "status": "completed",
            "epochs": epoch,
            "best_epoch": stopper.epoch if mode == "inner" else epochs,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "training_indices_sha256": sha256_array(indices),
        },
    )
    if mode == "final":
        checkpoint = torch.load(folder / "final-latest.pt", map_location="cuda", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        outer = np.load(folder / "outer.npy")
        test = FoldImages(run, outer, data.normalization)
        p = np.empty(len(outer), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(outer), config["batch_size"]):
                positions = np.arange(start, min(start + config["batch_size"], len(outer)))
                x, _ = test.batch(positions)
                p[positions] = model(x.cuda()).softmax(1)[:, 1].cpu().numpy()
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Invalid CNN probabilities")
        np.save(folder / "CNN-p.npy", p, allow_pickle=False)
