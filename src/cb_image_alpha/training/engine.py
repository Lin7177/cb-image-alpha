"""Bounded smoke training; no dataset selection or filesystem access at import."""
from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from cb_image_alpha.models.paper_cnn import PaperCNN


@dataclass(frozen=True)
class Normalization:
    mean: float
    std: float


def fit_normalization(train: NDArray[np.uint8]) -> Normalization:
    if train.ndim != 3 or train.shape[1:] != (64, 60) or len(train) == 0:
        raise ValueError("Nonempty I20 training images required")
    values = train.astype(np.float64) / 255
    mean, std = float(values.mean()), float(values.std(ddof=0))
    if not np.isfinite(mean + std) or std <= 0:
        raise ValueError("Invalid training normalization")
    return Normalization(mean, std)


def normalize(images: NDArray[np.uint8], stats: Normalization) -> Tensor:
    return ((torch.from_numpy(images.copy()).float() / 255 - stats.mean) / stats.std).unsqueeze(1)


@dataclass
class EarlyStop:
    patience: int
    best: float = float("inf")
    stale: int = 0
    epoch: int = 0

    def observe(self, loss: float, epoch: int) -> bool:
        if not np.isfinite(loss):
            raise ValueError("Non-finite validation loss")
        improved = loss < self.best
        if improved:
            self.best, self.stale, self.epoch = loss, 0, epoch
        else:
            self.stale += 1
        return improved

    @property
    def stopped(self) -> bool:
        return self.stale >= self.patience


@torch.no_grad()
def probabilities(model: nn.Module, x: Tensor, batch_size: int, device: str) -> Tensor:
    model.eval()
    batches = [torch.softmax(model(chunk.to(device)), dim=1).cpu()
               for chunk in x.split(batch_size)]
    result = torch.cat(batches)
    if not torch.isfinite(result).all() or not torch.allclose(
        result.sum(dim=1), torch.ones(len(result)), atol=1e-6, rtol=0
    ):
        raise ValueError("Invalid probabilities")
    return result


def train_smoke(
    images: NDArray[np.uint8], labels: NDArray[np.int64], train_count: int,
    output: Path, *, seed: int, batch_size: int, learning_rate: float,
    max_epochs: int, patience: int, seconds_limit: int, device: str,
    input_fingerprint: str,
) -> tuple[Tensor, dict[str, object]]:
    if not 0 < train_count < len(images) or len(images) != len(labels):
        raise ValueError("Train/Validation partition mismatch")
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError("Binary labels required")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    stats = fit_normalization(images[:train_count])
    x = normalize(images, stats)
    y = torch.from_numpy(labels.copy()).long()
    model = PaperCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    shuffle = torch.Generator().manual_seed(seed)
    stopper = EarlyStop(patience)
    history = []
    started = time.monotonic()
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(train_count, generator=shuffle)
        weighted_loss = 0.0
        for indices in order.split(batch_size):
            if time.monotonic() - started > seconds_limit:
                raise TimeoutError("Smoke training wall-time budget exceeded")
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x[indices].to(device)), y[indices].to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Non-finite gradients")
            optimizer.step()
            weighted_loss += float(loss.detach().cpu()) * len(indices)
        model.eval()
        with torch.no_grad():
            logits = model(x[train_count:].to(device))
            validation_loss = float(nn.functional.cross_entropy(
                logits, y[train_count:].to(device)).cpu())
        if time.monotonic() - started > seconds_limit:
            raise TimeoutError("Smoke budget exceeded after validation")
        improved = stopper.observe(validation_loss, epoch)
        history.append({"epoch": epoch, "train_loss": weighted_loss / train_count,
                        "validation_loss": validation_loss, "improved": improved})
        if improved:
            checkpoint = {
                "scope": "smoke", "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "validation_loss": validation_loss,
                "normalization": asdict(stats), "input_fingerprint": input_fingerprint,
                "torch_rng": torch.get_rng_state(), "shuffle_rng": shuffle.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "seed": seed,
            }
            temporary = output / "best.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "best.pt")
        if stopper.stopped:
            break
    final_epoch = epoch
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    if checkpoint["input_fingerprint"] != input_fingerprint or checkpoint["epoch"] != stopper.epoch:
        raise ValueError("Checkpoint identity mismatch")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    first = probabilities(model, x, batch_size, device)
    clone = PaperCNN().to(device)
    clone.load_state_dict(torch.load(output / "best.pt", map_location=device,
                                     weights_only=True)["model"])
    second = probabilities(clone, x, batch_size, device)
    if not torch.equal(first, second):
        raise ValueError("Fresh checkpoint inference differs")
    metrics: dict[str, object] = {
        "scope": "smoke", "history": history, "best_epoch": stopper.epoch,
        "final_epoch": final_epoch, "best_validation_loss": stopper.best,
        "stop_reason": "early_stopping" if stopper.stopped else "max_epochs",
        "normalization": asdict(stats), "checkpoint_reload_exact": True,
        "training_seconds": time.monotonic() - started,
    }
    for name, sl in (("train", slice(0, train_count)), ("validation", slice(train_count, None))):
        probs, truth = first[sl], y[sl]
        metrics[name] = {"count": len(truth),
                         "accuracy": float((probs.argmax(dim=1) == truth).float().mean()),
                         "log_loss": float(-probs[torch.arange(len(truth)), truth].clamp_min(1e-12).log().mean())}
    if time.monotonic() - started > seconds_limit:
        raise TimeoutError("Smoke budget exceeded after restored inference")
    return first, metrics