"""Synthetic-testable training state and equal-probability ensemble primitives."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class EarlyStop:
    best_loss: float = float("inf")
    best_epoch: int = -1
    stale: int = 0

    def observe(self, epoch: int, loss: float) -> bool:
        if not torch.isfinite(torch.tensor(loss)):
            raise ValueError("Nonfinite validation loss")
        improved = loss < self.best_loss
        if improved:
            self.best_loss, self.best_epoch, self.stale = loss, epoch, 0
        else:
            self.stale += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.stale >= 2


def ensemble_probabilities(logits: list[Tensor]) -> Tensor:
    if len(logits) != 5:
        raise ValueError("All five members required")
    shape = logits[0].shape
    if len(shape) != 2 or shape[1] != 2:
        raise ValueError("Expected two logits per sample")
    if any(x.shape != shape or not torch.isfinite(x).all() for x in logits):
        raise ValueError("Mismatched or nonfinite member output")
    return torch.stack([torch.softmax(x, dim=1) for x in logits]).mean(dim=0)


def capture_state(model, optimizer, shuffle: torch.Generator, stopping: EarlyStop) -> dict:
    return deepcopy({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "cpu_rng": torch.get_rng_state(), "shuffle_rng": shuffle.get_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "stopping": vars(stopping).copy(),
    })


def restore_state(state: dict, model, optimizer, shuffle: torch.Generator) -> EarlyStop:
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["cpu_rng"])
    shuffle.set_state(state["shuffle_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return EarlyStop(**state["stopping"])
