"""Binary metrics with explicit ties, clipping and undefined-class behavior."""
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray


def binary_metrics(y: NDArray[np.int64], p: NDArray[np.float64]) -> dict:
    if len(y) == 0 or y.shape != p.shape or set(np.unique(y)) - {0, 1}:
        raise ValueError("Invalid binary metric inputs")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probability")
    positives = int(y.sum())
    negatives = len(y) - positives
    predicted = (p > 0.5).astype(np.int64)
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    result: dict = {"count": len(y), "positives": positives, "negatives": negatives,
              "accuracy": float((predicted == y).mean()),
              "log_loss": float(-(y * np.log(clipped) + (1-y) * np.log1p(-clipped)).mean()),
              "balanced_accuracy": None, "auc": None, "undefined_reason": None}
    if not positives or not negatives:
        result["undefined_reason"] = "single_class"
        return result
    result["balanced_accuracy"] = float(((predicted[y == 1] == 1).mean()
                                         + (predicted[y == 0] == 0).mean()) / 2)
    order = np.argsort(p, kind="stable")
    ordered = p[order]
    boundaries = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1, len(y)]
    positive_rank_sum = 0.0
    for start, end in pairwise(boundaries):
        average_rank = (int(start) + 1 + int(end)) / 2
        positive_rank_sum += int(y[order[start:end]].sum()) * average_rank
    result["auc"] = float((positive_rank_sum - positives * (positives + 1) / 2)
                           / (positives * negatives))
    return result


def compare_baselines(y: NDArray[np.int64], p: NDArray[np.float64], train_rate: float) -> dict:
    return {"cnn": binary_metrics(y, p),
            "train_constant_probability": binary_metrics(y, np.full(len(y), train_rate)),
            "train_majority": binary_metrics(y, np.full(len(y), float(train_rate > 0.5)))}