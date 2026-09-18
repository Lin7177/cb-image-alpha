"""Pure development-pool selection, partition and pixel-statistic contracts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass(frozen=True)
class PixelStatistics:
    white_pixels: int
    total_pixels: int
    mean: float
    std: float


def eligibility(
    dates: NDArray, returns: NDArray, volatility: NDArray,
) -> tuple[NDArray, dict[str, int]]:
    """Date proxy is user-approved; it is not a verified return-endpoint purge."""
    dates = np.asarray(dates, dtype="datetime64[D]")
    returns = np.asarray(returns, dtype=np.float64)
    volatility = np.asarray(volatility, dtype=np.float64)
    if dates.ndim != 1 or dates.shape != returns.shape or dates.shape != volatility.shape:
        raise ValueError("Expected aligned one-dimensional arrays")
    years = dates.astype("datetime64[Y]").astype(np.int64) + 1970
    if np.isnat(dates).any() or np.any((years < 1993) | (years > 2000)):
        raise ValueError("Only the 1993–2000 development pool is authorized")
    valid_return = np.isfinite(returns) & (returns != -99)
    valid_vol = np.isfinite(volatility) & (volatility > 0)
    before_tail = dates < np.datetime64("2000-12-01")
    keep = valid_return & valid_vol & before_tail
    counts = {
        "raw": len(dates),
        "invalid_return": int((~valid_return).sum()),
        "invalid_vol_after_return": int((valid_return & ~valid_vol).sum()),
        "tail_after_valid": int((valid_return & valid_vol & ~before_tail).sum()),
        "eligible": int(keep.sum()),
    }
    return keep, counts


def partition(size: int, seed: int) -> tuple[NDArray, NDArray]:
    if size < 2:
        raise ValueError("At least two samples required")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(size, generator=generator).numpy()
    cut = size * 7 // 10
    return permutation[:cut], permutation[cut:]


def pixel_statistics(white_counts: NDArray, training_indices: NDArray) -> PixelStatistics:
    counts = np.asarray(white_counts)
    indices = np.asarray(training_indices)
    if counts.ndim != 1 or counts.dtype.kind not in "iu":
        raise ValueError("Expected integer white-pixel counts")
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices):
        raise ValueError("Expected nonempty integer training indices")
    if (indices < 0).any() or (indices >= len(counts)).any():
        raise ValueError("Training index outside pool")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Duplicate training index")
    selected = counts[indices]
    if (selected < 0).any() or (selected > 64 * 60).any():
        raise ValueError("Invalid white-pixel count")
    white = int(selected.sum(dtype=np.int64))
    total = len(indices) * 64 * 60
    mean = float(np.float64(white) / np.float64(total))
    std = float(np.sqrt(np.float64(mean) * (1.0 - mean)))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Degenerate normalization")
    return PixelStatistics(white, total, mean, std)


def normalize(images: NDArray, statistics: PixelStatistics) -> NDArray:
    if images.dtype != np.uint8 or images.shape[-2:] != (64, 60):
        raise ValueError("Expected uint8 I20 images")
    if not np.all((images == 0) | (images == 255)):
        raise ValueError("Nonbinary image")
    if statistics.std <= 0 or not np.isfinite(statistics.std):
        raise ValueError("Invalid normalization")
    result = images.astype(np.float32) / np.float32(255)
    result -= np.float32(statistics.mean)
    result /= np.float32(statistics.std)
    return result
