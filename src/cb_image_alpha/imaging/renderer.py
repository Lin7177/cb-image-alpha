"""Independent I20 raster implementation from the documented chart specification."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from cb_image_alpha.data.catalog import RawBar

RENDER_VERSION = "i20-v1"
ImageArray = NDArray[np.uint8]


@dataclass(frozen=True)
class Raster:
    pixels: ImageArray
    ma20: tuple[float, ...]
    price_min: float
    price_max: float


def render_i20(context: Sequence[RawBar]) -> Raster:
    """Render 39 already calendar-validated bars; this function never computes labels."""
    if len(context) != 39:
        raise ValueError("I20/MA20 needs exactly 39 calendar-complete bars")
    if len({b.instrument_id for b in context}) != 1 or any(
        a.ts_event >= b.ts_event for a, b in pairwise(context)
    ):
        raise ValueError("Context must have one identity and increasing timestamps")
    prices = np.array([[float(b.open), float(b.high), float(b.low), float(b.close)]
                       for b in context], dtype=np.float64)
    volumes = np.array([float(b.volume) for b in context[-20:]], dtype=np.float64)
    if (not np.isfinite(prices).all() or (prices <= 0).any()
            or not np.isfinite(volumes).all() or (volumes <= 0).any()):
        raise ValueError("Invalid price or nonpositive display volume")
    if ((prices[:, 2] > prices[:, [0, 3]].min(axis=1)).any()
            or (prices[:, 1] < prices[:, [0, 3]].max(axis=1)).any()):
        raise ValueError("Invalid OHLC")
    # Float64 arithmetic and ties-to-even agree with the reference numeric convention.
    ma = np.array([prices[i - 19:i + 1, 3].mean() for i in range(19, 39)])
    scale = prices[19, 3]
    displayed = prices[-20:] / scale
    moving = ma / scale
    low = float(min(displayed.min(), moving.min()))
    high = float(max(displayed.max(), moving.max()))
    if high == low:
        raise ValueError("Degenerate price range; do not silently remove the sample")
    price_y = np.rint((displayed - low) * (50 / (high - low))).astype(int)
    ma_y = np.rint((moving - low) * (50 / (high - low))).astype(int)
    volume_height = np.rint(volumes * (12 / volumes.max())).astype(int)

    # Draw in upward-price coordinates, then invert once for row-major images.
    canvas = Image.new("L", (60, 64), 0)
    draw = ImageDraw.Draw(canvas)
    centers = [3 * i + 1 for i in range(20)]
    draw.line([(x, int(y) + 13) for x, y in zip(centers, ma_y)], fill=255, width=1)
    for i, x in enumerate(centers):
        op, hi, lo, cl = (int(y) + 13 for y in price_y[i])
        draw.line((x, lo, x, hi), fill=255)
        draw.point((x - 1, op), fill=255)
        draw.point((x + 1, cl), fill=255)
        # Pillow clips the reference's 0 -> -1 segment to a baseline pixel at height 0.
        draw.line((x, 0, x, max(0, int(volume_height[i]) - 1)), fill=255)
    pixels = np.asarray(canvas, dtype=np.uint8)[::-1].copy()
    return Raster(pixels, tuple(float(x) for x in ma), low, high)