"""Independent Decimal coordinate checks of a G3 acceptance subset."""
from __future__ import annotations

import argparse
import hashlib
import json
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preview", type=Path)
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("Assertions must be enabled")
    root = args.preview
    report = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    images = np.load(root / "images.npy", allow_pickle=False)
    assert images.shape == (6, 64, 60) and images.dtype == np.uint8
    assert set(np.unique(images)) <= {0, 255}
    assert hashlib.sha256((root / "images.npy").read_bytes()).hexdigest() == report["images_sha256"]
    results = []
    for record in report["records"]:
        assert record["split"] in ("train", "validation")
        raw = json.loads((root / record["context_file"]).read_text(encoding="utf-8"))
        assert len(raw) == 39
        displayed = raw[19:]
        with localcontext() as ctx:
            ctx.prec = 40
            ctx.rounding = ROUND_HALF_EVEN
            closes = [Decimal(r["close"]) for r in raw]
            averages = [sum(closes[i - 19:i + 1]) / 20 for i in range(19, 39)]
            values = [Decimal(r[key]) for r in displayed for key in ("open", "high", "low", "close")]
            minimum, maximum = min(*values, *averages), max(*values, *averages)
            def row(value: Decimal, minimum: Decimal = minimum, maximum: Decimal = maximum) -> int:
                return 50 - int(((value - minimum) * 50 / (maximum - minimum)).to_integral_value())
            image = images[record["shard_row"]]
            assert not image[51].any()
            vols = [Decimal(r["volume"]) for r in displayed]
            for i, raw_bar in enumerate(displayed):
                x = 3 * i + 1
                high, low = row(Decimal(raw_bar["high"])), row(Decimal(raw_bar["low"]))
                assert (image[high:low + 1, x] == 255).all()
                assert image[row(Decimal(raw_bar["open"])), x - 1] == 255
                assert image[row(Decimal(raw_bar["close"])), x + 1] == 255
                assert image[row(averages[i]), x] == 255
                height = max(1, int((12 * vols[i] / max(vols)).to_integral_value()))
                assert np.count_nonzero(image[52:, x]) == height
            results.append({"sample_id": record["sample_id"], "split": record["split"],
                            "image_row": record["image_row"], "shard_row": record["shard_row"],
                            "first_open_row": row(Decimal(displayed[0]["open"])),
                            "first_ma_row": row(averages[0]),
                            "last_ma_row": row(averages[-1]),
                            "raw_price_min": str(minimum), "raw_price_max": str(maximum)})
    output = {"status": "passed", "method": "Decimal40 raw-price coordinates, independent of renderer",
              "ohlc_and_ma_vertices_checked": 120, "volume_columns_checked": 120,
              "records": results}
    (root / "coordinate-verification.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output))


if __name__ == "__main__":
    main()