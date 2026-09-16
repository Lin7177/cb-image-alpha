"""Create a traceable Train/Validation-only image acceptance subset; never train."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import PIL
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from cb_image_alpha.data.catalog import read_instrument
from cb_image_alpha.imaging.renderer import RENDER_VERSION, render_i20
from cb_image_alpha.paths import validate_output_path
from cb_image_alpha.samples.index import session


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    parent = args.dataset.resolve(strict=True)
    manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("Parent dataset not completed")
    for name, expected in manifest["artifacts_sha256"].items():
        if sha(parent / name) != expected:
            raise ValueError("Parent artifact changed")
    source = Path(manifest["inputs"]["source"])
    for name, expected in manifest["inputs"]["auxiliary_sha256"].items():
        if sha(Path(name)) != expected:
            raise ValueError("Auxiliary changed")
    with (source.parent / "download/base/calendar.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        calendar = [date.fromisoformat(r["trading_date"]) for r in csv.DictReader(handle)]
    positions = {d: i for i, d in enumerate(calendar)}
    columns = ["sample_id", "instrument_id", "context_start", "window_start",
               "window_end", "image_row", "split"]
    # Filter before materializing Python rows; labels/Test values are never requested.
    chosen = []
    for split in ("train", "validation"):
        table = pq.read_table(parent / "samples.parquet", columns=columns,
                              filters=[("split", "=", split)])
        seen = set()
        for batch in table.to_batches(max_chunksize=2048):
            for row in batch.to_pylist():
                if row["instrument_id"] not in seen:
                    seen.add(row["instrument_id"])
                    chosen.append(row)
                if len(seen) == 3:
                    break
            if len(seen) == 3:
                break
        if len(seen) != 3:
            raise ValueError("Need three distinct instruments per development split")
    out = validate_output_path(
        Path("artifacts/images") / datetime.now(UTC).strftime("g3-%Y%m%dT%H%M%S%fZ"),
        project_root=project, source_root=source.parent)
    if not out.is_relative_to((project / "artifacts").resolve()):
        raise ValueError("Output must be in artifacts")
    out.mkdir(parents=True, exist_ok=False)
    report_path = out / "manifest.json"
    report_path.write_text('{"status":"running"}', encoding="utf-8")
    try:
        hashes = {r["path"]: r["sha256"] for r in manifest["inputs"]["source_files"]}
        arrays, records = [], []
        montage = Image.new("RGB", (600, 460), "#dddddd")
        labels = ImageDraw.Draw(montage)
        for slot, row in enumerate(chosen):
            directory = source / "day01/data/bar" / (
                row["instrument_id"] + "-1-DAY-LAST-EXTERNAL")
            files = list(directory.glob("*.parquet"))
            if len(files) != 1:
                raise ValueError("Source file count changed")
            path = files[0]
            expected = hashes[str(path.relative_to(source))]
            if sha(path) != expected:
                raise ValueError("Source changed")
            by_date = {session(b): b for b in read_instrument(directory)}
            i = positions[row["window_end"]]
            days = calendar[i - 38:i + 1]
            if days[0] != row["context_start"] or days[19] != row["window_start"]:
                raise ValueError("Sample calendar boundaries mismatch")
            context = [by_date[d] for d in days]
            raster = render_i20(context)
            if sha(path) != expected:
                raise ValueError("Source changed during read")
            arrays.append(raster.pixels)
            image = Image.fromarray(raster.pixels)
            image.save(out / f"{slot}.png")
            left, top = (slot % 3) * 200, (slot // 3) * 230
            montage.paste(image.resize((180, 192), Image.Resampling.NEAREST), (left, top + 35))
            labels.text((left, top), f'{row["split"]} {row["instrument_id"]}', fill="black")
            labels.text((left, top + 15), str(row["window_end"]), fill="black")
            raw_name = f"{slot}-context.json"
            (out / raw_name).write_text(json.dumps(
                [{"session": str(d), **asdict(b)} for d, b in zip(days, context)],
                default=str, indent=2), encoding="utf-8")
            records.append({**row, "shard_row": slot, "source_sha256": expected,
                            "context_file": raw_name, "ma20": raster.ma20,
                            "price_min": raster.price_min, "price_max": raster.price_max,
                            "pixels_sha256": hashlib.sha256(raster.pixels.tobytes()).hexdigest()})
        np.save(out / "images.npy", np.stack(arrays), allow_pickle=False)
        loaded = np.load(out / "images.npy", allow_pickle=False)
        for r in records:
            assert hashlib.sha256(loaded[r["shard_row"]].tobytes()).hexdigest() == r["pixels_sha256"]
        assert len({r["sample_id"] for r in records}) == len(records)
        montage.save(out / "contact-sheet.png")
        report = {
            "status": "completed", "scope": "six Train/Validation acceptance images, not full cache",
            "parent_dataset": str(parent), "parent_manifest_sha256": sha(parent / "manifest.json"),
            "parent_samples_sha256": sha(parent / "samples.parquet"),
            "render_version": RENDER_VERSION,
            "code_sha256": {str(p.relative_to(project)): sha(p) for p in (
                project / "src/cb_image_alpha/imaging/renderer.py", Path(__file__).resolve())},
            "pillow_version": PIL.__version__, "numpy_version": np.__version__,
            "images_sha256": sha(out / "images.npy"), "records": records,
        }
        temporary = out / "manifest.tmp"
        temporary.write_text(json.dumps(report, default=str, indent=2), encoding="utf-8")
        temporary.replace(report_path)
        print(json.dumps({"output": str(out), "images": len(records), "status": "completed"}))
    except Exception as exc:
        report_path.write_text(json.dumps({"status": "failed", "error": str(exc)}), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()