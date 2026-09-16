"""Split-specific image shards. Only requested split metadata is materialized."""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from numpy.typing import NDArray
from torch import Tensor

from cb_image_alpha.artifacts import read_json, sha256, write_json
from cb_image_alpha.data.catalog import read_instrument
from cb_image_alpha.imaging.renderer import RENDER_VERSION, render_i20
from cb_image_alpha.samples.index import session

COUNTS = {"train": 339682, "validation": 123747, "test": 151024}


def prepare_split(run: Path, split: str) -> None:
    frozen = read_json(run / "frozen.json")
    parent = Path(frozen["parent"])
    source = Path(frozen["source"])
    config = read_json(run / "config.json")
    if split not in COUNTS:
        raise ValueError("Unknown split")
    if split == "test" and not (run / "seal.json").exists():
        raise ValueError("Test access requires sealed checkpoint")
    if split == "test":
        event = read_json(run / "test-access.json")
        if event["seal_sha256"] != sha256(run / "seal.json"):
            raise ValueError("Test event does not match seal")
    columns = ["sample_id", "instrument_id", "context_start", "window_start",
               "window_end", "label_end", "forward_return", "image_row", "label", "split"]
    table = pq.read_table(parent / "samples.parquet", columns=columns,
                          filters=[("split", "=", split)]).sort_by([("image_row", "ascending")])
    if len(table) != COUNTS[split]:
        raise ValueError("Frozen split count mismatch")
    rows = table.to_pylist()
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate sample ID")
    with (source.parent / "download/base/calendar.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        calendar = [date.fromisoformat(r["trading_date"]) for r in csv.DictReader(handle)]
    positions = {day: i for i, day in enumerate(calendar)}
    size = config["shard_size"]
    folder = run / split
    folder.mkdir(exist_ok=False)
    buffer = np.empty((size, 64, 60), dtype=np.uint8)
    current = None
    lookup = {}
    sources = {}
    shards = []
    white = 0
    for index, row in enumerate(rows):
        if current != row["instrument_id"]:
            current = row["instrument_id"]
            directory = source / "day01/data/bar" / f"{current}-1-DAY-LAST-EXTERNAL"
            paths = list(directory.glob("*.parquet"))
            if len(paths) != 1:
                raise ValueError("Source file count changed")
            path = paths[0]
            expected = frozen["source_files"][str(path.relative_to(source))]
            if sha256(path) != expected:
                raise ValueError(f"Source changed: {path}")
            lookup = {session(b): b for b in read_instrument(directory)}
            if sha256(path) != expected:
                raise ValueError("Source changed while reading")
            sources[str(path)] = expected
        i = positions[row["window_end"]]
        days = calendar[i - 38:i + 1]
        if len(days) != 39 or days[0] != row["context_start"] or days[19] != row["window_start"]:
            raise ValueError(f"Calendar mismatch: {row['sample_id']}")
        try:
            image = render_i20([lookup[d] for d in days]).pixels
        except Exception as exc:
            raise ValueError(f"Rendering failed {row['sample_id']}: {exc}") from exc
        offset = index % size
        buffer[offset] = image
        white += int(np.count_nonzero(image))
        row["shard_id"], row["shard_row"] = index // size, offset
        if offset == size - 1 or index == len(rows) - 1:
            filename = f"{index // size:04d}.npy"
            temporary = folder / (filename + ".tmp")
            with temporary.open("wb") as handle:
                np.save(handle, buffer[:offset + 1], allow_pickle=False)
            temporary.replace(folder / filename)
            shards.append({"file": filename, "count": offset + 1,
                           "sha256": sha256(folder / filename)})
    index_path = folder / "index.parquet"
    indexed = pa.Table.from_pylist(rows)
    times = pq.read_table(parent / "samples.parquet", columns=["image_row", "decision_time"],
                          filters=[("split", "=", split)]).sort_by([("image_row", "ascending")])
    indexed = indexed.append_column("decision_time", times["decision_time"])
    pq.write_table(indexed, index_path)
    labels = np.array([r["label"] for r in rows], dtype=np.int64)
    np.save(folder / "labels.npy", labels, allow_pickle=False)
    write_json(folder / "manifest.json", {
        "scope": "formal", "status": "completed", "split": split, "count": len(rows),
        "render_version": RENDER_VERSION, "white_pixels": white, "pixels": len(rows) * 3840,
        "sources": sources, "shards": shards,
        "index_sha256": sha256(index_path), "labels_sha256": sha256(folder / "labels.npy"),
        "parent_samples_sha256": frozen["parent_files"]["samples.parquet"],
        "render_excluded": 0})
    if split == "train":
        mean = white / (len(rows) * 3840)
        std = float(np.sqrt(mean * (1 - mean)))
        if std <= 0:
            raise ValueError("Invalid training normalization")
        write_json(run / "normalization.json", {
            "mean": mean, "std": std, "ddof": 0, "fit_split": "train", "count": len(rows),
            "method": "exact binary pixel counts /255; population variance p*(1-p)",
            "positive_rate": float(labels.mean())})


def verify_split(run: Path, split: str) -> None:
    folder = run / split
    info = read_json(folder / "manifest.json")
    if info["status"] != "completed" or info["count"] != COUNTS[split]:
        raise ValueError("Invalid split manifest")
    for name, expected in (("index.parquet", info["index_sha256"]),
                           ("labels.npy", info["labels_sha256"])):
        if sha256(folder / name) != expected:
            raise ValueError("Split index/labels changed")
    for shard in info["shards"]:
        if sha256(folder / shard["file"]) != shard["sha256"]:
            raise ValueError("Image shard changed")


class ImageBatches:
    """Map uint8 shards; materialize and normalize only the current batch."""

    def __init__(self, run: Path, split: str) -> None:
        verify_split(run, split)
        self.folder = run / split
        info = read_json(self.folder / "manifest.json")
        self.shard_size = read_json(run / "config.json")["shard_size"]
        self.length = info["count"]
        self.arrays = [np.load(self.folder / r["file"], mmap_mode="r", allow_pickle=False)
                       for r in info["shards"]]
        self.labels = np.load(self.folder / "labels.npy", allow_pickle=False)
        self.stats = read_json(run / "normalization.json")

    def batch(self, indices: NDArray[np.int64]) -> tuple[Tensor, Tensor]:
        data = np.empty((len(indices), 64, 60), dtype=np.uint8)
        for shard in np.unique(indices // self.shard_size):
            mask = indices // self.shard_size == shard
            data[mask] = self.arrays[int(shard)][indices[mask] % self.shard_size]
        x = torch.from_numpy(data).float().unsqueeze(1) / 255
        x = (x - self.stats["mean"]) / self.stats["std"]
        return x, torch.from_numpy(self.labels[indices]).long()