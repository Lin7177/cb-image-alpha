"""Fixed P2 adapter and artificial P3 fixture. Exact development-path allowlist."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from paper_replication.contracts import PixelStatistics, normalize
from paper_replication.p3_checkpoint import GateError, digest, read_json, verify_file

ROOT = Path(__file__).resolve().parents[2]
P2 = ROOT / "artifacts/paper-v1/p2-paper-b-v1"
CONFIG = ROOT / "configs/paper/p3_training_v1.toml"
B_CONFIG = ROOT / "configs/paper/paper_b_v1.toml"
SOURCE = ROOT / "artifacts/paper-v1/acquire-20260917T010443459965Z/recovery-1"


def image_path(year: int) -> Path:
    if year not in range(1993, 2001):
        raise GateError("Forbidden OOS path/year")
    return SOURCE / "monthly_20d" / f"20d_month_has_vb_[20]_ma_{year}_images.dat"


def authorized_image(path: Path) -> Path:
    if path.resolve() not in {image_path(y).resolve() for y in range(1993, 2001)}:
        raise GateError("Forbidden OOS or non-allowlisted path")
    return path


def verify_inputs() -> dict:
    p3 = tomllib.loads(CONFIG.read_text())
    b = tomllib.loads(B_CONFIG.read_text())
    if (
        digest(CONFIG)
        != read_json(ROOT / "artifacts/paper-v1/p3-protocol-freeze.json")["protocol_sha256"]
    ):
        raise GateError("P3 frozen protocol changed")
    bindings = {
        B_CONFIG: p3["parent_config_sha256"],
        P2 / "manifest.json": p3["p2_manifest_sha256"],
        P2 / "members.json": p3["p2_members_sha256"],
        P2 / "p2-acceptance.json": p3["p2_acceptance_sha256"],
        ROOT / b["input_manifest"]: b["input_manifest_sha256"],
        ROOT / b["model_source"]: b["model_source_sha256"],
        ROOT / "uv.lock": b["environment_lock_sha256"],
    }
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise GateError(f"Frozen fingerprint mismatch: {path}")
    p2 = read_json(P2 / "manifest.json")
    for record in p2["files"]:
        path = P2 / record["path"]
        if path.resolve().parent != P2.resolve():
            raise GateError("Invalid P2 manifest path")
        verify_file(path, record)
        bindings[path] = record["sha256"]
    for record in p2["code"]:
        path = ROOT / record["path"]
        if digest(path) != record["sha256"]:
            raise GateError("Frozen P2 source changed")
        bindings[path] = record["sha256"]
    source = read_json(SOURCE / "manifest.json")
    allowed = {
        f"monthly_20d/20d_month_has_vb_[20]_ma_{y}_{s}"
        for y in range(1993, 2001)
        for s in ("images.dat", "labels_w_delay.feather")
    }
    if {r["path"] for r in source["files"]} != allowed or len(source["files"]) != 16:
        raise GateError("Source allowlist mismatch")
    for record in source["files"]:
        path = SOURCE / record["path"]
        verify_file(path, record)
        bindings[path] = record["sha256"]
    return {str(p.relative_to(ROOT)): h for p, h in bindings.items()}


def code_fingerprints() -> dict:
    files = [
        *sorted((ROOT / "src/paper_replication").glob("*.py")),
        ROOT / "src/cb_image_alpha/models/paper_cnn.py",
        *sorted((ROOT / "scripts").glob("paper_p3*.py")),
        *sorted((ROOT / "tests").glob("test_paper_p3*.py")),
        CONFIG,
        B_CONFIG,
        ROOT / "uv.lock",
    ]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def environment(device: str) -> dict:
    import subprocess

    return {
        "python": sys.version,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip()
        if device == "cuda"
        else None,
        "threads": torch.get_num_threads(),
    }


class Images(Dataset):
    def __init__(self, ids, labels, statistics, *, pixels=None, years=None, rows=None):
        self.ids = np.asarray(ids, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.statistics = statistics
        self.pixels, self.years, self.rows = pixels, years, rows
        self.maps = {}
        if len(np.unique(self.ids)) != len(self.ids) or len(self.labels) != len(self.ids):
            raise GateError("Duplicate/missing dataset IDs")
        if not np.isin(self.labels, [0, 1]).all():
            raise GateError("Invalid labels")
        if pixels is None:
            for year in np.unique(years):
                path = authorized_image(image_path(int(year)))
                count, remainder = divmod(path.stat().st_size, 3840)
                if remainder:
                    raise GateError("Invalid image file shape")
                self.maps[int(year)] = np.memmap(
                    path, dtype=np.uint8, mode="r", shape=(count, 64, 60)
                )
            if not np.array_equal(self.ids, years.astype(np.int64) * 1000000000 + rows):
                raise GateError("Image sample_id mapping mismatch")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        pixels = (
            self.pixels[index]
            if self.pixels is not None
            else self.maps[int(self.years[index])][int(self.rows[index])]
        )
        array = normalize(pixels, self.statistics)
        return torch.from_numpy(array[None]), int(self.labels[index]), int(self.ids[index])


def synthetic():
    rng = np.random.default_rng(63017)
    pixels = (rng.random((384, 64, 60)) < 0.1063).astype(np.uint8) * 255
    labels = rng.integers(0, 2, 384, dtype=np.int64)
    mean = np.float64(np.count_nonzero(pixels[:256])) / np.float64(256 * 3840)
    statistics = PixelStatistics(
        int(np.count_nonzero(pixels[:256])),
        256 * 3840,
        float(mean),
        float(np.sqrt(mean * (1 - mean))),
    )
    train = Images(np.arange(256), labels[:256], statistics, pixels=pixels[:256])
    val = Images(np.arange(256, 384), labels[256:], statistics, pixels=pixels[256:])
    return train, val, {"model_seed": 2701, "shuffle_seed": 3701, "normalization": vars(statistics)}


def real(member: int):
    if member not in range(5):
        raise GateError("Unregistered member")
    meta = read_json(P2 / "members.json")["members"][member]
    stats = PixelStatistics(**meta["normalization"])
    with (
        np.load(P2 / "pool.npz", allow_pickle=False) as pool,
        np.load(P2 / meta["index_path"], allow_pickle=False) as indices,
    ):
        datasets = []
        for split, count in [("train", 544954), ("validation", 233553)]:
            positions = indices[split]
            ids = pool["sample_id"][positions]
            if len(ids) != count or not np.array_equal(ids, indices[split + "_ids"]):
                raise GateError("P2 index count/ID mismatch")
            datasets.append(
                Images(
                    ids,
                    pool["label"][positions],
                    stats,
                    years=pool["year"][positions],
                    rows=pool["row"][positions],
                )
            )
    return *datasets, meta
