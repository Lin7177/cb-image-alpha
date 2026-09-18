"""Single supervised P2 preparation; never constructs a model or opens OOS files."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import tomllib
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import psutil
import torch
from pyarrow import feather

from paper_replication.contracts import eligibility, partition, pixel_statistics

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/paper/paper_b_v1.toml"
OUTPUT = ROOT / "artifacts/paper-v1/p2-paper-b-v1"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def prepare() -> None:
    torch.set_num_threads(1)
    config = tomllib.loads(CONFIG.read_text())
    freeze = json.loads((ROOT / "artifacts/paper-v1/b-protocol-freeze.json").read_text())
    if config["status"] != "FROZEN" or digest(CONFIG) != freeze["config_sha256"]:
        raise ValueError("Protocol not frozen or changed")
    for path, expected in [
        (ROOT / config["input_manifest"], config["input_manifest_sha256"]),
        (ROOT / config["model_source"], config["model_source_sha256"]),
        (ROOT / "uv.lock", config["environment_lock_sha256"]),
    ]:
        if digest(path) != expected:
            raise ValueError(f"Input fingerprint changed: {path.name}")
    source_path = ROOT / config["input_manifest"]
    source = json.loads(source_path.read_text())
    if source["status"] != "completed" or source["archive_sha256"] != config["input_archive_sha256"]:
        raise ValueError("Source not completed or wrong archive")
    expected_names = {
        f"monthly_20d/20d_month_has_vb_[20]_ma_{year}_{suffix}"
        for year in range(1993, 2001)
        for suffix in ("images.dat", "labels_w_delay.feather")
    }
    if len(source["files"]) != 16 or {x["path"] for x in source["files"]} != expected_names:
        raise ValueError("Only exact development inputs allowed")
    for item in source["files"]:
        path = source_path.parent / item["path"]
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Development file changed: {item['path']}")
    pool = {key: [] for key in ("year", "row", "stock", "date", "ret", "label", "white")}
    exclusions = []
    for year in config["development_years"]:
        if year not in range(1993, 2001):
            raise ValueError("OOS year prohibited")
        stem = source_path.parent / "monthly_20d" / f"20d_month_has_vb_[20]_ma_{year}"
        table = feather.read_table(stem.with_name(stem.name + "_labels_w_delay.feather"))
        dates = np.array(table["Date"].to_pylist(), dtype="datetime64[D]")
        returns = np.array(table["Ret_month"].to_pylist(), dtype=np.float64)
        volatility = np.array(table["EWMA_vol"].to_pylist(), dtype=np.float64)
        stocks = np.array(table["StockID"].to_pylist(), dtype=str)
        if any(value is None for value in table["StockID"].to_pylist()):
            raise ValueError("Null stock ID")
        if np.any(dates.astype("datetime64[Y]").astype(int) + 1970 != year):
            raise ValueError("Year mismatch")
        if len(set(zip(stocks.tolist(), dates.astype(str).tolist(), strict=True))) != len(dates):
            raise ValueError("Duplicate stock/date")
        keep, counts = eligibility(dates, returns, volatility)
        exclusions.append({"year": year, **counts})
        rows = np.flatnonzero(keep)
        image_path = stem.with_name(stem.name + "_images.dat")
        if image_path.stat().st_size != len(dates) * 3840:
            raise ValueError("Image row count mismatch")
        images = np.memmap(image_path, mode="r", dtype=np.uint8, shape=(len(dates), 64, 60))
        whites = np.empty(len(rows), dtype=np.int64)
        for start in range(0, len(rows), 512):
            chunk = images[rows[start:start+512]]
            if not np.all((chunk == 0) | (chunk == 255)):
                raise ValueError("Nonbinary development image")
            count = np.count_nonzero(chunk, axis=(1, 2))
            if np.any(count == 0):
                raise ValueError("Unexpected blank image; no silent exclusion")
            whites[start:start+len(count)] = count
        del images
        values = (np.full(len(rows), year, dtype=np.int16), rows, stocks[keep], dates[keep],
                  returns[keep], (returns[keep] > 0).astype(np.int64), whites)
        for key, value in zip(pool, values, strict=True):
            pool[key].append(value)
    arrays = {key: np.concatenate(value) for key, value in pool.items()}
    size = len(arrays["row"])
    if sum(x["raw"] for x in exclusions) != config["expected_raw_rows"] or size != config["expected_eligible_rows"]:
        raise ValueError("Frozen expected row counts differ")
    sample_ids = arrays["year"].astype(np.int64) * 1_000_000_000 + arrays["row"]
    if len(np.unique(sample_ids)) != size:
        raise ValueError("Duplicate sample ID")
    np.savez(OUTPUT / "pool.npz", sample_id=sample_ids, **arrays)
    dump(OUTPUT / "exclusions.json", {"years": exclusions})
    members = []
    for member, seed in enumerate(config["split_seeds"]):
        train, validation = partition(size, seed)
        if not np.array_equal(np.sort(np.concatenate([train, validation])), np.arange(size)):
            raise ValueError("Incomplete partition")
        stats = pixel_statistics(arrays["white"], train)
        filename = f"member-{member}.npz"
        np.savez(OUTPUT / filename, train=train, validation=validation,
                 train_ids=sample_ids[train], validation_ids=sample_ids[validation])
        members.append({"member": member, "split_seed": seed,
                        "model_seed": config["model_seeds"][member],
                        "shuffle_seed": config["shuffle_seeds"][member],
                        "train_count": len(train), "validation_count": len(validation),
                        "normalization": asdict(stats), "index_path": filename})
    dump(OUTPUT / "members.json", {"members": members})
    dump(OUTPUT / "config.json", config)
    files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.suffix in (".npz", ".json") and p.name != "supervision.json"]
    dump(OUTPUT / "manifest.json", {"status": "completed", "scope": "P2_no_real_training",
         "protocol_sha256": digest(CONFIG), "input_manifest_sha256": digest(source_path),
         "rows": size, "files": files, "endpoint_verified": False,
         "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "torch_cpu_threads": 1},
         "code": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in [Path(__file__), *sorted((ROOT / "src/paper_replication").glob("*.py"))]],
         "strict_leakage_free_claim": False})


def main() -> None:
    OUTPUT.mkdir(exist_ok=False)
    started = time.monotonic()
    config = tomllib.loads(CONFIG.read_text())
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = {"status": "running", "peak_rss_tree_bytes": 0}
    dump(OUTPUT / "supervision.json", result)
    with (OUTPUT / "worker.log").open("xb") as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__)), "--worker"],
                                 stdout=log, stderr=log, env=environment, cwd=ROOT)
        reason = None
        while child.poll() is None:
            process = psutil.Process(child.pid)
            processes = [process, *process.children(recursive=True)]
            rss = 0
            for item in processes:
                try:
                    rss += item.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            result["peak_rss_tree_bytes"] = max(result["peak_rss_tree_bytes"], rss)
            size = sum(p.stat().st_size for p in OUTPUT.rglob("*") if p.is_file())
            if time.monotonic()-started > config["p2_wall_seconds"]:
                reason = "wall_budget"
            elif rss > config["p2_rss_tree_bytes"]:
                reason = "rss_budget"
            elif size > config["p2_output_bytes"]:
                reason = "disk_budget"
            if reason:
                for item in reversed(processes):
                    try:
                        item.kill()
                    except psutil.NoSuchProcess:
                        pass
                break
            time.sleep(0.5)
        code = child.wait()
    result.update(status="completed" if code == 0 and reason is None else "failed",
                  exit_code=code, reason=reason, wall_seconds=time.monotonic()-started,
                  output_bytes=sum(p.stat().st_size for p in OUTPUT.rglob("*") if p.is_file()))
    if result["output_bytes"] > config["p2_output_bytes"]:
        result["status"] = "failed"
    dump(OUTPUT / "supervision.json", result)
    print(json.dumps(result), flush=True)
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        try:
            prepare()
        except BaseException:
            dump(OUTPUT / "failure.json", {"status": "failed", "error": traceback.format_exc()})
            raise
    else:
        main()
