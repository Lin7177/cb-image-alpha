"""Bounded acquisition and development-pool audit of the author's public images.

No reference source is imported. OOS members remain compressed and are not decoded.
One invocation is one recorded attempt; existing attempts are never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
import traceback
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np
from pyarrow import feather

ROOT = Path(__file__).resolve().parents[1]
URL = "https://dachxiu.chicagobooth.edu/download/img_data.zip"
ARCHIVE_BYTES = 567446780
LIMIT_SECONDS = 1800
LIMIT_BYTES = 12 * 1024**3
MEMBER = re.compile(
    r"monthly_20d/20d_month_has_vb_\[20\]_ma_(\d{4})_"
    r"(images\.dat|labels_w_delay\.feather)"
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def selected_members(infos: list[zipfile.ZipInfo]) -> list[zipfile.ZipInfo]:
    """Reject unsafe inventory before selecting exactly eight development years."""
    names = [info.filename for info in infos]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate archive member")
    selected = []
    for info in infos:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or ":" in info.filename:
            raise ValueError(f"Unsafe archive path: {info.filename}")
        if "\\" in info.filename or (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError(f"Unsafe archive type/path: {info.filename}")
        match = MEMBER.fullmatch(info.filename)
        if match and 1993 <= int(match[1]) <= 2000:
            selected.append(info)
    expected = {
        f"monthly_20d/20d_month_has_vb_[20]_ma_{year}_{suffix}"
        for year in range(1993, 2001)
        for suffix in ("images.dat", "labels_w_delay.feather")
    }
    if {info.filename for info in selected} != expected:
        raise ValueError("Development inventory is incomplete")
    if ARCHIVE_BYTES + sum(info.file_size for info in selected) >= LIMIT_BYTES:
        raise ValueError("Disk budget exceeded")
    return sorted(selected, key=lambda info: info.filename)


def audit_year(directory: Path, year: int) -> dict:
    base = directory / f"20d_month_has_vb_[20]_ma_{year}"
    label_path = base.with_name(base.name + "_labels_w_delay.feather")
    image_path = base.with_name(base.name + "_images.dat")
    table = feather.read_table(label_path)
    required = {"Date", "StockID", "Ret_month", "EWMA_vol"}
    if not required.issubset(table.column_names):
        raise ValueError(f"Missing fields in {year}: {required - set(table.column_names)}")
    rows = table.num_rows
    if image_path.stat().st_size != rows * 64 * 60:
        raise ValueError(f"Image/label count mismatch in {year}")
    dates = np.array(table["Date"].to_pylist(), dtype="datetime64[D]")
    if np.isnat(dates).any() or np.any(dates.astype("datetime64[Y]").astype(int) + 1970 != year):
        raise ValueError(f"Invalid development dates in {year}")
    stocks = table["StockID"].to_pylist()
    if any(stock is None for stock in stocks):
        raise ValueError(f"Null stock identifiers in {year}")
    keys = list(zip(dates.astype(str).tolist(), stocks, strict=True))
    if len(set(keys)) != rows:
        raise ValueError(f"Duplicate date/stock keys in {year}")
    returns = np.asarray(table["Ret_month"].to_pylist(), dtype=np.float64)
    volatility = np.asarray(table["EWMA_vol"].to_pylist(), dtype=np.float64)
    images = np.memmap(image_path, dtype=np.uint8, mode="r", shape=(rows, 64, 60))
    histogram = np.zeros(256, dtype=np.int64)
    blank = 0
    for start in range(0, rows, 1024):
        chunk = images[start : start + 1024]
        histogram += np.bincount(chunk.reshape(-1), minlength=256)
        blank += int(np.count_nonzero(~np.any(chunk, axis=(1, 2))))
    del images
    finite_return = np.isfinite(returns)
    sentinel = returns == -99
    tail = dates >= np.datetime64("2000-12-01")
    return {
        "year": year,
        "rows": rows,
        "schema": str(table.schema),
        "date_min": str(dates.min()),
        "date_max": str(dates.max()),
        "distinct_dates": len(np.unique(dates)),
        "distinct_stocks": len(set(stocks)),
        "nonfinite_ret_month": int((~finite_return).sum()),
        "sentinel_ret_month_minus99": int(sentinel.sum()),
        "nonfinite_ewma": int((~np.isfinite(volatility)).sum()),
        "zero_ewma": int((volatility == 0).sum()),
        "negative_ewma": int((volatility < 0).sum()),
        "tail_rows": int(tail.sum()),
        "candidate_rows_finite_nonsentinel_positive_ewma_before_tail": int(
            (finite_return & ~sentinel & np.isfinite(volatility) & (volatility > 0) & ~tail).sum()
        ),
        "blank_images": blank,
        "pixel_histogram": {str(i): int(n) for i, n in enumerate(histogram) if n},
    }


def main() -> None:
    started = time.monotonic()
    run_id = datetime.now(UTC).strftime("acquire-%Y%m%dT%H%M%S%fZ")
    output = ROOT / "artifacts" / "paper-v1" / run_id
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"run_id": run_id, "status": "running", "source": URL,
               "script_sha256": sha256(Path(__file__)), "files": [], "years": []}

    def save() -> None:
        receipt["wall_seconds"] = time.monotonic() - started
        (output / "manifest.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def budget() -> None:
        if time.monotonic() - started >= LIMIT_SECONDS:
            raise TimeoutError("P1 wall-clock budget exhausted")

    save()
    try:
        if shutil.disk_usage(output).free < LIMIT_BYTES:
            raise OSError("Less than 12 GiB free disk")
        archive = output / "img_data.zip"
        command = ["curl.exe", "--fail", "--location", "--silent", "--show-error",
                   "--connect-timeout", "30", "--max-time", "1500",
                   "--max-filesize", str(ARCHIVE_BYTES), "--output", str(archive),
                   "--dump-header", str(output / "http-headers.txt"),
                   "--write-out", "\nfinal_url=%{url_effective}\nhttp_code=%{http_code}\n", URL]
        receipt["command"] = command
        with (output / "download.log").open("wb") as log:
            result = subprocess.run(command, stdout=log, stderr=log, timeout=1520, check=False)
        receipt["download_exit_code"] = result.returncode
        save()
        if result.returncode or archive.stat().st_size != ARCHIVE_BYTES:
            raise RuntimeError("Download failed or archive size changed; no automatic retry")
        receipt["archive_sha256"] = sha256(archive)
        with zipfile.ZipFile(archive) as source:
            selected = selected_members(source.infolist())
            receipt["archive_members"] = len(source.infolist())
            for info in selected:
                budget()
                target = output / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(info) as incoming, target.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
                if target.stat().st_size != info.file_size:
                    raise ValueError("Extracted length mismatch")
                receipt["files"].append({"path": info.filename, "bytes": info.file_size,
                                         "zip_crc32": info.CRC, "sha256": sha256(target)})
                save()
        for year in range(1993, 2001):
            budget()
            receipt["years"].append(audit_year(output / "monthly_20d", year))
            save()
        budget()
        receipt["status"] = "completed"
        receipt["exit_code"] = 0
    except BaseException:
        receipt["status"] = "failed"
        receipt["exit_code"] = 1
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        receipt["output_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        save()
        print(output, flush=True)


if __name__ == "__main__":
    main()
