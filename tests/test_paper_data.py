"""Synthetic archive isolation checks; no external data is opened."""
import importlib.util
from pathlib import Path
from zipfile import ZipInfo

import pytest

SPEC = importlib.util.spec_from_file_location(
    "paper_data", Path(__file__).resolve().parents[1] / "scripts" / "paper_data.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def inventory():
    return [ZipInfo(f"monthly_20d/20d_month_has_vb_[20]_ma_{year}_{suffix}")
            for year in range(1993, 2020)
            for suffix in ("images.dat", "labels_w_delay.feather")]


def test_only_development_members():
    selected = MODULE.selected_members(inventory())
    assert len(selected) == 16
    assert all(int(MODULE.MEMBER.fullmatch(info.filename)[1]) <= 2000 for info in selected)


@pytest.mark.parametrize("name", ["../escape", "C:/escape", "/escape", "x\\escape"])
def test_unsafe_inventory_rejected(name):
    with pytest.raises(ValueError, match="Unsafe"):
        info = ZipInfo("placeholder")
        info.filename = name  # ZipInfo constructor normalizes backslashes on Windows.
        MODULE.selected_members(inventory() + [info])


def test_missing_and_duplicate_rejected():
    infos = inventory()
    with pytest.raises(ValueError, match="incomplete"):
        MODULE.selected_members(infos[1:])
    with pytest.raises(ValueError, match="Duplicate"):
        MODULE.selected_members(infos + [infos[0]])


def test_symlink_rejected():
    info = ZipInfo("link")
    info.external_attr = 0o120777 << 16
    with pytest.raises(ValueError, match="Unsafe"):
        MODULE.selected_members(inventory() + [info])

def test_quality_audit_preserves_invalid_labels_as_counts(tmp_path):
    import numpy as np
    import pyarrow as pa
    from pyarrow import feather

    base = tmp_path / "20d_month_has_vb_[20]_ma_2000"
    table = pa.table({
        "Date": ["2000-10-31", "2000-10-31", "2000-11-30", "2000-12-29"],
        "StockID": [1, 2, 1, 1],
        "Ret_month": [0.0, float("nan"), -99.0, 0.1],
        "EWMA_vol": [1.0, 1.0, 0.0, 1.0],
    })
    feather.write_feather(table, base.with_name(base.name + "_labels_w_delay.feather"))
    images = np.zeros((4, 64, 60), dtype=np.uint8)
    images[0, 0, 0] = 255
    images.tofile(base.with_name(base.name + "_images.dat"))
    report = MODULE.audit_year(tmp_path, 2000)
    assert report["nonfinite_ret_month"] == 1
    assert report["sentinel_ret_month_minus99"] == 1
    assert report["tail_rows"] == 1
    assert report["candidate_rows_finite_nonsentinel_positive_ewma_before_tail"] == 1
    assert report["blank_images"] == 3
    assert report["pixel_histogram"] == {"0": 4 * 64 * 60 - 1, "255": 1}