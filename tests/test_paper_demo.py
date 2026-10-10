"""Publication semantics and tampering fail before presentation output."""
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("paper_demo", ROOT / "scripts/paper_demo.py")
DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEMO)


def test_public_monthly_arithmetic_and_confusion_reconcile():
    value = DEMO.recalculate()
    assert value["months"] == 227 and value["rows"] == 1387781
    assert value["portfolio"]["ew"]["annualized_sharpe"] == pytest.approx(1.9932780082793577)
    assert value["portfolio"]["vw"]["annualized_return"] == pytest.approx(.03655107176872348)
    assert value["accuracy_gain_percentage_points"] == pytest.approx(.26027161346061645)


def test_public_changed_bytes_rejected(tmp_path):
    shutil.copytree(DEMO.PUBLIC, tmp_path / "report")
    folder = tmp_path / "report"
    target = folder / "tables/monthly_portfolios.csv"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Public file mismatch"):
        DEMO.recalculate(folder)


def test_manifest_cannot_read_outside_public_directory(tmp_path):
    folder = tmp_path / "report"
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps({
        "scope": "paper_I20R20_five_member_limited_replication",
        "files": [{"path": "../private.txt", "bytes": 0, "sha256": "unused"}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        DEMO.verify_public(folder)


def test_confusion_denominator_checked_beyond_file_hash(tmp_path):
    shutil.copytree(DEMO.PUBLIC, tmp_path / "report")
    folder = tmp_path / "report"
    target = folder / "tables/classification.csv"
    text = target.read_text(encoding="utf-8-sig").replace("409890", "409891", 1)
    target.write_text(text, encoding="utf-8")
    manifest = DEMO.read(folder / "manifest.json")
    record = next(r for r in manifest["files"] if r["path"] == "tables/classification.csv")
    record.update(bytes=target.stat().st_size, sha256=DEMO.digest(target))
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="denominator"):
        DEMO.recalculate(folder)
