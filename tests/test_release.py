import importlib.util
import json
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "release_audit", Path(__file__).parents[1] / "scripts/audit_release.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@pytest.fixture
def publication(tmp_path, monkeypatch):
    report = tmp_path / "reports/cb-v1/results.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"scope": "convertible_bond_method_transfer"}))
    (tmp_path / "README.md").write_text("[Results](reports/cb-v1/results.json)")
    files = ["README.md", "reports/cb-v1/results.json"]
    monkeypatch.setattr(AUDIT, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["audit_release.py"])
    monkeypatch.setattr(
        AUDIT.subprocess, "check_output", lambda *a, **k: ("\0".join(files) + "\0").encode()
    )
    return tmp_path, files


def test_publishable_subset(publication, capsys):
    AUDIT.main()
    assert json.loads(capsys.readouterr().out)["status"] == "passed"


@pytest.mark.parametrize("filename", ["artifacts/result.json", "weights.pt", ".env", "paper.pdf"])
def test_private_or_unlicensed_file_rejected(publication, filename):
    root, files = publication
    file = root / filename
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("placeholder")
    files.append(filename)
    with pytest.raises(ValueError, match="Forbidden publication"):
        AUDIT.main()


def test_link_to_unpublished_file_rejected(publication):
    root, _ = publication
    (root / "local.json").write_text("{}")
    (root / "README.md").write_text("[Local](local.json)")
    with pytest.raises(ValueError, match="Nonpublished Markdown link"):
        AUDIT.main()


def test_secret_detected_without_echo(publication):
    root, _ = publication
    fake = "ghp_" + "A" * 36
    (root / "README.md").write_text(fake)
    with pytest.raises(ValueError) as error:
        AUDIT.main()
    assert fake not in str(error.value)
    assert "Potential secret" in str(error.value)
