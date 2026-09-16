"""Behavioral tests for source isolation; all fixtures use disposable paths."""

import os
from pathlib import Path

import pytest

from cb_image_alpha.paths import UnsafeOutputPathError, validate_output_path


def test_output_is_resolved_without_creating_directories(tmp_path: Path) -> None:
    project = tmp_path / "project"
    result = validate_output_path(
        Path("artifacts/new/run"), project_root=project, source_root=tmp_path / "catalog"
    )
    assert result == project / "artifacts/new/run"
    assert not project.exists()


@pytest.mark.parametrize("suffix", ["", "bars/file.parquet"])
def test_source_and_descendants_are_rejected(tmp_path: Path, suffix: str) -> None:
    source = tmp_path / "project/catalog"
    with pytest.raises(UnsafeOutputPathError, match="^Output overlaps read-only source:"):
        validate_output_path(source / suffix, project_root=tmp_path / "project", source_root=source)


def test_source_ancestor_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(UnsafeOutputPathError, match="^Output overlaps read-only source:"):
        validate_output_path(tmp_path, project_root=tmp_path, source_root=tmp_path / "catalog")


@pytest.mark.parametrize("output", ["../outside", "artifacts/../../outside"])
def test_parent_traversal_escape_is_rejected(tmp_path: Path, output: str) -> None:
    with pytest.raises(UnsafeOutputPathError, match="^Output is outside project:"):
        validate_output_path(
            Path(output), project_root=tmp_path / "project", source_root=tmp_path / "catalog"
        )


def test_absolute_external_output_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(UnsafeOutputPathError, match="^Output is outside project:"):
        validate_output_path(
            tmp_path / "project-other/result",
            project_root=tmp_path / "project",
            source_root=tmp_path / "catalog",
        )


def test_internal_parent_traversal_is_normalized(tmp_path: Path) -> None:
    assert validate_output_path(
        Path("artifacts/child/../result"), project_root=tmp_path, source_root=tmp_path / "catalog"
    ) == tmp_path / "artifacts/result"


@pytest.mark.skipif(os.name != "nt", reason="Windows path casing semantics")
def test_windows_case_variation_cannot_bypass_source_boundary(tmp_path: Path) -> None:
    source = tmp_path / "Catalog"
    with pytest.raises(UnsafeOutputPathError, match="^Output overlaps read-only source:"):
        validate_output_path(
            Path(str(source).swapcase()) / "result",
            project_root=tmp_path,
            source_root=source,
        )


@pytest.mark.parametrize("destination", ["catalog", "outside"])
def test_link_escape_is_rejected(tmp_path: Path, destination: str) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = tmp_path / destination
    target.mkdir()
    link = project / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        if os.name == "nt" and exc.winerror == 1314:
            pytest.skip("Windows symlink privilege unavailable")
        raise
    message = "overlaps read-only source" if destination == "catalog" else "is outside project"
    with pytest.raises(UnsafeOutputPathError, match=f"^Output {message}:"):
        validate_output_path(link / "new/result", project_root=project, source_root=tmp_path / "catalog")


@pytest.mark.skipif(os.name != "nt", reason="Windows junction semantics")
def test_windows_junction_escape_is_rejected(tmp_path: Path) -> None:
    import subprocess

    project = tmp_path / "project"
    source = tmp_path / "catalog"
    project.mkdir()
    source.mkdir()
    junction = project / "junction"
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    with pytest.raises(UnsafeOutputPathError, match="^Output overlaps read-only source:"):
        validate_output_path(junction / "new/file", project_root=project, source_root=source)
