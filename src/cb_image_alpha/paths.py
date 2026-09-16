"""Output boundaries for the read-only catalog pipeline."""

from pathlib import Path


class UnsafeOutputPathError(ValueError):
    """An output path violates the project/source isolation boundary."""


def validate_output_path(output: Path, *, project_root: Path, source_root: Path) -> Path:
    """Resolve an output without creating it, rejecting source overlap or escape.

    Relative outputs are interpreted relative to project_root. Existing symlinks
    and Windows junctions are resolved, including ancestors of new destinations.
    Call again immediately before writes; this check cannot prevent another
    process from replacing a link between validation and opening a file.
    """
    try:
        project = project_root.resolve()
        source = source_root.resolve()
        resolved = (output if output.is_absolute() else project / output).resolve()
    except (OSError, RuntimeError) as exc:
        raise UnsafeOutputPathError(f"Cannot resolve output boundary: {output}") from exc

    # Path comparisons are case-insensitive on Windows, matching its filesystem.
    if resolved.is_relative_to(source) or source.is_relative_to(resolved):
        raise UnsafeOutputPathError(f"Output overlaps read-only source: {resolved}")
    if not resolved.is_relative_to(project):
        raise UnsafeOutputPathError(f"Output is outside project: {resolved}")
    return resolved
