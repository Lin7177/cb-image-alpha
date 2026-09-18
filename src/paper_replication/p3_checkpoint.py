"""P3 immutable checkpoints and explicit identity checks; no market-data access."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


class GateError(RuntimeError):
    """A frozen execution gate failed; caller must stop, never retry implicitly."""


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def durable_json(path: Path, value, *, exclusive: bool = True) -> None:
    payload = json.dumps(value, indent=2, allow_nan=False).encode("utf-8")
    with path.open("xb" if exclusive else "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def file_record(path: Path) -> dict:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}


def verify_file(path: Path, record: dict) -> None:
    if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
        raise GateError(f"File fingerprint mismatch: {path}")


def commit(
    folder: Path,
    state: dict,
    predictions: dict,
    identity: dict,
    parent: str | None,
    fault: str = "",
) -> str:
    """Never replace an epoch; only COMMIT.json makes its files authoritative."""
    epoch = int(state["epoch"])
    target = folder / f"epoch-{epoch:03d}"
    target.mkdir()  # uncommitted directories are retained and cannot be overwritten
    for name, value in [("state.pt", state), ("predictions.pt", predictions)]:
        with (target / name).open("xb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
    files = [file_record(target / name) for name in ("state.pt", "predictions.pt")]
    for record in files:
        verify_file(target / record["path"], record)
    marker = {"epoch": epoch, "identity": identity, "parent": parent, "files": files}
    durable_json(target / "COMMIT.pending", marker)
    if fault == "incomplete_commit":
        raise GateError("injected incomplete_commit")
    os.replace(target / "COMMIT.pending", target / "COMMIT.json")
    return digest(target / "COMMIT.json")


def committed(folder: Path, identity: dict) -> list[Path]:
    """Verify every marker, including the latest; never fall back on corruption."""
    markers = sorted(folder.glob("epoch-*/COMMIT.json"))
    parent = None
    for epoch, path in enumerate(markers):
        marker = read_json(path)
        if (
            marker["epoch"] != epoch
            or path.parent.name != f"epoch-{epoch:03d}"
            or marker["parent"] != parent
            or marker["identity"] != identity
        ):
            raise GateError("Checkpoint chain or identity mismatch")
        if {x["path"] for x in marker["files"]} != {"state.pt", "predictions.pt"}:
            raise GateError("Unexpected checkpoint files")
        for record in marker["files"]:
            verify_file(path.parent / record["path"], record)
        parent = digest(path)
    if not markers:
        raise GateError("No committed checkpoint")
    return [p.parent for p in markers]


def load_latest(folder: Path, identity: dict) -> tuple[dict, str]:
    latest = committed(folder, identity)[-1]
    # Only locally produced, hash-verified state with the bound code/input identity.
    state = torch.load(latest / "state.pt", map_location="cpu", weights_only=False)
    if state["identity"] != identity:
        raise GateError("State fingerprint mismatch")
    return state, digest(latest / "COMMIT.json")


def exact(left, right, location: str = "root") -> None:
    """Recursive same-device numeric comparison, including optimizer and RNG state."""
    import numpy as np

    if type(left) is not type(right):
        raise GateError(f"Type mismatch at {location}")
    if isinstance(left, torch.Tensor):
        if left.dtype != right.dtype or not torch.equal(left.cpu(), right.cpu()):
            raise GateError(f"Tensor mismatch at {location}")
    elif isinstance(left, np.ndarray):
        if left.dtype != right.dtype or not np.array_equal(left, right):
            raise GateError(f"Array mismatch at {location}")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise GateError(f"Keys mismatch at {location}")
        for key in left:
            exact(left[key], right[key], f"{location}.{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise GateError(f"Length mismatch at {location}")
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            exact(a, b, f"{location}[{i}]")
    elif left != right:
        raise GateError(f"Value mismatch at {location}")
