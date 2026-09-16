"""Content-addressed evidence helpers; importing never creates outputs."""
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, default=str, sort_keys=True, indent=2),
                         encoding="utf-8")
    temporary.replace(path)


def verify_files(root: Path, hashes: dict) -> None:
    for name, expected in hashes.items():
        if sha256(root / name) != expected:
            raise ValueError(f"Changed input: {root / name}")