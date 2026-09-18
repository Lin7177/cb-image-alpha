"""P3 one-shot process supervisor. Mechanical monitoring, no retry or resume."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import psutil

GIB = 1024**3


def append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def size(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def stop_tree(process: subprocess.Popen) -> None:
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
    except psutil.NoSuchProcess:
        pass
    process.wait()


def supervise(
    command: list[str],
    folder: Path,
    artifact_root: Path,
    seconds: float,
    *,
    rss_limit: int = 8 * GIB,
    disk_limit: int = 8 * GIB,
) -> dict:
    """A new segment only. Existing segment evidence is never overwritten."""
    folder.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    ledger = folder / "ledger.jsonl"
    append(
        ledger,
        {
            "event": "starting",
            "elapsed": 0.0,
            "pid": os.getpid(),
            "command": command,
            "wall_limit": seconds,
        },
    )
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONUNBUFFERED"] = "1"
    env["PAPER_P3_LEDGER"] = str(ledger.resolve())
    reason, peak, next_ledger, code = None, 0, 0.0, None
    process = None
    try:
        with (
            (folder / "stdout.log").open("xb") as stdout,
            (folder / "stderr.log").open("xb") as stderr,
        ):
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env)
            me = psutil.Process()
            while True:
                elapsed = time.monotonic() - start
                rss = 0
                for item in [me, *me.children(recursive=True)]:
                    try:
                        rss += item.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, rss)
                used = size(artifact_root)
                if elapsed >= seconds:
                    reason = "wall_limit"
                elif rss > rss_limit:
                    reason = "rss_tree_limit"
                elif used > disk_limit:
                    reason = "artifact_limit"
                elif shutil.disk_usage(artifact_root).free < 256 * 1024**2:
                    reason = "disk_exhaustion"
                if elapsed >= next_ledger or reason is not None:
                    append(
                        ledger,
                        {
                            "event": "sample",
                            "elapsed": elapsed,
                            "rss_tree": rss,
                            "artifact_bytes": used,
                            "reason": reason,
                        },
                    )
                    next_ledger = elapsed + 5
                code = process.poll()
                if reason is not None:
                    if code is None:
                        stop_tree(process)
                        code = process.returncode
                    break
                if code is not None:
                    break
                time.sleep(1)
    except BaseException as error:  # noqa: BLE001 -- terminal boundary preserves failure evidence
        reason = f"{type(error).__name__}: {error}"
        if process is not None and process.poll() is None:
            stop_tree(process)
        code = process.returncode if process else None
    result = {
        "status": "passed" if code == 0 and reason is None else "failed",
        "exit_code": code,
        "reason": reason,
        "elapsed": time.monotonic() - start,
        "peak_rss_tree": peak,
        "artifact_bytes": size(artifact_root),
        "scope": "supervisor_plus_all_descendants_sampled_1s",
        "automatic_retry": False,
    }
    append(ledger, {"event": "finished", **result})
    with (folder / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    return result


def ledger_snapshot() -> dict:
    path = Path(os.environ["PAPER_P3_LEDGER"])
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    elapsed = [x["elapsed"] for x in records]
    if elapsed != sorted(elapsed) or records[0]["event"] != "starting":
        raise RuntimeError("Invalid external resource ledger")
    return {"path": str(path), "records": len(records), "last": records[-1]}


def check_space(root: Path, projected: int, reserve: int = 128 * 1024**2) -> None:
    if size(root) + projected + reserve > 8 * GIB:
        raise RuntimeError("Projected epoch plus seal exceeds artifact budget")
    if shutil.disk_usage(root).free < projected + reserve:
        raise RuntimeError("Insufficient space for full epoch plus seal")


if __name__ == "__main__":
    raise SystemExit("Use the paper_p3 CLI; direct supervisor invocation is not supported")
