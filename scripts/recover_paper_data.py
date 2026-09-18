"""One explicitly authorized conditional range recovery, preserving failed evidence."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import psutil
from paper_data import (
    ARCHIVE_BYTES,
    LIMIT_BYTES,
    LIMIT_SECONDS,
    ROOT,
    URL,
    audit_year,
    selected_members,
    sha256,
)

RUN = ROOT / "artifacts/paper-v1/acquire-20260917T010443459965Z"
OFFSET = 550598230
ETAG = '"21d28cfc-64bb36a60b7b0"'
OLD_SECONDS = 301.547


def validate_response(status: str, headers: dict[str, str]) -> None:
    if len(status.split()) < 2 or status.split()[1] != "206":
        raise ValueError(f"Expected 206, got {status!r}; no body accepted")
    if headers.get("etag") != ETAG:
        raise ValueError("ETag changed or absent")
    expected = f"bytes {OFFSET}-{ARCHIVE_BYTES - 1}/{ARCHIVE_BYTES}"
    if headers.get("content-range") != expected:
        raise ValueError("Unexpected Content-Range")
    if headers.get("content-length") != str(ARCHIVE_BYTES - OFFSET):
        raise ValueError("Unexpected Content-Length")
    if headers.get("content-encoding", "identity") != "identity":
        raise ValueError("Encoded response cannot be appended")


def worker(directory: Path, started: float) -> None:
    deadline = started + LIMIT_SECONDS - OLD_SECONDS
    receipt = {"status": "running", "parent_run": RUN.name, "files": [], "years": [],
               "old_seconds": OLD_SECONDS, "script_sha256": sha256(Path(__file__))}

    def save() -> None:
        receipt["recovery_seconds"] = time.monotonic() - started
        receipt["cumulative_seconds"] = OLD_SECONDS + receipt["recovery_seconds"]
        (directory / "manifest.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    def budget() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("Cumulative time budget exhausted")
        if sum(p.stat().st_size for p in RUN.rglob("*") if p.is_file()) > LIMIT_BYTES:
            raise OSError("Cumulative disk budget exhausted")

    def copy(incoming, outgoing) -> None:
        while chunk := incoming.read(1024 * 1024):
            budget()
            outgoing.write(chunk)

    save()
    try:
        partial = RUN / "img_data.zip"
        old = json.loads((RUN / "manifest.json").read_text())
        handoff = json.loads((RUN.parent / "p1-handoff.json").read_text())
        expected = next(x["sha256"] for x in handoff["files"] if x["path"] == "img_data.zip")
        if old["status"] != "failed" or old["wall_seconds"] != OLD_SECONDS:
            raise ValueError("Parent attempt changed")
        if partial.stat().st_size != OFFSET or sha256(partial) != expected:
            raise ValueError("Partial archive changed")
        receipt["parent_manifest_sha256"] = sha256(RUN / "manifest.json")
        receipt["partial_sha256"] = expected
        budget()
        command = ["curl.exe", "--silent", "--show-error", "--no-buffer", "--include",
                   "--connect-timeout", "30", "--max-time", str(max(1, int(deadline-time.monotonic()))),
                   "--max-filesize", str(ARCHIVE_BYTES-OFFSET),
                   "--range", f"{OFFSET}-{ARCHIVE_BYTES-1}",
                   "--header", f"If-Range: {ETAG}", URL]
        receipt["command"] = command
        save()
        with (directory / "curl-stderr.log").open("xb") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log)
            assert process.stdout is not None
            try:
                raw = bytearray()
                while True:
                    line = process.stdout.readline(65537)
                    raw.extend(line)
                    if not line or len(raw) > 65536:
                        raise ValueError("Missing/oversize HTTP response headers")
                    if line in (b"\r\n", b"\n"):
                        break
                (directory / "response-headers.txt").write_bytes(raw)
                lines = raw.decode("iso-8859-1").splitlines()
                headers = {}
                for line in lines[1:]:
                    if not line:
                        continue
                    key, value = line.split(":", 1)
                    key = key.lower().strip()
                    if key in headers:
                        raise ValueError("Duplicate HTTP header")
                    headers[key] = value.strip()
                validate_response(lines[0], headers)
                receipt["validated_response"] = {"status": lines[0], **headers}
                save()
                tail = directory / "remaining.bin"
                received = 0
                with tail.open("xb") as output:
                    while chunk := process.stdout.read(1024 * 1024):
                        budget()
                        received += len(chunk)
                        if received > ARCHIVE_BYTES - OFFSET:
                            raise ValueError("Response body exceeds authorized range")
                        output.write(chunk)
                code = process.wait(timeout=max(0.01, deadline-time.monotonic()))
                receipt["curl_exit_code"] = code
                if code or received != ARCHIVE_BYTES - OFFSET:
                    raise ValueError("Incomplete range transfer")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                receipt.setdefault("curl_exit_code", process.returncode)
                process.stdout.close()
        archive = directory / "img_data.zip"
        with archive.open("xb") as output:
            for source in (partial, tail):
                with source.open("rb") as incoming:
                    copy(incoming, output)
        if archive.stat().st_size != ARCHIVE_BYTES:
            raise ValueError("Combined archive length mismatch")
        receipt["archive_sha256"] = sha256(archive)
        with zipfile.ZipFile(archive) as source:
            selected = selected_members(source.infolist())
            projected = sum(p.stat().st_size for p in RUN.rglob("*") if p.is_file())
            if projected + sum(x.file_size for x in selected) + 1024**2 > LIMIT_BYTES:
                raise OSError("Projected cumulative disk budget exceeded")
            for info in selected:
                target = directory / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(info) as incoming, target.open("xb") as output:
                    copy(incoming, output)
                if target.stat().st_size != info.file_size:
                    raise ValueError("Member length mismatch")
                receipt["files"].append({"path": info.filename, "bytes": info.file_size,
                                         "crc32": info.CRC, "sha256": sha256(target)})
                save()
        for year in range(1993, 2001):
            budget()
            receipt["years"].append(audit_year(directory / "monthly_20d", year))
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
        save()


def main() -> None:
    directory = RUN / "recovery-1"
    directory.mkdir(exist_ok=False)  # One authorized attempt, never silently reused.
    started = time.monotonic()
    command = [sys.executable, str(Path(__file__)), "--worker", str(started)]
    supervision = {"status": "running", "old_seconds": OLD_SECONDS,
                   "remaining_seconds": LIMIT_SECONDS-OLD_SECONDS, "command": command}
    with (directory / "worker.log").open("xb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            code = child.wait(timeout=max(0.01, LIMIT_SECONDS-OLD_SECONDS-(time.monotonic()-started)))
            supervision.update(status="completed" if code == 0 else "failed", exit_code=code)
        except subprocess.TimeoutExpired:
            for process in psutil.Process(child.pid).children(recursive=True):
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            child.kill()
            child.wait()
            supervision.update(status="timeout", exit_code=child.returncode)
        finally:
            supervision["recovery_seconds"] = time.monotonic()-started
            supervision["cumulative_seconds"] = OLD_SECONDS+supervision["recovery_seconds"]
            supervision["total_bytes"] = sum(p.stat().st_size for p in RUN.rglob("*") if p.is_file())
            (directory / "supervision.json").write_text(json.dumps(supervision, indent=2))
    print(json.dumps(supervision), flush=True)
    if supervision["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(RUN / "recovery-1", float(sys.argv[2]))
    else:
        main()
