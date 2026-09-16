"""Verify the locked environment and real CPU/CUDA convolution execution."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    import numpy as np
    import pyarrow as pa
    import torch

    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts" / "environment"
    # This command has no source-data argument and writes only under the project.
    if not output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Environment report output escaped project root")
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pyarrow", "pillow", "pytest", "ruff", "mypy")
        },
        "lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "arrow_roundtrip": pa.array(np.array([1, 2, 3])).to_pylist() == [1, 2, 3],
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: G1.1 requires the local GPU verification")
        torch.manual_seed(17)
        device_checks: dict[str, object] = {}
        for device in ("cpu", "cuda"):
            model = torch.nn.Conv2d(1, 8, kernel_size=3, padding=1).to(device)
            inputs = torch.randn(8, 1, 64, 60, device=device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            before = model.weight.detach().clone()
            loss = model(inputs).square().mean()
            loss.backward()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss on {device}")
            for parameter in model.parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"Missing/non-finite gradient on {device}")
            optimizer.step()
            if torch.equal(before, model.weight):
                raise RuntimeError(f"Optimizer failed to change weights on {device}")
            if device == "cuda":
                torch.cuda.synchronize()
            device_checks[device] = {"loss": loss.item(), "optimizer_update": True}
        report["device_checks"] = device_checks
        report["gpu"] = torch.cuda.get_device_name(0)
        report["compute_capability"] = list(torch.cuda.get_device_capability(0))
        report["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        report["torch_arch_list"] = torch.cuda.get_arch_list()
        report["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        target = output / f"check-{stamp}.json"
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"report": str(target), **report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
