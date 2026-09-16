"""Fixed G8 estimators. No hyperparameter search or outer-set selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from g8_data import FoldImages, sha256_array
from torch import nn

from cb_image_alpha.artifacts import read_json, sha256, write_json
from cb_image_alpha.models.paper_cnn import PaperCNN
from cb_image_alpha.training.engine import EarlyStop


def logistic_fit(
    x: np.ndarray, labels: np.ndarray, config: dict, report_path: Path | None = None
) -> tuple[dict, dict]:
    mean = x.mean(0)
    std = x.std(0, ddof=0)
    zero = std == 0
    scale = std.copy()
    scale[zero] = 1
    z = (x - mean) / scale
    z[:, zero] = 0
    tx = torch.from_numpy(np.asarray(z, dtype=np.float64))
    ty = torch.from_numpy(np.asarray(labels, dtype=np.float64))
    weight = nn.Parameter(torch.zeros(x.shape[1], dtype=torch.float64))
    bias = nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=config["logistic_lr"],
        max_iter=config["logistic_iterations"],
        max_eval=config["logistic_evaluations"],
        tolerance_grad=config["logistic_grad_tolerance"],
        tolerance_change=config["logistic_change_tolerance"],
        history_size=config["logistic_history"],
        line_search_fn=config["logistic_line_search"],
    )
    evaluations = 0
    last_parameters = None
    last_gradient = None
    last_loss = None

    def closure():
        nonlocal evaluations, last_parameters, last_gradient, last_loss
        if evaluations >= config["logistic_evaluations"]:
            raise RuntimeError("Logistic evaluation budget exhausted")
        evaluations += 1
        optimizer.zero_grad(set_to_none=True)
        loss = (
            nn.functional.binary_cross_entropy_with_logits(tx @ weight + bias, ty)
            + config["logistic_lambda"] * weight.square().sum() / 2
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite logistic loss")
        loss.backward()
        gradient = torch.cat((weight.grad, bias.grad.reshape(1)))
        if not torch.isfinite(gradient).all():
            raise RuntimeError("Nonfinite logistic gradient")
        last_parameters = torch.cat((weight.detach().clone(), bias.detach().reshape(1).clone()))
        last_gradient = float(gradient.abs().max())
        last_loss = float(loss.detach())
        return loss

    failure = None
    try:
        optimizer.step(closure)
        current = torch.cat((weight.detach(), bias.detach().reshape(1)))
        if last_parameters is None or not torch.equal(current, last_parameters):
            closure()
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report = {
            "evaluations": evaluations,
            "iterations": int(optimizer.state.get(weight, {}).get("n_iter", 0)),
            "gradient_inf": last_gradient,
            "objective": last_loss,
            "failure": failure,
            "converged": failure is None
            and last_gradient is not None
            and last_gradient <= config["logistic_grad_tolerance"],
        }
        if report_path is not None:
            write_json(report_path, report)
    model = {
        "weight": weight.detach().numpy(),
        "bias": np.array(float(bias.detach())),
        "mean": mean,
        "std": scale,
        "zero_columns": zero,
    }
    return model, report


def logistic_predict(model: dict, x: np.ndarray) -> np.ndarray:
    z = (x - model["mean"]) / model["std"]
    z[:, model["zero_columns"]] = 0
    score = z @ model["weight"] + model["bias"]
    result = np.empty_like(score)
    positive = score >= 0
    result[positive] = 1 / (1 + np.exp(-score[positive]))
    exp = np.exp(score[~positive])
    result[~positive] = exp / (1 + exp)
    return result


def logistic_stage(run: Path) -> None:
    config = read_json(run / "plan.json")["config"]
    labels = pq.read_table(run / "index.parquet", columns=["label"])["label"].to_numpy()
    for year in config["fold_years"]:
        folder = run / str(year)
        train = np.load(folder / "final_train.npy")
        outer = np.load(folder / "outer.npy")
        rate = float(labels[train].mean())
        write_json(
            folder / "constant.json",
            {
                "train_positive_rate": rate,
                "train_count": len(train),
                "train_indices_sha256": sha256_array(train),
            },
        )
        np.save(folder / "Constant-p.npy", np.full(len(outer), rate), allow_pickle=False)
        for name in ("U", "Q"):
            write_json(
                folder / f"{name}-attempt.json",
                {
                    "status": "running",
                    "method": name,
                    "fold": year,
                    "configuration": config,
                    "train_indices_sha256": sha256_array(train),
                },
            )
            data = np.load(run / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            try:
                model, report = logistic_fit(
                    data[train], labels[train], config, folder / f"{name}-optimization.json"
                )
            except BaseException as exc:
                write_json(
                    folder / f"{name}-attempt.json",
                    {"status": "failed", "method": name, "fold": year, "error": str(exc)},
                )
                raise
            write_json(folder / f"{name}-optimization.json", report)
            np.savez(folder / f"{name}-model.npz", **model)
            if not report["converged"]:
                write_json(
                    folder / f"{name}-attempt.json",
                    {"status": "failed", "reason": "nonconvergence", "report": report},
                )
                raise RuntimeError(
                    f"Logistic {year}/{name} did not meet frozen convergence criterion"
                )
            probabilities = logistic_predict(model, data[outer])
            if not np.isfinite(probabilities).all():
                raise RuntimeError("Nonfinite probabilities")
            np.save(folder / f"{name}-p.npy", probabilities, allow_pickle=False)
            write_json(
                folder / f"{name}-attempt.json",
                {
                    "status": "completed",
                    "method": name,
                    "fold": year,
                    "configuration": config,
                    "train_indices_sha256": sha256_array(train),
                    "outer_indices_sha256": sha256_array(outer),
                    "model_sha256": sha256(folder / f"{name}-model.npz"),
                    "report": report,
                },
            )
            del data, model


@torch.no_grad()
def loss_on(model: nn.Module, data: FoldImages, batch_size: int) -> float:
    model.eval()
    total = 0.0
    for start in range(0, len(data.indices), batch_size):
        x, y = data.batch(np.arange(start, min(start + batch_size, len(data.indices))))
        total += float(
            nn.functional.cross_entropy(model(x.cuda()), y.cuda(), reduction="sum").cpu()
        )
    result = total / len(data.indices)
    if not np.isfinite(result):
        raise ValueError("Nonfinite validation loss")
    return result


def cnn_stage(run: Path, year: int, mode: str) -> None:
    config = read_json(run / "plan.json")["config"]
    folder = run / str(year)
    torch.cuda.set_per_process_memory_fraction(config["gpu_memory_fraction"])
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    role = "inner_train" if mode == "inner" else "final_train"
    indices = np.load(folder / f"{role}.npy")
    data = FoldImages(run, indices)
    write_json(folder / f"{mode}-normalization.json", data.normalization)
    valid = (
        FoldImages(run, np.load(folder / "inner_validation.npy"), data.normalization)
        if mode == "inner"
        else None
    )
    epochs = (
        config["max_epochs"]
        if mode == "inner"
        else read_json(folder / "inner-summary.json")["best_epoch"]
    )
    model = PaperCNN().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    generator = torch.Generator().manual_seed(config["seed"])
    stopper = EarlyStop(config["patience"])
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        order = torch.randperm(len(indices), generator=generator).numpy()
        for start in range(0, len(order), config["batch_size"]):
            x, y = data.batch(order[start : start + config["batch_size"]])
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x.cuda()), y.cuda())
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            gradients = [
                torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
            ]
            if len(gradients) != len(list(model.parameters())) or not torch.stack(gradients).all():
                raise ValueError("Invalid gradients")
            optimizer.step()
            total += float(loss.detach().cpu()) * len(y)
        record = {"epoch": epoch, "train_loss": total / len(indices)}
        improved = False
        if valid is not None:
            value = loss_on(model, valid, config["batch_size"])
            improved = stopper.observe(value, epoch)
            record["validation_loss"] = value
        history.append(record)
        write_json(folder / f"{mode}-history.json", history)
        checkpoint = {
            "scope": "development",
            "fold": year,
            "mode": mode,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "shuffle_rng": generator.get_state(),
            "plan_sha256": sha256(run / "plan.json"),
            "training_indices_sha256": sha256_array(indices),
            "normalization": data.normalization,
        }
        for name in [f"{mode}-latest.pt", f"{mode}-best.pt"] if improved else [f"{mode}-latest.pt"]:
            torch.save(checkpoint, folder / (name + ".tmp"))
            (folder / (name + ".tmp")).replace(folder / name)
        if mode == "inner" and stopper.stopped:
            break
    write_json(
        folder / f"{mode}-summary.json",
        {
            "status": "completed",
            "epochs": epoch,
            "best_epoch": stopper.epoch if mode == "inner" else epochs,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "training_indices_sha256": sha256_array(indices),
        },
    )
    if mode == "final":
        checkpoint = torch.load(folder / "final-latest.pt", map_location="cuda", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        outer = np.load(folder / "outer.npy")
        test = FoldImages(run, outer, data.normalization)
        p = np.empty(len(outer), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(outer), config["batch_size"]):
                positions = np.arange(start, min(start + config["batch_size"], len(outer)))
                x, _ = test.batch(positions)
                p[positions] = model(x.cuda()).softmax(1)[:, 1].cpu().numpy()
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Invalid CNN probabilities")
        np.save(folder / "CNN-p.npy", p, allow_pickle=False)
