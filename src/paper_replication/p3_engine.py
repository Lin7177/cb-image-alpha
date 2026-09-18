"""Production P3 training engine, shared without a toy alternative by both preflights."""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cb_image_alpha.models.paper_cnn import PaperCNN
from paper_replication.p3_checkpoint import GateError, commit, durable_json, load_latest
from paper_replication.p3_supervision import check_space, ledger_snapshot


def configure(device: str, seed: int) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise GateError("CUBLAS deterministic environment missing")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "cuda":
        if not torch.cuda.is_available():
            raise GateError("CUDA required but unavailable")
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, (6 * 1024**3) / total), 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def rng_state(train_rng, val_rng):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "train": train_rng.get_state(),
        "validation": val_rng.get_state(),
    }


def restore_rng(state, train_rng, val_rng):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    train_rng.set_state(state["train"])
    val_rng.set_state(state["validation"])


def finite(tensor, name: str):
    if not torch.isfinite(tensor).all():
        raise GateError(f"Nonfinite {name}")


def coverage(observed, expected):
    actual = np.asarray(observed, dtype=np.int64)
    if (
        len(actual) != len(expected)
        or len(np.unique(actual)) != len(actual)
        or not np.array_equal(np.sort(actual), np.sort(expected))
    ):
        raise GateError("Duplicate/missing epoch IDs")


def cuda_budget(device):
    if device == "cuda" and torch.cuda.memory_reserved() > 6 * 1024**3:
        raise GateError("CUDA reserved limit")


def probe(model, dataset, device, train_rng, val_rng):
    saved = rng_state(train_rng, val_rng)
    was_training = model.training
    try:
        model.eval()
        images = torch.stack([dataset[i][0] for i in range(min(128, len(dataset)))])
        with torch.no_grad():
            output = model(images.to(device)).softmax(1).cpu()
        finite(output, "probe")
        return output
    finally:
        model.train(was_training)
        restore_rng(saved, train_rng, val_rng)


def run(
    folder: Path,
    artifact_root: Path,
    train,
    validation,
    meta: dict,
    identity: dict,
    device: str,
    *,
    epochs: int = 50,
    resume: bool = False,
    interrupt_after: int | None = None,
    fault: str = "",
    synthetic: bool = False,
) -> dict:
    """No internal retries. Unexpected exceptions are fatal to the whole batch."""
    if (epochs != 50 or interrupt_after is not None or fault) and not synthetic:
        raise GateError("Real training cannot override epochs or inject faults")
    configure(device, meta["model_seed"])
    model = PaperCNN().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        foreach=False,
        fused=False,
    )
    train_rng = torch.Generator().manual_seed(meta["shuffle_seed"])
    val_rng = torch.Generator().manual_seed(meta["shuffle_seed"] + 1000000)
    train_loader = DataLoader(
        train,
        batch_size=128,
        shuffle=True,
        generator=train_rng,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )
    val_loader = DataLoader(
        validation,
        batch_size=128,
        shuffle=False,
        generator=val_rng,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )
    epoch, step, best_loss, best_epoch, stale, history, parent = 0, 0, float("inf"), 0, 0, [], None

    def snapshot():
        return {
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "rng": rng_state(train_rng, val_rng),
            "epoch": epoch,
            "global_step": step,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "stale": stale,
            "history": copy.deepcopy(history),
            "identity": identity,
            "normalization": meta["normalization"],
            "ledger": ledger_snapshot(),
        }

    if resume:
        state, parent = load_latest(folder, identity)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"], train_rng, val_rng)
        epoch, step = state["epoch"], state["global_step"]
        best_loss, best_epoch, stale = state["best_loss"], state["best_epoch"], state["stale"]
        history = state["history"]
        # Partial files remain billed; a resumed epoch never overwrites them.
        for partial in sorted(folder.glob("epoch-*")):
            if not (partial / "COMMIT.json").exists():
                partial.rename(folder / ("partial-" + partial.name))
    else:
        folder.mkdir(parents=True, exist_ok=False)
        durable_json(
            folder / "started.json",
            {"identity": identity, "synthetic": synthetic, "meta": meta, "status": "started"},
        )
        check_space(artifact_root, 32 * 1024**2 + len(validation) * 32)
        parent = commit(folder, snapshot(), {}, identity, parent)

    while epoch < epochs and stale < 2:
        epoch += 1
        check_space(artifact_root, 32 * 1024**2 + len(validation) * 32)
        model.train()
        train_ids, train_sum, train_count = [], 0.0, 0
        for batch_index, (images, labels, ids) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            losses = nn.functional.cross_entropy(logits, labels, reduction="none")
            if fault == "nonfinite_loss" and epoch == 1 and batch_index == 0:
                losses = losses * float("nan")
            finite(losses, "loss")
            losses.mean().backward()
            if fault == "nonfinite_gradient" and epoch == 1 and batch_index == 0:
                next(model.parameters()).grad.fill_(float("nan"))
            for parameter in model.parameters():
                if parameter.grad is None:
                    raise GateError("Missing gradient")
                finite(parameter.grad, "gradient")
            optimizer.step()
            for parameter in model.parameters():
                finite(parameter, "parameter")
            step += 1
            train_ids.extend(ids.tolist())
            train_sum += losses.detach().double().sum().item()
            train_count += len(ids)
            cuda_budget(device)
            if fault == "interrupt_during_epoch" and epoch == 2 and batch_index == 0:
                durable_json(folder / "injected-interruption.json", {"epoch": epoch, "step": step})
                return {"status": "injected_interruption", "epoch": epoch}
        coverage(train_ids, train.ids)
        model.eval()
        val_ids, val_labels, probabilities, val_losses = [], [], [], []
        with torch.no_grad():
            for images, labels, ids in val_loader:
                logits = model(images.to(device))
                losses = nn.functional.cross_entropy(logits, labels.to(device), reduction="none")
                probability = logits.softmax(1)
                finite(losses, "validation loss")
                finite(probability, "probability")
                val_ids.append(ids)
                val_labels.append(labels)
                probabilities.append(probability.cpu())
                val_losses.append(losses.cpu())
                cuda_budget(device)
        predictions = {
            "sample_id": torch.cat(val_ids),
            "label": torch.cat(val_labels),
            "probability": torch.cat(probabilities),
            "loss": torch.cat(val_losses),
            "train_order": torch.tensor(train_ids, dtype=torch.int64),
        }
        coverage(predictions["sample_id"].numpy(), validation.ids)
        if not np.array_equal(predictions["sample_id"].numpy(), validation.ids):
            raise GateError("Validation order changed")
        loss = predictions["loss"].double().sum().item() / len(validation)
        if loss < best_loss:
            best_loss, best_epoch, stale = loss, epoch, 0
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "global_step": step,
                "train_loss": train_sum / train_count,
                "validation_loss": loss,
                "best_epoch": best_epoch,
                "stale": stale,
            }
        )
        # Probe each synthetic epoch; restoring RNG makes it numerically inert.
        if synthetic:
            from paper_replication.p3_checkpoint import exact

            before = rng_state(train_rng, val_rng)
            exact(
                probe(model, validation, device, train_rng, val_rng),
                predictions["probability"][:128],
                "epoch_probe",
            )
            exact(before, rng_state(train_rng, val_rng), "probe_rng")
        parent = commit(
            folder, snapshot(), predictions, identity, parent, fault if epoch == 1 else ""
        )
        if interrupt_after == epoch:
            return {"status": "injected_interruption", "epoch": epoch}
    result = {
        "status": "early_stopped" if stale >= 2 else "capped",
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "synthetic": synthetic,
    }
    durable_json(folder / "terminal.json", result)
    return result
