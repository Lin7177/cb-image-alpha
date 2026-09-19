"""Paper-faithful I20/R20 decile portfolio analysis from sealed predictions.

This script is an independent implementation of the documented PortfolioManager
behavior. It reads only the already-authorized paper OOS archive and sealed r3
predictions; it never trains, selects a model, or changes the sample.
"""
from __future__ import annotations

import hashlib
import json
import tomllib
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from pyarrow import feather

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/paper-v1/p4-reval-20260919-r3"
CFG = ROOT / "configs/paper/p4_reval_v1.toml"
INV = ROOT / "artifacts/paper-v1/p4-oos-inventory.json"
ARCHIVE = ROOT / "artifacts/paper-v1/acquire-20260917T010443459965Z/recovery-1/img_data.zip"
OUTPUT = RUN / "portfolio-i20-r20"
YEARS = range(2001, 2020)
CUT = 10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_predictions() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    ids: np.ndarray | None = None
    labels: np.ndarray | None = None
    probabilities: dict[str, np.ndarray] = {}
    for member in range(5):
        data = np.load(RUN / f"member-{member}-predictions.npz")
        current_ids = np.asarray(data["sample_id"])
        current_labels = np.asarray(data["label"])
        current_probability = np.asarray(data["probability"], dtype=np.float64)
        if ids is None:
            ids, labels = current_ids, current_labels
        elif not np.array_equal(ids, current_ids) or not np.array_equal(labels, current_labels):
            raise ValueError("member sample_id/label mismatch")
        if not np.isfinite(current_probability).all() or np.any((current_probability < 0) | (current_probability > 1)):
            raise ValueError("non-finite or out-of-range probability")
        probabilities[f"member-{member}"] = current_probability
    assert ids is not None and labels is not None
    probabilities["ensemble"] = np.mean(
        np.stack([probabilities[f"member-{m}"] for m in range(5)]), axis=0
    )
    return ids, labels, probabilities


def load_context(ids: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    context_ids: list[np.ndarray] = []
    context_labels: list[np.ndarray] = []
    dates: list[np.ndarray] = []
    returns: list[np.ndarray] = []
    market_caps: list[np.ndarray] = []
    stock_ids: list[np.ndarray] = []
    inventory = json.loads(INV.read_text(encoding="utf-8"))
    expected = {item["name"] for item in inventory["data_members"]}
    with zipfile.ZipFile(ARCHIVE) as archive:
        for year in YEARS:
            image_name = f"monthly_20d/20d_month_has_vb_[20]_ma_{year}_images.dat"
            label_name = f"monthly_20d/20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather"
            if image_name not in expected or label_name not in expected:
                raise ValueError(f"missing expected member for {year}")
            table = feather.read_table(BytesIO(archive.read(label_name)))
            required = {"Date", "StockID", "MarketCap", "Ret_month", "EWMA_vol"}
            if not required.issubset(table.column_names):
                raise ValueError(f"missing context fields for {year}")
            n = table.num_rows
            ret = np.asarray(table["Ret_month"].to_pylist(), dtype=np.float64)
            vol = np.asarray(table["EWMA_vol"].to_pylist(), dtype=np.float64)
            keep = np.isfinite(ret) & (ret != -99) & np.isfinite(vol) & (vol > 0)
            row = np.flatnonzero(keep)
            current_ids = year * 1_000_000_000 + row
            current_labels = (ret[keep] > 0).astype(np.int64)
            if not np.array_equal(current_labels, labels[len(np.concatenate(context_labels)) : len(np.concatenate(context_labels)) + len(current_labels)] if context_labels else current_labels):
                # The global check below is authoritative; this avoids silently changing label rules.
                pass
            context_ids.append(current_ids)
            context_labels.append(current_labels)
            dates.append(np.asarray(table["Date"].to_pylist(), dtype="datetime64[D]")[keep])
            returns.append(ret[keep])
            market_caps.append(np.asarray([np.nan if value is None else float(value) for value in table["MarketCap"].to_pylist()], dtype=np.float64)[keep])
            stock_ids.append(np.asarray(table["StockID"].to_pylist(), dtype=str)[keep])
    joined_ids = np.concatenate(context_ids)
    joined_labels = np.concatenate(context_labels)
    if not np.array_equal(joined_ids, ids) or not np.array_equal(joined_labels, labels):
        raise ValueError("context/sample prediction mapping mismatch")
    return {
        "sample_id": joined_ids,
        "label": joined_labels,
        "date": np.concatenate(dates),
        "ret": np.concatenate(returns),
        "market_cap": np.concatenate(market_caps),
        "stock_id": np.concatenate(stock_ids),
    }


def portfolio(probability: np.ndarray, context: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    date_values = np.unique(context["date"])
    ew = np.zeros((len(date_values), CUT), dtype=np.float64)
    vw = np.zeros((len(date_values), CUT), dtype=np.float64)
    skipped = 0
    usable_rows = 0
    for date_index, date in enumerate(date_values):
        row = context["date"] == date
        row &= np.isfinite(probability) & np.isfinite(context["ret"])
        row &= np.isfinite(context["market_cap"])
        p = probability[row]
        ret = context["ret"][row]
        cap = np.abs(context["market_cap"][row])
        usable_rows += int(row.sum())
        if len(p) == 0 or np.percentile(p, 10) == np.percentile(p, 90):
            skipped += 1
            continue
        for decile in range(CUT):
            low = np.percentile(p, decile * 100.0 / CUT)
            high = np.percentile(p, (decile + 1) * 100.0 / CUT)
            mask = (p >= low) & (p <= high) if decile == 0 else (p > low) & (p <= high)
            if not mask.any():
                continue
            ew[date_index, decile] = float(np.mean(ret[mask]))
            weight_sum = float(np.sum(cap[mask]))
            vw[date_index, decile] = float(np.sum((cap[mask] / weight_sum) * ret[mask])) if weight_sum > 0 else 0.0
    ew = np.column_stack([ew, ew[:, -1] - ew[:, 0]])
    vw = np.column_stack([vw, vw[:, -1] - vw[:, 0]])
    return ew, vw, {"dates": len(date_values), "skipped_dates": skipped, "usable_rows": usable_rows}


def summary(matrix: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    labels = ["low"] + [f"d{i}" for i in range(2, CUT)] + ["high", "H-L"]
    for column, label in enumerate(labels):
        series = matrix[:, column]
        mean = float(np.mean(series))
        std = float(np.std(series, ddof=1))
        out[label] = {
            "mean_monthly": mean,
            "annualized_return": mean * 12.0,
            "annualized_volatility": std * np.sqrt(12.0),
            "annualized_sharpe": (mean * 12.0) / (std * np.sqrt(12.0)) if std > 0 else None,
        }
    return out


def main() -> None:
    config = tomllib.loads(CFG.read_text(encoding="utf-8"))
    result = json.loads((RUN / "results.json").read_text(encoding="utf-8"))
    if result["status"] != "COMPLETED_AWAITING_AUDIT" or result["run_id"] != "p4-reval-20260919-r3":
        raise RuntimeError("sealed r3 result gate")
    ids, labels, probabilities = load_predictions()
    context = load_context(ids, labels)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    payload: dict[str, object] = {
        "status": "COMPLETED_AWAITING_AUDIT",
        "analysis": "paper_i20_r20_decile_portfolio",
        "run_id": result["run_id"],
        "protocol_sha256": sha256(CFG),
        "archive_sha256": sha256(ARCHIVE),
        "inventory_sha256": sha256(INV),
        "rows": int(len(ids)),
        "date_min": str(context["date"].min()),
        "date_max": str(context["date"].max()),
        "cut": CUT,
        "frequency": "month",
        "no_training": True,
        "no_tuning": True,
        "test_qualification": "none",
        "endpoint_verified": False,
        "strict_leakage_free_claim": False,
        "signals": {},
    }
    for name, probability in probabilities.items():
        ew, vw, counts = portfolio(probability, context)
        np.savez_compressed(OUTPUT / f"{name}.npz", dates=context["date"].astype("datetime64[D]").astype(str), ew=ew, vw=vw)
        payload["signals"][name] = {"counts": counts, "ew": summary(ew), "vw": summary(vw)}
    (OUTPUT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    files = []
    for path in sorted(OUTPUT.iterdir()):
        files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    (OUTPUT / "manifest.json").write_text(json.dumps({"status": "SEALED_AWAITING_AUDIT", "files": files, "inputs": payload}, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
