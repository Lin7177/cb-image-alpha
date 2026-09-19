"""Independent paper-style period metrics for the sealed I20/R20 predictions."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from paper_p4_portfolio import CFG, OUTPUT, RUN, load_context, load_predictions, sha256


def pooled(probability: np.ndarray, label: np.ndarray) -> dict[str, float | None]:
    pred = (probability > 0.5).astype(np.int64)
    n = len(pred)
    tp = float(np.sum(pred * label)) / n
    tn = float(np.sum((pred - 1) * (label - 1))) / n
    fp = float(abs(np.sum(pred * (label - 1)))) / n
    fn = float(abs(np.sum((pred - 1) * label))) / n
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = None if denom == 0 else float((tp * tn - fp * fn) / math.sqrt(denom))
    loss = float(-np.mean(label * np.log(np.clip(probability, 1e-12, 1.0)) + (1 - label) * np.log(np.clip(1 - probability, 1e-12, 1.0))))
    return {
        "rows": int(n),
        "loss": loss,
        "accy": float(tp + tn),
        "diff": float((tp + fp) - (tn + fn)),
        "MCC": mcc,
    }


def corr(x: np.ndarray, y: np.ndarray, spearman: bool) -> float | None:
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    if spearman:
        x = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(float)
        y = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort").astype(float)
    return float(np.corrcoef(x, y)[0, 1])


def by_period(probability: np.ndarray, context: dict[str, np.ndarray]) -> dict[str, object]:
    dates = context["date"]
    labels = (context["ret"] > 0).astype(np.int64)
    date_values = np.unique(dates)
    rows = []
    for date in date_values:
        mask = dates == date
        p = probability[mask]
        ret = context["ret"][mask]
        pred = (p > 0.5).astype(np.int64)
        rows.append({
            "date": str(date),
            "rows": int(mask.sum()),
            "accuracy": float(np.mean(pred == labels[mask])),
            "spearman": corr(p, ret, True),
            "pearson": corr(p, ret, False),
            "positive_rate": float(np.mean(labels[mask])),
        })
    def mean(field: str) -> float | None:
        values = [row[field] for row in rows if row[field] is not None]
        return None if not values else float(np.mean(values))
    yearly = {}
    for year in sorted({str(date)[:4] for date in date_values}):
        subset = [row for row in rows if row["date"].startswith(year)]
        yearly[year] = {
            "dates": len(subset),
            "accuracy_mean": float(np.mean([row["accuracy"] for row in subset])),
            "spearman_mean": float(np.mean([row["spearman"] for row in subset if row["spearman"] is not None])) if any(row["spearman"] is not None for row in subset) else None,
            "pearson_mean": float(np.mean([row["pearson"] for row in subset if row["pearson"] is not None])) if any(row["pearson"] is not None for row in subset) else None,
        }
    return {"dates": len(rows), "accuracy_mean": mean("accuracy"), "spearman_mean": mean("spearman"), "pearson_mean": mean("pearson"), "yearly": yearly}


def main() -> None:
    ids, labels, probabilities = load_predictions()
    ctx = load_context(ids, labels)
    payload = {
        "status": "COMPLETED_AWAITING_AUDIT",
        "analysis": "paper_i20_r20_period_metrics",
        "run_id": json.loads((RUN / "results.json").read_text())["run_id"],
        "protocol_sha256": sha256(CFG),
        "return_field": "Ret_month",
        "return_semantics": "official period-ret sidecar not present; Ret_month label field used as a paper-faithful proxy",
        "test_qualification": "none",
        "endpoint_verified": False,
        "strict_leakage_free_claim": False,
        "no_training": True,
        "no_tuning": True,
        "no_backtest": True,
        "signals": {name: {"pooled": pooled(prob, labels), "period": by_period(prob, ctx)} for name, prob in probabilities.items()},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "period_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({name: value["pooled"] | {k: value["period"][k] for k in ("accuracy_mean", "spearman_mean", "pearson_mean")} for name, value in payload["signals"].items()}, indent=2))


if __name__ == "__main__":
    main()
