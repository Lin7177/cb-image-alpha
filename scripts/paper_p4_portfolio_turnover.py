"""Add official-style top/bottom decile turnover for the I20/R20 analysis."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paper_p4_portfolio import CUT, OUTPUT, load_context, load_predictions


def positions(probability: np.ndarray, ctx: dict[str, np.ndarray], weight_type: str) -> tuple[dict[str, dict[str, float]], int]:
    result: dict[str, dict[str, float]] = {}
    skipped = 0
    for date in np.unique(ctx["date"]):
        row = (ctx["date"] == date) & np.isfinite(probability) & np.isfinite(ctx["ret"]) & np.isfinite(ctx["market_cap"])
        p = probability[row]
        if len(p) == 0 or np.percentile(p, 10) == np.percentile(p, 90):
            skipped += 1
            continue
        ids = ctx["stock_id"][row]
        ret = ctx["ret"][row]
        cap = np.abs(ctx["market_cap"][row])
        weights: dict[str, float] = {}
        for decile in (0, CUT - 1):
            low = np.percentile(p, decile * 100.0 / CUT)
            high = np.percentile(p, (decile + 1) * 100.0 / CUT)
            mask = (p >= low) & (p <= high) if decile == 0 else (p > low) & (p <= high)
            if not mask.any():
                continue
            if weight_type == "ew":
                base = np.full(int(mask.sum()), 1.0 / int(mask.sum()))
            else:
                denom = float(cap[mask].sum())
                base = cap[mask] / denom if denom > 0 else np.zeros(int(mask.sum()))
            sign = -1.0 if decile == 0 else 1.0
            for stock, weight in zip(ids[mask], base, strict=True):
                weights[str(stock)] = sign * float(weight)
        result[str(date)] = {"weights": weights, "returns": {str(stock): float(value) for stock, value in zip(ids, ret, strict=True)}}
    return result, skipped


def turnover(book: dict[str, dict[str, dict[str, float]]]) -> tuple[float | None, int]:
    dates = sorted(book)
    values = []
    previous = None
    for date in dates:
        current = book[date]
        if previous is not None:
            stocks = set(current["weights"]) | set(previous["weights"])
            total_current = sum(current["weights"].get(stock, 0.0) for stock in stocks)
            total_previous_return = sum(previous["weights"].get(stock, 0.0) * previous["returns"].get(stock, 0.0) for stock in stocks)
            denom = 1.0 + total_previous_return
            values.append(0.5 * sum(abs(current["weights"].get(stock, 0.0) - previous["weights"].get(stock, 0.0) * (1.0 + previous["returns"].get(stock, 0.0)) / denom) for stock in stocks))
        previous = current
    return (None if not values else float(np.mean(values)), len(values))


def main() -> None:
    ids, labels, probabilities = load_predictions()
    ctx = load_context(ids, labels)
    payload = {"status": "COMPLETED_AWAITING_AUDIT", "analysis": "paper_i20_r20_decile_turnover", "return_semantics": "Ret_month label proxy; official next_month_ret sidecar unavailable", "signals": {}}
    for name, probability in probabilities.items():
        payload["signals"][name] = {}
        for weight_type in ("ew", "vw"):
            book, skipped = positions(probability, ctx, weight_type)
            value, transitions = turnover(book)
            payload["signals"][name][weight_type] = {"mean_top_bottom_turnover": value, "transitions": transitions, "skipped_dates": skipped}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "turnover.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
