"""Deterministic, calendar-based I20/R5 sample construction."""
from __future__ import annotations

import hashlib
import tomllib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from itertools import pairwise
from pathlib import Path

from cb_image_alpha.data.catalog import RawBar

SHANGHAI = timezone(timedelta(hours=8), "Asia/Shanghai")
VERSION = "g2-v1.2"
EXPECTED = {
    "version": VERSION, "train_start": "2020-01-02", "train_end": "2023-12-31",
    "validation_start": "2024-01-01", "validation_end": "2024-12-31",
    "test_start": "2025-01-01", "test_end": "2026-07-31",
}


@dataclass(frozen=True)
class Partition:
    name: str
    start: date
    end: date


def read_partitions(path: Path) -> tuple[Partition, ...]:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw != EXPECTED:
        raise ValueError("Unsupported split protocol; use the approved exact configuration")
    return tuple(
        Partition(name, date.fromisoformat(raw[f"{name}_start"]),
                  date.fromisoformat(raw[f"{name}_end"]))
        for name in ("train", "validation", "test")
    )


def session(bar: RawBar) -> date:
    return datetime.fromtimestamp(bar.ts_event // 10**9, UTC).date()


def eod(day: date) -> datetime:
    return datetime.combine(day, time(23, 59, 59), SHANGHAI)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    instrument_id: str
    context_start: date
    window_start: date
    window_end: date
    decision_time: datetime
    label_start: date
    label_end: date
    label_available_at: datetime
    horizon: int
    forward_return: str
    label: int
    split: str
    start_close: str
    end_close: str


@dataclass(frozen=True)
class Decision:
    instrument_id: str
    decision_date: date
    candidate_split: str
    reason: str


@dataclass
class BuildResult:
    samples: list[Sample]
    decisions: list[Decision]
    coverage_missing_sessions: int


def build_samples(
    bars: Sequence[RawBar], calendar: Sequence[date], partitions: Sequence[Partition],
) -> BuildResult:
    """One exclusion reason per observed decision bar; never compress calendar gaps.

    Assumes reader-validated OHLCV. No fitting, files or random state.
    """
    if not calendar or list(calendar) != sorted(set(calendar)):
        raise ValueError("Calendar must be nonempty, sorted and unique")
    if not partitions or any(p.start > p.end for p in partitions):
        raise ValueError("Invalid partitions")
    if any(a.end >= b.start for a, b in pairwise(partitions)):
        raise ValueError("Overlapping or unordered partitions")
    if not bars:
        return BuildResult([], [], 0)
    days = [session(b) for b in bars]
    if days != sorted(set(days)) or len({b.instrument_id for b in bars}) != 1:
        raise ValueError("Bars must have one identity and unique sorted dates")
    positions = {day: i for i, day in enumerate(calendar)}
    if any(day not in positions for day in days):
        raise ValueError("Bar outside market calendar")
    first_decisions = []
    for p in partitions:
        matching = [d for d in calendar if p.start <= d <= p.end]
        if not matching:
            raise ValueError(f"No calendar sessions for {p.name}")
        first_decisions.append(eod(matching[0]))
    lookup = dict(zip(days, bars))
    result = BuildResult([], [], positions[days[-1]] - positions[days[0]] + 1 - len(days))
    for day, bar in zip(days, bars):
        part_index = next((i for i, p in enumerate(partitions)
                           if p.start <= day <= p.end), None)
        part = partitions[part_index] if part_index is not None else None
        split = part.name if part else "outside"
        i = positions[day]
        reason = "retained"
        endpoint = calendar[i + 5] if i + 5 < len(calendar) else None
        if part is None:
            reason = "outside_range"
        elif i < 38 or calendar[i - 38] < days[0]:
            reason = "history_insufficient"
        elif any(d not in lookup for d in calendar[i - 38:i + 1]):
            reason = "history_gap"
        elif any(lookup[d].volume == 0 for d in calendar[i - 19:i + 1]):
            reason = "display_zero_volume"
        elif endpoint is None or endpoint not in lookup:
            reason = "label_missing"
        elif endpoint > part.end or (
            part_index is not None and part_index + 1 < len(partitions)
            and eod(endpoint) >= first_decisions[part_index + 1]
        ):
            reason = "purged_boundary"
        result.decisions.append(Decision(bar.instrument_id, day, split, reason))
        if reason != "retained":
            continue
        assert endpoint is not None
        end_bar = lookup[endpoint]
        with localcontext() as ctx:
            ctx.prec = 34
            ctx.rounding = ROUND_HALF_EVEN
            value = str(end_bar.close / bar.close - Decimal(1))
        key = f"{VERSION}|{bar.instrument_id}|{day.isoformat()}"
        result.samples.append(Sample(
            hashlib.sha256(key.encode()).hexdigest(), bar.instrument_id,
            calendar[i - 38], calendar[i - 19], day, eod(day),
            day, endpoint, eod(endpoint), 5, value, int(end_bar.close > bar.close),
            split, str(bar.close), str(end_bar.close),
        ))
    return result


def exclusion_summary(decisions: Sequence[Decision]) -> dict[str, int]:
    return dict(sorted(Counter(
        f"{d.decision_date.year}|{d.candidate_split}|{d.reason}" for d in decisions
    ).items()))