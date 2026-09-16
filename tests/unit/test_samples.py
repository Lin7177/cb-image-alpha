from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from cb_image_alpha.data.catalog import RawBar
from cb_image_alpha.samples.index import (
    Partition,
    build_samples,
    eod,
    read_partitions,
)


def fixture():
    days = [date(2023, 9, 1) + timedelta(days=i) for i in range(150)]
    # Synthetic exchange calendar deliberately omits weekends.
    days = [d for d in days if d.weekday() < 5]
    bars = [
        RawBar("CB.SH", Decimal(100 + i), Decimal(101 + i), Decimal(99 + i),
               Decimal(100 + i), Decimal(10),
               int(datetime(d.year, d.month, d.day, 15, tzinfo=UTC).timestamp()) * 10**9, 0)
        for i, d in enumerate(days)
    ]
    parts = (Partition("train", days[0], days[59]),
             Partition("validation", days[60], days[79]),
             Partition("test", days[80], days[-1]))
    return days, bars, parts


def test_hand_calculated_first_window_and_return():
    days, bars, parts = fixture()
    result = build_samples(bars, days, parts)
    first = result.samples[0]
    assert first.context_start == days[0]
    assert first.window_start == days[19]
    assert first.window_end == days[38]
    assert first.label_end == days[43]  # market sessions, not civil days
    assert first.decision_time == eod(days[38])
    assert first.label_available_at == eod(days[43])
    assert first.forward_return == "0.036231884057971014492753623188406"
    assert first.label == 1
    assert sum(d.reason == "history_insufficient" for d in result.decisions) == 38


def test_boundaries_purge_and_left_history():
    days, bars, parts = fixture()
    result = build_samples(bars, days, parts)
    by_day = {s.window_end: s for s in result.samples}
    assert days[54] in by_day and days[55] not in by_day
    assert days[74] in by_day and days[75] not in by_day
    assert by_day[days[60]].window_start == days[41]  # crosses left boundary
    assert by_day[days[80]].context_start == days[42]
    for s in result.samples:
        p = next(p for p in parts if p.name == s.split)
        assert s.label_end <= p.end
        if s.split == "train":
            assert s.label_available_at < eod(days[60])
        elif s.split == "validation":
            assert s.label_available_at < eod(days[80])
    assert sum(d.reason == "purged_boundary" for d in result.decisions) == 10


def test_gap_is_not_compressed():
    days, bars, parts = fixture()
    result = build_samples(bars[:25] + bars[26:], days, parts)
    assert next(d for d in result.decisions if d.decision_date == days[38]).reason == "history_gap"
    assert result.coverage_missing_sessions == 1


def test_zero_volume_only_display_not_warmup_or_future():
    days, bars, parts = fixture()
    changed = [replace(b, volume=Decimal(0)) if i in (0, 43) else b
               for i, b in enumerate(bars)]
    result = build_samples(changed, days, parts)
    assert any(s.window_end == days[38] for s in result.samples)
    assert next(d for d in result.decisions if d.decision_date == days[43]).reason == (
        "display_zero_volume"
    )


def test_missing_future_intermediate_allowed_endpoint_missing_counted():
    days, bars, parts = fixture()
    result = build_samples(bars[:40] + bars[41:], days, parts)
    assert any(s.window_end == days[38] for s in result.samples)
    result = build_samples(bars[:43] + bars[44:], days, parts)
    assert next(d for d in result.decisions if d.decision_date == days[38]).reason == "label_missing"


def test_future_perturbation_keeps_historical_window_and_membership():
    days, bars, parts = fixture()
    original = next(s for s in build_samples(bars, days, parts).samples if s.window_end == days[38])
    changed = [replace(b, close=Decimal(50), open=Decimal(50),
                       high=Decimal(51), low=Decimal(49), volume=Decimal(0))
               if i > 38 else b for i, b in enumerate(bars)]
    mutated = next(s for s in build_samples(changed, days, parts).samples
                   if s.window_end == days[38])
    for field in ("sample_id", "context_start", "window_start", "window_end",
                  "decision_time", "split", "start_close"):
        assert getattr(original, field) == getattr(mutated, field)
    assert bars[:39] == changed[:39]
    assert original.label == 1 and mutated.label == 0


def test_zero_return_and_rebuild_exact():
    days, bars, parts = fixture()
    bars[43] = replace(bars[43], close=bars[38].close)
    result = build_samples(bars, days, parts)
    assert result.samples[0].label == 0
    assert Decimal(result.samples[0].forward_return) == 0
    assert result == build_samples(bars, days, parts)
    assert len({s.sample_id for s in result.samples}) == len(result.samples)
    assert len(result.decisions) == len(bars)


def test_validation_values_cannot_change_retained_train():
    days, bars, parts = fixture()
    a = build_samples(bars, days, parts)
    changed = [replace(b, close=Decimal(1)) if i >= 60 else b for i, b in enumerate(bars)]
    b = build_samples(changed, days, parts)
    assert [s for s in a.samples if s.split == "train"] == [
        s for s in b.samples if s.split == "train"
    ]


def test_reject_bad_calendar_and_identity():
    days, bars, parts = fixture()
    with pytest.raises(ValueError):
        build_samples(bars, days[::-1], parts)
    with pytest.raises(ValueError):
        build_samples([bars[0], bars[0]], days, parts)
    with pytest.raises(ValueError):
        build_samples(bars + [replace(bars[-1], instrument_id="OTHER")], days, parts)


def test_approved_config_is_exact(tmp_path):
    path = Path("configs/splits.toml")
    assert read_partitions(path)[-1].end == date(2026, 7, 31)
    changed = tmp_path / "changed.toml"
    changed.write_text(path.read_text().replace("2023-12-31", "2022-12-31"))
    with pytest.raises(ValueError):
        read_partitions(changed)

def test_return_precision_independent_of_caller_context():
    from decimal import ROUND_DOWN, localcontext
    days, bars, parts = fixture()
    expected = build_samples(bars, days, parts)
    with localcontext() as ctx:
        ctx.prec = 6
        ctx.rounding = ROUND_DOWN
        actual = build_samples(bars, days, parts)
    assert expected == actual

def test_parquet_shanghai_timestamp_roundtrip(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    instant = eod(date(2024, 1, 2))
    table = pa.table({"decision_time": pa.array([instant],
                     type=pa.timestamp("us", tz="Asia/Shanghai"))})
    path = tmp_path / "sample.parquet"
    pq.write_table(table, path)
    assert pq.read_table(path).to_pylist() == [{"decision_time": instant}]