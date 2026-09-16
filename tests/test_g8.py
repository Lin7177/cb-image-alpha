import sys
import tomllib
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g8_data import fold_roles, numerical_features
from g8_evaluate import draw_blocks
from g8_models import logistic_fit, logistic_predict


def config():
    return tomllib.loads(
        (Path(__file__).resolve().parents[1] / "configs/research/g8.toml").read_text(
            encoding="utf-8-sig"
        )
    )


def test_fold_right_purge_and_left_context():
    days = [
        date(2020, 12, 24),
        date(2020, 12, 30),
        date(2021, 6, 1),
        date(2021, 12, 30),
        date(2022, 6, 1),
        date(2022, 12, 30),
    ]
    ends = [
        date(2020, 12, 31),
        date(2021, 1, 7),
        date(2021, 6, 8),
        date(2022, 1, 6),
        date(2022, 6, 8),
        date(2023, 1, 6),
    ]
    roles = fold_roles(days, ends, 2022)
    assert roles["inner_train"].tolist() == [0]
    assert roles["inner_validation"].tolist() == [2]
    assert roles["final_train"].tolist() == [
        0,
        1,
        2,
    ]  # Earlier boundary is reopened within final prefix.
    assert roles["outer"].tolist() == [4]


def test_features_hand_and_no_scale():
    prices = np.tile([2.0, 3.0, 1.0, 2.0], (39, 1))
    vol = np.ones(20)
    vol[-1] = 24
    u, q = numerical_features(prices, vol)
    np.testing.assert_array_equal(u[:5], [0.5, 1, 0, 0.5, 0.5])
    assert u[5] == 1 / 24 and q[5] == 1 / 12
    np.testing.assert_allclose(u, numerical_features(prices * 3, vol * 5)[0])


def test_logistic_intercept_oracle_and_zero_columns():
    torch.set_num_threads(2)
    x = np.ones((40, 3))
    y = np.array([0] * 10 + [1] * 30)
    model, report = logistic_fit(x, y, config())
    assert report["converged"]
    assert report["evaluations"] <= 400
    np.testing.assert_allclose(model["weight"], 0, atol=1e-12)
    np.testing.assert_allclose(logistic_predict(model, x), 0.75, atol=2e-6)


def test_logistic_synthetic_gradient_and_fit():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 4))
    y = (x[:, 0] > 0.3).astype(int)
    model, report = logistic_fit(x, y, config())
    assert report["converged"] and report["gradient_inf"] <= 1e-6
    p = logistic_predict(model, x)
    assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
    broken = config()
    broken["logistic_evaluations"] = 1
    with pytest.raises(RuntimeError, match="evaluation budget"):
        logistic_fit(x, y, broken)


def test_blocks_repeat_and_truncate():
    draw = draw_blocks(np.random.default_rng(5), 23, 20)
    assert len(draw) == 23 and draw.min() >= 0 and draw.max() < 23
    assert len(set(draw)) < 23
    loss = np.arange(23, dtype=float)
    count = np.arange(23) + 1
    rows = np.repeat(loss / count, count)
    offsets = np.r_[0, np.cumsum(count)]
    selected = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in draw])
    assert rows[selected].mean() == pytest.approx(loss[draw].sum() / count[draw].sum())


def test_image_mapping_noncontiguous_across_shards(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from g8_data import FoldImages

    from cb_image_alpha.artifacts import write_json

    original = tmp_path / "original"
    for split, shard, value in (("train", 0, 0), ("train", 1, 255), ("validation", 0, 255)):
        folder = original / split
        folder.mkdir(parents=True, exist_ok=True)
        pixels = np.full((2, 64, 60), value, dtype=np.uint8)
        pixels[1] = 255 - value
        np.save(folder / f"{shard:04d}.npy", pixels)
    write_json(tmp_path / "plan.json", {"original": str(original)})
    pq.write_table(
        pa.table(
            {
                "split": ["train", "validation", "train"],
                "shard_id": [1, 0, 0],
                "shard_row": [1, 0, 1],
                "label": [0, 1, 0],
            }
        ),
        tmp_path / "index.parquet",
    )
    np.save(tmp_path / "white-counts.npy", np.array([0, 3840, 3840]))
    view = FoldImages(tmp_path, np.array([2, 0]))
    assert view.normalization["mean"] == 0.5
    x, y = view.batch(np.array([1, 0]))
    assert torch.all(x[0] == -1) and torch.all(x[1] == 1)
    assert y.tolist() == [0, 0]
    validation = FoldImages(tmp_path, np.array([1]), view.normalization)
    assert torch.all(validation.batch(np.array([0]))[0] == 1)


def test_sparse_effective_dates_no_interval(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from g8_evaluate import METHODS, evaluate

    from cb_image_alpha.artifacts import read_json, write_json

    original = tmp_path / "original"
    original.mkdir()
    calendar = tmp_path / "download/base/calendar.csv"
    calendar.parent.mkdir(parents=True)
    calendar.write_text("trading_date\n" + "\n".join(f"2022-01-{i:02d}" for i in range(1, 26)))
    write_json(original / "frozen.json", {"source": str(tmp_path / "catalog")})
    cfg = config()
    cfg["fold_years"] = [2022]
    write_json(tmp_path / "plan.json", {"original": str(original), "config": cfg})
    pq.write_table(
        pa.table(
            {
                "sample_id": ["a", "b"],
                "label": [0, 1],
                "window_end": [date(2022, 1, 2), date(2022, 1, 3)],
            }
        ),
        tmp_path / "index.parquet",
    )
    fold = tmp_path / "2022"
    fold.mkdir()
    np.save(fold / "outer.npy", np.array([0, 1]))
    for method in METHODS:
        np.save(fold / f"{method}-p.npy", np.array([0.4, 0.6]))
    write_json(fold / "constant.json", {"train_positive_rate": 0.5})
    evaluate(tmp_path)
    assert (
        read_json(tmp_path / "reconciliation.json")["2022"]["interval_undefined_reason"]
        == "fewer_than_20_effective_dates"
    )
    assert read_json(tmp_path / "results.json")["fold_intervals"]["2022"]["U-Q"]["row"] is None


def test_failed_logistic_preserves_evaluation_count(tmp_path):
    from cb_image_alpha.artifacts import read_json

    cfg = config()
    cfg["logistic_evaluations"] = 1
    report = tmp_path / "report.json"
    with pytest.raises(RuntimeError, match="evaluation budget"):
        logistic_fit(np.arange(40).reshape(20, 2), np.arange(20) % 2, cfg, report_path=report)
    assert read_json(report)["evaluations"] == 1
    assert not read_json(report)["converged"]
