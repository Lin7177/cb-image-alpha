import numpy as np
import pytest

from cb_image_alpha.training.engine import EarlyStop, fit_normalization, normalize


def test_normalization_training_only_and_population_std():
    train = np.zeros((2, 64, 60), dtype=np.uint8)
    train[1] = 255
    stats = fit_normalization(train)
    assert stats.mean == 0.5 and stats.std == 0.5
    validation = np.full((3, 64, 60), 255, dtype=np.uint8)
    assert normalize(validation, stats).eq(1).all()
    validation[:] = 0
    assert normalize(validation, stats).eq(-1).all()
    assert stats.mean == 0.5 and stats.std == 0.5


def test_early_stop_recovers_first_minimum_not_last_epoch():
    stop = EarlyStop(2)
    assert stop.observe(0.8, 1)
    assert stop.observe(0.4, 2)
    assert not stop.observe(0.5, 3) and not stop.stopped
    assert not stop.observe(0.4, 4) and stop.stopped
    assert stop.epoch == 2 and stop.best == 0.4


def test_improvement_resets_patience_and_nonfinite_fails():
    stop = EarlyStop(2)
    stop.observe(0.8, 1)
    stop.observe(0.9, 2)
    assert stop.observe(0.7, 3) and stop.stale == 0
    with pytest.raises(ValueError):
        stop.observe(float("nan"), 4)


def test_degenerate_normalization_fails():
    with pytest.raises(ValueError):
        fit_normalization(np.zeros((2, 64, 60), dtype=np.uint8))