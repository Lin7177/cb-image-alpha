import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from cb_image_alpha.artifacts import read_json
from cb_image_alpha.evaluation.metrics import binary_metrics, compare_baselines


def test_hand_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    result = binary_metrics(y, p)
    assert result['auc'] == 0.75
    assert result['accuracy'] == 0.75
    assert result['balanced_accuracy'] == 0.75
    assert result['log_loss'] == pytest.approx(-np.log([0.9, 0.6, 0.35, 0.8]).mean())
    ties = binary_metrics(y, np.full(4, 0.5))
    assert ties['auc'] == 0.5
    assert ties['accuracy'] == 0.5
    one = binary_metrics(np.ones(2, dtype=np.int64), np.array([0.0, 1.0]))
    assert one['auc'] is None and one['balanced_accuracy'] is None
    assert np.isfinite(one['log_loss'])
    baseline = compare_baselines(np.ones(4, dtype=np.int64), p, 0.25)
    assert baseline['train_majority']['accuracy'] == 0
    assert baseline['train_constant_probability']['log_loss'] == pytest.approx(-np.log(0.25))


@pytest.mark.parametrize('p', [[np.nan], [-0.1], [1.1]])
def test_invalid_probability(p):
    with pytest.raises(ValueError):
        binary_metrics(np.array([0]), np.array(p))


@pytest.mark.parametrize('limit,expected', [('time', 'wall_time_limit'),
                                         ('ram', 'ram_limit'), ('disk', 'artifact_limit')])
def test_supervisor_limits(tmp_path, limit, expected):
    path = Path(__file__).resolve().parents[1] / 'scripts/run_formal.py'
    spec = importlib.util.spec_from_file_location('formal_supervisor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {'ram_limit_bytes': 2**30, 'artifact_limit_bytes': 2**30}
    if limit == 'ram':
        config['ram_limit_bytes'] = 1
    if limit == 'disk':
        (tmp_path / 'payload').write_bytes(b'1234')
        config['artifact_limit_bytes'] = 1
    with pytest.raises(RuntimeError):
        module.supervise(tmp_path, 'fixture', 1 if limit == 'time' else 10,
                         config, [sys.executable, '-c', 'import time; time.sleep(30)'])
    report = read_json(tmp_path / 'fixture-supervision.json')
    assert report['failure'] == expected
    assert report['exit_code'] != 0
