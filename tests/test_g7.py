import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('research_g7', Path(__file__).resolve().parents[1]/'scripts/research_g7.py')
g7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g7)


def test_bins_edges_and_empty():
    rows = g7.bins(np.array([0, 1, 0, 1]), np.array([0., .1, .9, 1.]))
    assert rows[0]['count'] == 1
    assert rows[1]['count'] == 1
    assert rows[9]['count'] == 2
    assert rows[2]['mean_probability'] is None
    assert rows[9]['positive_rate'] == .5


def test_quantization_hand_and_invariance():
    prices = np.tile([2., 3., 1., 2.], (39, 1))
    volumes = np.ones(20)
    volumes[-1] = 24
    u, q, v, h = g7.quantize(prices, volumes)
    np.testing.assert_array_equal(q[0], [25, 50, 0, 25, 25])
    assert h[0] == 0 and h[-1] == 12  # rint(0.5), then visible baseline is one pixel.
    np.testing.assert_allclose(u, g7.quantize(prices*2, volumes*3)[0], atol=1e-12)
    np.testing.assert_allclose(v, g7.quantize(prices*2, volumes*3)[2], atol=1e-12)


def test_chart_empty_bins(tmp_path):
    path=tmp_path/'empty.png'
    g7.plot_lines(path,'fixture',['a','b'],{'empty':[float('nan'), .5]},0,1)
    assert path.stat().st_size > 0


def test_mixed_windows_fingerprint_paths():
    import pytest
    assert g7.normalize_fingerprints({'train/0000.npy':'abc', r'train\0000.npy':'abc'}) == {'train/0000.npy':'abc'}
    with pytest.raises(ValueError, match='Conflicting'):
        g7.normalize_fingerprints({'train/0000.npy':'abc', r'train\0000.npy':'other'})
