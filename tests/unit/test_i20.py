from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from cb_image_alpha.data.catalog import RawBar
from cb_image_alpha.imaging.renderer import render_i20
from cb_image_alpha.models.paper_cnn import PaperCNN


def context():
    start = datetime(2023, 1, 1, 15, tzinfo=UTC)
    return [RawBar("X", Decimal(96), Decimal(192), Decimal(64), Decimal(128),
                   Decimal(24), int((start + timedelta(days=i)).timestamp()) * 10**9, 0)
            for i in range(39)]


def test_entire_hand_drawn_pixel_fixture():
    bars = context()
    bars[19] = replace(bars[19], volume=Decimal(1))  # half pixel -> even zero -> baseline
    bars[20] = replace(bars[20], volume=Decimal(3))  # 1.5 -> two pixels
    result = render_i20(bars)
    expected = np.zeros((64, 60), dtype=np.uint8)
    expected[:51, 1::3] = 255  # high-low columns: min64 -> row50, max192 -> row0
    expected[38, ::3] = 255   # open96: ties-to-even 12.5 ->12, row38
    expected[25, 1:] = 255   # MA128 joins centers and close right ticks
    expected[52:, 7::3] = 255
    expected[63, 1] = 255
    expected[62:, 4] = 255
    np.testing.assert_array_equal(result.pixels, expected)
    assert result.ma20 == (128.0,) * 20
    assert np.all(result.pixels[51] == 0)


def test_ma_warmup_changes_bounds_and_connects_centers():
    bars = [replace(b, open=Decimal(10), high=Decimal(10), low=Decimal(10),
                    close=Decimal(10)) if i < 19 else
            replace(b, open=Decimal(100), high=Decimal(110), low=Decimal(100),
                    close=Decimal(100)) for i, b in enumerate(context())]
    r = render_i20(bars)
    assert r.ma20[0] == 14.5 and r.ma20[1] == 19 and r.ma20[-1] == 100
    assert r.price_min == 0.145 and r.price_max == 1.1
    assert r.pixels[50, 1] == 255
    assert r.pixels[49, 2] == 255 and r.pixels[49, 3] == 255
    assert r.pixels[48, 4] == 255


def test_reference_zero_height_pillow_clipping_fixture():
    # Primitive API test, not execution of reference code.
    image = Image.new("L", (3, 12))
    ImageDraw.Draw(image).line((1, 0, 1, -1), fill=255)
    assert np.asarray(image)[:, 1].tolist() == [255] + [0] * 11


def test_reject_degenerate_and_invalid_windows():
    with pytest.raises(ValueError):
        render_i20(context()[:-1])
    bars = [replace(b, open=Decimal(100), close=Decimal(100),
                    low=Decimal(100), high=Decimal(100)) for b in context()]
    with pytest.raises(ValueError, match="Degenerate"):
        render_i20(bars)
    bars = context()
    bars[-1] = replace(bars[-1], volume=Decimal(0))
    with pytest.raises(ValueError):
        render_i20(bars)


def test_positive_scale_invariance():
    bars = context()
    scaled = [replace(b, open=b.open * 2, close=b.close * 2, high=b.high * 2,
                      low=b.low * 2, volume=b.volume * 3) for b in bars]
    np.testing.assert_array_equal(render_i20(bars).pixels, render_i20(scaled).pixels)


def test_cnn_structure_gradient_and_roundtrip():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    model = PaperCNN()
    assert sum(p.numel() for p in model.parameters()) == 708866
    for block in model.blocks:
        bn = block[1]
        assert torch.equal(bn.running_mean, torch.zeros_like(bn.running_mean))
        assert torch.equal(bn.running_var, torch.ones_like(bn.running_var))
        assert bn.num_batches_tracked.item() == 0
        assert torch.equal(bn.weight, torch.ones_like(bn.weight))
        assert torch.equal(bn.bias, torch.zeros_like(bn.bias))
    x = torch.randn(2, 1, 64, 60)
    shapes = []
    handles = [b.register_forward_hook(lambda m, inp, out: shapes.append(tuple(out.shape)))
               for b in model.blocks]
    conv_shapes = []
    handles += [b[0].register_forward_hook(
        lambda m, inp, out: conv_shapes.append(tuple(out.shape))) for b in model.blocks]
    logits = model(x)
    assert conv_shapes == [(2, 64, 20, 60), (2, 128, 10, 60), (2, 256, 5, 60)]
    assert shapes == [(2, 64, 10, 60), (2, 128, 5, 60), (2, 256, 3, 60)]
    for h in handles:
        h.remove()
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    expected = model(x).detach()
    stream = BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    clone = PaperCNN().eval()
    clone.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(expected, clone(x), rtol=0, atol=0)
    assert model.head.bias.tolist() == pytest.approx([0.01, 0.01])
    for i, block in enumerate(model.blocks):
        conv, bn, activation, pool = block
        assert conv.kernel_size == (5, 3) and conv.padding == (2, 1)
        assert conv.stride == ((3, 1) if i == 0 else (1, 1))
        assert conv.dilation == ((2, 1) if i == 0 else (1, 1))
        assert bn.eps == 1e-5 and bn.momentum == 0.1
        assert activation.negative_slope == 0.01
        assert pool.ceil_mode
    assert model.dropout.p == 0.5


def test_cnn_reject_wrong_shape():
    with pytest.raises(ValueError):
        PaperCNN()(torch.zeros(1, 1, 60, 64))