import numpy as np

from tinydlp.quant import (
    Int8MatmulDemoResult,
    choose_scale_zero_point,
    dequantize,
    int8_matmul_demo,
    quantize,
)


def test_choose_scale_zero_point_for_symmetric_range() -> None:
    scale, zero_point = choose_scale_zero_point(-1.0, 1.0)

    assert scale > 0
    assert -128 <= zero_point <= 127


def test_quantize_and_dequantize_round_trip() -> None:
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    scale, zero_point = choose_scale_zero_point(float(x.min()), float(x.max()))

    q = quantize(x, scale, zero_point)
    x_dequant = dequantize(q, scale, zero_point)

    assert q.dtype == np.int8
    assert q.min() >= -128
    assert q.max() <= 127
    assert x_dequant.dtype == np.float32
    assert np.allclose(x, x_dequant, atol=scale)


def test_int8_matmul_demo_returns_error_metrics() -> None:
    result = int8_matmul_demo(M=4, K=8, N=4)

    assert isinstance(result, Int8MatmulDemoResult)
    assert result.scale_a > 0
    assert result.scale_b > 0
    assert -128 <= result.zero_point_a <= 127
    assert -128 <= result.zero_point_b <= 127
    assert result.mae >= 0
    assert result.max_error >= result.mae
