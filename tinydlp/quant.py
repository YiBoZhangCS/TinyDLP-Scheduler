"""量化小工具：演示 per-tensor affine INT8 量化和矩阵乘误差。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Int8MatmulDemoResult:
    """INT8 矩阵乘 demo 的数值误差汇总。"""

    scale_a: float
    zero_point_a: int
    scale_b: float
    zero_point_b: int
    mae: float
    max_error: float


def choose_scale_zero_point(
    x_min: float,
    x_max: float,
    qmin: int = -128,
    qmax: int = 127,
) -> tuple[float, int]:
    """为一个张量选择 affine 量化参数。

    Affine 量化把浮点数映射到整数：
        q = round(x / scale + zero_point)
        x ~= scale * (q - zero_point)

    scale 表示一个整数步长覆盖的浮点范围；zero_point 是尽量表示真实 0
    的整数值。
    """

    if qmin >= qmax:
        raise ValueError("qmin must be smaller than qmax")
    if x_min > x_max:
        raise ValueError("x_min must be <= x_max")

    if x_min == x_max:
        return 1.0, 0

    scale = (x_max - x_min) / float(qmax - qmin)
    zero_point_from_min = qmin - x_min / scale
    zero_point = int(round(zero_point_from_min))
    zero_point = max(qmin, min(qmax, zero_point))
    return float(scale), zero_point


def quantize(
    x: np.ndarray,
    scale: float,
    zero_point: int,
    qmin: int = -128,
    qmax: int = 127,
) -> np.ndarray:
    """使用 affine 参数把 float 张量量化为 int8。"""

    if scale <= 0:
        raise ValueError("scale must be positive")

    # zero_point 参与量化公式：q = round(x / scale + zp)。
    q = np.round(x.astype(np.float32) / scale + zero_point)
    q = np.clip(q, qmin, qmax)
    return q.astype(np.int8)


def dequantize(
    q: np.ndarray,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    """把 affine 量化后的张量反量化回 float32。"""

    if scale <= 0:
        raise ValueError("scale must be positive")

    # zero_point 参与反量化公式：x ~= scale * (q - zp)。
    return (scale * (q.astype(np.int32) - zero_point)).astype(np.float32)


def int8_matmul_demo(M: int = 16, K: int = 32, N: int = 16) -> Int8MatmulDemoResult:
    """运行一个小型 INT8 矩阵乘 demo，并返回误差指标。

    这个 demo 用来解释量化矩阵乘的数值含义，不是工业级量化方案。
    A/B 分别量化，int8 数值先减去 zero_point，再用 int32 累加乘法结果，
    最后用 scale_A * scale_B 反量化。
    """

    if M <= 0 or K <= 0 or N <= 0:
        raise ValueError("M, K, and N must be positive")

    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, size=(M, K)).astype(np.float32)
    b = rng.normal(0.0, 1.0, size=(K, N)).astype(np.float32)
    c_ref = a @ b

    scale_a, zero_point_a = choose_scale_zero_point(float(a.min()), float(a.max()))
    scale_b, zero_point_b = choose_scale_zero_point(float(b.min()), float(b.max()))
    q_a = quantize(a, scale_a, zero_point_a)
    q_b = quantize(b, scale_b, zero_point_b)

    # int8 操作数必须先提升到 int32，再做减 zero_point 和乘法。
    # 矩阵乘前先减 zero_point，使 int32 累加近似浮点乘法的缩放前结果。
    a_int = q_a.astype(np.int32) - zero_point_a
    b_int = q_b.astype(np.int32) - zero_point_b
    c_int32 = a_int @ b_int

    c_dequant = (scale_a * scale_b * c_int32).astype(np.float32)
    error = np.abs(c_ref - c_dequant)

    return Int8MatmulDemoResult(
        scale_a=scale_a,
        zero_point_a=zero_point_a,
        scale_b=scale_b,
        zero_point_b=zero_point_b,
        mae=float(error.mean()),
        max_error=float(error.max()),
    )
