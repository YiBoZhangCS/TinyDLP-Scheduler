"""Per-tensor affine quantization helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Int8MatmulDemoResult:
    """Numerical summary for the int8 matmul demo."""

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
    """Choose affine quantization parameters for one tensor.

    Affine quantization maps float values to integers with:
        q = round(x / scale + zero_point)
        x ~= scale * (q - zero_point)

    scale represents the float range covered by one integer step. zero_point is
    the integer value that represents real value 0 as closely as possible.
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
    """Quantize a float tensor to int8 using affine parameters."""

    if scale <= 0:
        raise ValueError("scale must be positive")

    # zero_point participates in quantization: q = round(x / scale + zp).
    q = np.round(x.astype(np.float32) / scale + zero_point)
    q = np.clip(q, qmin, qmax)
    return q.astype(np.int8)


def dequantize(
    q: np.ndarray,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    """Dequantize an affine-quantized tensor back to float32."""

    if scale <= 0:
        raise ValueError("scale must be positive")

    # zero_point participates in dequantization: x ~= scale * (q - zp).
    return (scale * (q.astype(np.int32) - zero_point)).astype(np.float32)


def int8_matmul_demo(M: int = 16, K: int = 32, N: int = 16) -> Int8MatmulDemoResult:
    """Run a small int8 matmul demo and return error metrics.

    This demo explains the numerical meaning of quantized matmul. It is not an
    industrial quantization recipe. A and B are quantized independently, int8
    values are shifted by their zero points, multiplication is accumulated in
    int32, and the result is dequantized with scale_A * scale_B.
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

    # int8 operands must be promoted to int32 before subtraction/multiplication.
    # The zero points are subtracted before matmul so that int32 accumulation
    # approximates sum((A / scale_A) * (B / scale_B)).
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
