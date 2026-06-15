"""Baseline DRAM traffic estimates for GEMM."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tinydlp.gemm import GEMMShape
from tinydlp.hardware import HardwareConfig


def _ceil_div_float(numerator: int, denominator: float) -> int:
    return math.ceil(numerator / denominator)


@dataclass(frozen=True)
class MemoryResult:
    """Summary of estimated DRAM traffic and memory-bound cycles."""

    dram_bytes: int
    memory_cycles: int
    description: str


def baseline_dram_bytes(gemm: GEMMShape, hw: HardwareConfig) -> int:
    """Return baseline DRAM bytes assuming A, B, and C each move once."""

    # Baseline traffic model for C = A x B:
    # A bytes = M * K * data_bytes, because A stores input activations.
    # B bytes = K * N * data_bytes, because B stores weights.
    # C bytes = M * N * data_bytes, because this is the final output writeback.
    #
    # Note that final output writeback usually uses data_width_bits. Partial
    # sums are accumulated on chip with acc_width_bits, such as INT8 multiply
    # with INT32 accumulation, and are not counted as baseline DRAM traffic here.
    data_bytes = hw.data_bytes()
    a_bytes = gemm.M * gemm.K * data_bytes
    b_bytes = gemm.K * gemm.N * data_bytes
    c_bytes = gemm.M * gemm.N * data_bytes
    return a_bytes + b_bytes + c_bytes


def memory_cycles(dram_bytes: int, hw: HardwareConfig) -> int:
    """Return cycles needed to move dram_bytes at configured DRAM bandwidth."""

    if dram_bytes < 0:
        raise ValueError(f"dram_bytes must be non-negative, got {dram_bytes}")
    # dram_bytes_per_cycle converts off-chip bandwidth into bytes/cycle.
    # cycles = ceil(total DRAM bytes / bytes per cycle).
    return _ceil_div_float(dram_bytes, hw.dram_bytes_per_cycle())


def estimate_baseline_memory(gemm: GEMMShape, hw: HardwareConfig) -> MemoryResult:
    """Estimate baseline DRAM traffic and memory cycles for one GEMM."""

    dram_bytes = baseline_dram_bytes(gemm, hw)
    return MemoryResult(
        dram_bytes=dram_bytes,
        memory_cycles=memory_cycles(dram_bytes, hw),
        description=(
            "Baseline DRAM traffic: read A once, read B once, "
            "write final C once at data precision."
        ),
    )
