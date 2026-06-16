"""基础 DRAM 访存模型：估计 GEMM 输入、权重和输出的片外搬运量。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tinydlp.gemm import GEMMShape
from tinydlp.hardware import HardwareConfig


def _ceil_div_float(numerator: int, denominator: float) -> int:
    return math.ceil(numerator / denominator)


@dataclass(frozen=True)
class MemoryResult:
    """DRAM 访存字节数和对应 memory-bound 周期的汇总。"""

    dram_bytes: int
    memory_cycles: int
    description: str


def baseline_dram_bytes(gemm: GEMMShape, hw: HardwareConfig) -> int:
    """假设 A/B/C 各搬运一次，返回基础 DRAM 字节数。"""

    # C = A x B 的基础 traffic 模型：
    # A bytes = M * K * data_bytes，因为 A 存输入激活。
    # B bytes = K * N * data_bytes，因为 B 存权重。
    # C bytes = M * N * data_bytes，因为 C 是最终输出写回。
    #
    # 注意：最终输出写回通常按 data_width_bits 估算。
    # partial sum 在片上用 acc_width_bits 累加，例如 INT8 乘法配 INT32 累加；
    # 这里的 baseline 不把片上 partial sum 计入 DRAM traffic。
    data_bytes = hw.data_bytes()
    a_bytes = gemm.M * gemm.K * data_bytes
    b_bytes = gemm.K * gemm.N * data_bytes
    c_bytes = gemm.M * gemm.N * data_bytes
    return a_bytes + b_bytes + c_bytes


def memory_cycles(dram_bytes: int, hw: HardwareConfig) -> int:
    """根据硬件 DRAM 带宽，返回搬运 dram_bytes 需要的周期数。"""

    if dram_bytes < 0:
        raise ValueError(f"dram_bytes must be non-negative, got {dram_bytes}")
    # dram_bytes_per_cycle 把片外带宽换算成 bytes/cycle。
    # cycles = ceil(total DRAM bytes / bytes per cycle)。
    return _ceil_div_float(dram_bytes, hw.dram_bytes_per_cycle())


def estimate_baseline_memory(gemm: GEMMShape, hw: HardwareConfig) -> MemoryResult:
    """估计一个 GEMM 的基础 DRAM traffic 和 memory cycles。"""

    dram_bytes = baseline_dram_bytes(gemm, hw)
    return MemoryResult(
        dram_bytes=dram_bytes,
        memory_cycles=memory_cycles(dram_bytes, hw),
        description=(
            "Baseline DRAM traffic: read A once, read B once, "
            "write final C once at data precision."
        ),
    )
