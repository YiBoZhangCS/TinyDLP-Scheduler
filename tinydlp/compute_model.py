"""简化 DLP 阵列上的 GEMM 计算周期模型。"""

from __future__ import annotations

from dataclasses import dataclass

from tinydlp.gemm import GEMMShape
from tinydlp.hardware import HardwareConfig


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class ComputeResult:
    """一次 GEMM 的理想、阵列感知和脉动阵列计算估计结果。"""

    macs: int
    ideal_cycles: int
    array_aware_cycles: int
    systolic_cycles: int
    pe_utilization_ideal: float
    pe_utilization_array_aware: float
    pe_utilization_systolic: float


@dataclass(frozen=True)
class TileComputeResult:
    """一个 GEMM/Conv tile 在脉动阵列上的计算估计。"""

    useful_macs: int
    ideal_cycles: int
    array_aware_cycles: int
    systolic_cycles: int
    pe_utilization: float
    provided_mac_slots: int
    m_blocks: int
    n_blocks: int


def _pe_utilization(true_macs: int, cycles: int, peak_macs_per_cycle: int) -> float:
    # PE 利用率 = 真实有用 MAC 数 / 硬件提供的 MAC 槽位数。
    # 硬件槽位数 = cycles * peak_macs_per_cycle。
    # 这里限制最大为 1.0，避免乐观模型因为取整误差报告超过 100% 的占用率。
    util = true_macs / (cycles * peak_macs_per_cycle)
    return min(util, 1.0)


def estimate_tile_compute(
    M_tile: int,
    N_tile: int,
    K_tile: int,
    array_m: int,
    array_n: int,
    mac_per_pe_per_cycle: int = 1,
) -> TileComputeResult:
    """估计一个输出 tile 在脉动阵列上的计算周期。

    这个函数显式体现两个关键影响：

    - 模数效应：M_tile/N_tile 的尾块仍然占用完整阵列 block；
    - fill/drain 开销：每个阵列 block 额外付出 array_m + array_n - 2 周期。
    """

    for name, value in (
        ("M_tile", M_tile),
        ("N_tile", N_tile),
        ("K_tile", K_tile),
        ("array_m", array_m),
        ("array_n", array_n),
        ("mac_per_pe_per_cycle", mac_per_pe_per_cycle),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    m_blocks = _ceil_div(M_tile, array_m)
    n_blocks = _ceil_div(N_tile, array_n)
    useful_macs = M_tile * N_tile * K_tile
    peak_macs = array_m * array_n * mac_per_pe_per_cycle

    ideal_cycles = _ceil_div(useful_macs, peak_macs)
    array_aware_cycles = m_blocks * n_blocks * K_tile

    # 对一个脉动阵列 block，K_tile 个周期负责 reduction。
    # 额外的 array_m + array_n - 2 周期近似表示：
    # 输入操作数从阵列边缘流入并填满二维 PE 波前，以及最后结果排出的时间。
    systolic_cycles = m_blocks * n_blocks * (K_tile + array_m + array_n - 2)
    provided_mac_slots = systolic_cycles * peak_macs
    pe_utilization = useful_macs / provided_mac_slots

    return TileComputeResult(
        useful_macs=useful_macs,
        ideal_cycles=ideal_cycles,
        array_aware_cycles=array_aware_cycles,
        systolic_cycles=systolic_cycles,
        pe_utilization=pe_utilization,
        provided_mac_slots=provided_mac_slots,
        m_blocks=m_blocks,
        n_blocks=n_blocks,
    )


def ideal_compute_cycles(gemm: GEMMShape, hw: HardwareConfig) -> int:
    """返回假设 PE 完美利用时的理想计算周期。"""

    # 理想模型忽略阵列 tile 边界和调度开销：
    # cycles = ceil(总 MAC 数 / 每周期峰值 MAC 数)。
    return _ceil_div(gemm.macs(), hw.peak_macs_per_cycle())


def array_aware_gemm_cycles(gemm: GEMMShape, hw: HardwareConfig) -> int:
    """返回考虑 M/N 阵列切块、但不考虑 fill/drain 的 GEMM 周期。"""

    # 输出矩阵 C 会沿 M 和 N 方向映射到 PE 阵列上。
    # m_tiles/n_tiles 使用向上取整，因为即使 M 或 N 不能整除阵列尺寸，
    # 尾块也仍然会占用一个完整的 array_m x array_n 阵列 tile。
    #
    # 这就是“模数效应”：真实 MAC 数没有变化，但尾块会产生空 PE 槽位。
    # 这些空槽位让实际执行周期大于只看 MAC 总数的理想周期估计。
    #
    # K 是 reduction 维度。在这个简化的 array-aware 模型中，
    # 每个输出 tile 需要 K 个周期，暂不加入脉动阵列 fill/drain 开销。
    result = estimate_tile_compute(
        M_tile=gemm.M,
        N_tile=gemm.N,
        K_tile=gemm.K,
        array_m=hw.array_m,
        array_n=hw.array_n,
        mac_per_pe_per_cycle=hw.mac_per_pe_per_cycle,
    )
    return result.array_aware_cycles


def systolic_gemm_cycles(gemm: GEMMShape, hw: HardwareConfig) -> int:
    """返回加入简化脉动阵列 fill/drain 开销后的 GEMM 周期。"""

    # 一个脉动阵列 tile 对应输出矩阵 C 的 array_m x array_n 子块。
    # K 是 reduction 维度，会沿时间方向逐拍推进和累加。
    # 简化的 fill/drain 开销是 array_m + array_n - 2 个周期：
    # 数据需要从二维阵列边缘流入、填满阵列，再把最终结果排出。
    result = estimate_tile_compute(
        M_tile=gemm.M,
        N_tile=gemm.N,
        K_tile=gemm.K,
        array_m=hw.array_m,
        array_n=hw.array_n,
        mac_per_pe_per_cycle=hw.mac_per_pe_per_cycle,
    )
    return result.systolic_cycles


def estimate_compute(gemm: GEMMShape, hw: HardwareConfig) -> ComputeResult:
    """估计一个 GEMM 的计算周期和 PE 利用率。"""

    macs = gemm.macs()
    peak = hw.peak_macs_per_cycle()
    ideal_cycles = ideal_compute_cycles(gemm, hw)
    array_aware_cycles = array_aware_gemm_cycles(gemm, hw)
    systolic_cycles = systolic_gemm_cycles(gemm, hw)

    return ComputeResult(
        macs=macs,
        ideal_cycles=ideal_cycles,
        array_aware_cycles=array_aware_cycles,
        systolic_cycles=systolic_cycles,
        pe_utilization_ideal=_pe_utilization(macs, ideal_cycles, peak),
        pe_utilization_array_aware=_pe_utilization(
            macs, array_aware_cycles, peak
        ),
        pe_utilization_systolic=_pe_utilization(macs, systolic_cycles, peak),
    )
