"""测试 compute_model：验证理想周期、模数效应和 fill/drain 周期。"""

from tinydlp.compute_model import (
    ComputeResult,
    array_aware_gemm_cycles,
    estimate_compute,
    ideal_compute_cycles,
    systolic_gemm_cycles,
)
from tinydlp.gemm import GEMMShape
from tinydlp.hardware import HardwareConfig


def test_compute_model_module_imports() -> None:
    assert ComputeResult is not None


def test_gemm_compute_cycles_for_16x16_array() -> None:
    gemm = GEMMShape(M=1024, K=27, N=16, name="conv1_gemm")
    hw = HardwareConfig(name="dlp_16x16", array_m=16, array_n=16)

    assert ideal_compute_cycles(gemm, hw) == (1024 * 27 * 16 + 255) // 256
    assert array_aware_gemm_cycles(gemm, hw) == 64 * 1 * 27
    assert systolic_gemm_cycles(gemm, hw) == 64 * 1 * (27 + 16 + 16 - 2)


def test_estimate_compute_utilization_drops_with_systolic_overhead() -> None:
    gemm = GEMMShape(M=1024, K=27, N=16, name="conv1_gemm")
    hw = HardwareConfig(name="dlp_16x16", array_m=16, array_n=16)

    result = estimate_compute(gemm, hw)

    assert result.macs == gemm.macs()
    assert result.ideal_cycles == 1_728
    assert result.array_aware_cycles == 1_728
    assert result.systolic_cycles == 3_648
    assert result.pe_utilization_array_aware == 1.0
    assert result.pe_utilization_systolic < result.pe_utilization_array_aware
