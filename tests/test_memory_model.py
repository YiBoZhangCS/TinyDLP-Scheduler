from tinydlp.gemm import GEMMShape
from tinydlp.hardware import HardwareConfig
from tinydlp.memory_model import (
    MemoryResult,
    baseline_dram_bytes,
    estimate_baseline_memory,
    memory_cycles,
)


def test_baseline_dram_bytes() -> None:
    gemm = GEMMShape(M=4, K=8, N=16, name="toy")
    hw = HardwareConfig(name="dlp_16x16", array_m=16, array_n=16)

    assert baseline_dram_bytes(gemm, hw) == 4 * 8 + 8 * 16 + 4 * 16


def test_memory_cycles_positive() -> None:
    hw = HardwareConfig(name="dlp_16x16", array_m=16, array_n=16)

    assert memory_cycles(1, hw) > 0


def test_estimate_baseline_memory_is_positive() -> None:
    gemm = GEMMShape(M=1024, K=27, N=16, name="conv1_gemm")
    hw = HardwareConfig(name="dlp_16x16", array_m=16, array_n=16)

    result = estimate_baseline_memory(gemm, hw)

    assert isinstance(result, MemoryResult)
    assert result.dram_bytes > 0
    assert result.memory_cycles > 0
    assert "Baseline DRAM traffic" in result.description
