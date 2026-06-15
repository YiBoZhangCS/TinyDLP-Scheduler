import pytest

from tinydlp.compute_model import estimate_compute
from tinydlp.gemm import GEMMShape
from tinydlp.hardware import HardwareConfig
from tinydlp.scheduler import (
    ConvScheduleResult,
    ScheduleResult,
    evaluate_schedule,
    pretty_print_schedule,
    search_best_conv_schedule,
    search_best_schedule,
    search_topk_schedules,
)
from tinydlp.layer import Conv2DLayer
from tinydlp.tile import GEMMTile


def test_evaluate_schedule_combines_compute_and_memory() -> None:
    gemm = GEMMShape(M=64, K=64, N=128, name="toy")
    hw = HardwareConfig(name="dlp_16x16", array_m=16, array_n=16)
    tile = GEMMTile(tile_m=16, tile_k=64, tile_n=32)

    result = evaluate_schedule(
        gemm=gemm,
        hw=hw,
        tile=tile,
        dataflow_name="weight_stationary",
    )
    compute = estimate_compute(gemm, hw)

    assert isinstance(result, ScheduleResult)
    assert result.layer_name == "toy"
    assert result.gemm == gemm
    assert result.tile == tile
    assert result.dataflow == "weight_stationary"
    assert result.macs == gemm.macs()
    assert result.ideal_compute_cycles == compute.ideal_cycles
    assert result.array_compute_cycles == compute.array_aware_cycles
    assert result.systolic_compute_cycles == compute.systolic_cycles
    assert result.pe_utilization == compute.pe_utilization_systolic
    assert result.dram_bytes > 0
    assert result.memory_cycles > 0
    assert result.no_overlap_cycles == (
        result.systolic_compute_cycles + result.memory_cycles
    )
    assert result.ideal_overlap_cycles == max(
        result.systolic_compute_cycles,
        result.memory_cycles,
    )


def test_evaluate_schedule_reports_bottleneck() -> None:
    gemm = GEMMShape(M=16, K=16, N=16, name="toy")
    hw = HardwareConfig(
        name="low_bandwidth",
        array_m=16,
        array_n=16,
        dram_bandwidth_gb_s=0.01,
    )
    tile = GEMMTile(tile_m=16, tile_k=16, tile_n=16)

    result = evaluate_schedule(gemm, hw, tile, "output_stationary")

    assert result.bottleneck in {"compute-bound", "memory-bound"}
    assert result.bottleneck == "memory-bound"


def test_search_best_schedule_uses_ideal_overlap_by_default() -> None:
    gemm = GEMMShape(M=64, K=64, N=128, name="toy")
    hw = HardwareConfig(name="dlp_16x16", array_m=16, array_n=16, sram_kb=64)

    best = search_best_schedule(gemm, hw)
    top = search_topk_schedules(gemm, hw, k=5)

    assert best == top[0]
    assert all(
        top[index].ideal_overlap_cycles <= top[index + 1].ideal_overlap_cycles
        for index in range(len(top) - 1)
    )


def test_search_topk_schedules_can_use_no_overlap_objective() -> None:
    gemm = GEMMShape(M=64, K=64, N=128, name="toy")
    hw = HardwareConfig(name="dlp_16x16", array_m=16, array_n=16, sram_kb=64)

    top = search_topk_schedules(gemm, hw, k=5, overlap_mode="none")

    assert len(top) == 5
    assert all(
        top[index].no_overlap_cycles <= top[index + 1].no_overlap_cycles
        for index in range(len(top) - 1)
    )


def test_pretty_print_schedule_contains_key_fields(capsys) -> None:
    gemm = GEMMShape(M=16, K=16, N=16, name="toy")
    hw = HardwareConfig(name="dlp_16x16", array_m=16, array_n=16)
    tile = GEMMTile(tile_m=16, tile_k=16, tile_n=16)
    result = evaluate_schedule(gemm, hw, tile, "output_stationary")

    text = pretty_print_schedule(result)
    printed = capsys.readouterr().out

    assert "GEMM shape" in text
    assert "tile" in text
    assert "dataflow" in text
    assert "no-overlap cycles" in text
    assert "ideal-overlap cycles" in text
    assert text in printed


def test_search_topk_schedules_raises_when_no_tile_fits() -> None:
    gemm = GEMMShape(M=1, K=1, N=1, name="too_large_for_sram")
    hw = HardwareConfig(
        name="tiny_sram",
        array_m=1,
        array_n=1,
        sram_kb=1,
        data_width_bits=8192,
        acc_width_bits=8192,
    )

    with pytest.raises(ValueError, match="SRAM may be too small"):
        search_topk_schedules(gemm, hw)


def test_search_best_conv_schedule_returns_breakdowns() -> None:
    layer = Conv2DLayer(
        name="toy_conv",
        batch=1,
        in_channels=16,
        in_h=8,
        in_w=8,
        out_channels=32,
        kernel_h=3,
        kernel_w=3,
        stride=1,
        padding=1,
    )
    hw = HardwareConfig(name="dlp_16x16", array_m=16, array_n=16, sram_kb=16)

    result = search_best_conv_schedule(layer, hw)

    assert isinstance(result, ConvScheduleResult)
    assert result.conv_tile.tb == 1
    assert result.gemm_tile.tile_m == (
        result.conv_tile.tb * result.conv_tile.tp * result.conv_tile.tq
    )
    assert result.gemm_tile.tile_n == result.conv_tile.tm
    assert result.gemm_tile.tile_k == result.conv_tile.tc * 3 * 3
    assert result.sram_usage.fits_sram
    assert result.dram_traffic.total_dram_bytes > 0
    assert result.total_cycles == result.ideal_overlap_cycles
