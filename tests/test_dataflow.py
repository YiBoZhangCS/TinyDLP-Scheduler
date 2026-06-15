from tinydlp.dataflow import (
    DataflowResult,
    compare_dataflows,
    input_stationary,
    output_stationary_conv,
    output_stationary,
    weight_stationary,
)
from tinydlp.gemm import GEMMShape
from tinydlp.hardware import HardwareConfig
from tinydlp.layer import Conv2DLayer
from tinydlp.tile import ConvTile, GEMMTile


def test_compare_dataflows_returns_three_positive_results() -> None:
    gemm = GEMMShape(M=64, K=64, N=128, name="toy")
    tile = GEMMTile(tile_m=16, tile_k=64, tile_n=32)
    hw = HardwareConfig(name="dlp_16x16", array_m=16, array_n=16)

    results = compare_dataflows(gemm, tile, hw)

    assert [result.dataflow for result in results] == [
        "output_stationary",
        "weight_stationary",
        "input_stationary",
    ]
    assert all(isinstance(result, DataflowResult) for result in results)
    assert all(result.dram_bytes > 0 for result in results)
    assert all(result.memory_cycles > 0 for result in results)


def test_dataflow_traffic_reflects_reuse_choice() -> None:
    gemm = GEMMShape(M=64, K=64, N=128, name="toy")
    tile = GEMMTile(tile_m=16, tile_k=64, tile_n=32)
    hw = HardwareConfig(name="dlp_16x16", array_m=16, array_n=16)

    output = output_stationary(gemm, tile, hw)
    weight = weight_stationary(gemm, tile, hw)
    input_ = input_stationary(gemm, tile, hw)

    assert weight.dram_bytes < output.dram_bytes
    assert input_.dram_bytes != weight.dram_bytes
    assert "C partial sums" in output.explanation
    assert "B/weight" in weight.explanation
    assert "A/input" in input_.explanation


def test_conv_dataflow_models_partial_sum_spill() -> None:
    layer = Conv2DLayer(
        name="toy_conv",
        batch=1,
        in_channels=64,
        in_h=8,
        in_w=8,
        out_channels=16,
        kernel_h=3,
        kernel_w=3,
        stride=1,
        padding=1,
    )
    tile = ConvTile(tb=1, tm=16, tc=16, tp=4, tq=4)
    hw = HardwareConfig(name="dlp_16x16", array_m=16, array_n=16)

    in_sram = output_stationary_conv(layer, tile, hw, psum_in_sram=True)
    spilled = output_stationary_conv(layer, tile, hw, psum_in_sram=False)

    assert in_sram.num_c_tiles == 4
    assert in_sram.psum_read_write_bytes == 0
    assert spilled.psum_read_write_bytes == (
        2 * (4 - 1) * in_sram.output_write_bytes
    )
    assert spilled.total_dram_bytes > in_sram.total_dram_bytes
