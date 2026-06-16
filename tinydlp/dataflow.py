"""简化 dataflow 访存模型。

这些函数是教学/分析模型，不是精确硬件仿真。
它们只估计不同复用策略会如何改变 DRAM traffic。
"""

from __future__ import annotations

from dataclasses import dataclass

from tinydlp.gemm import GEMMShape
from tinydlp.hardware import HardwareConfig
from tinydlp.layer import Conv2DLayer
from tinydlp.memory_model import memory_cycles
from tinydlp.tile import ConvTile, GEMMTile


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class DataflowResult:
    """某一种简化 dataflow 的 DRAM traffic 估计结果。"""

    dataflow: str
    dram_bytes: int
    memory_cycles: int
    explanation: str
    input_read_bytes: int = 0
    weight_read_bytes: int = 0
    psum_read_write_bytes: int = 0
    output_write_bytes: int = 0
    total_dram_bytes: int = 0
    psum_in_sram: bool = True
    num_c_tiles: int = 1

    def __post_init__(self) -> None:
        if self.total_dram_bytes == 0:
            object.__setattr__(self, "total_dram_bytes", self.dram_bytes)


def _tile_counts(gemm: GEMMShape, tile: GEMMTile) -> tuple[int, int, int]:
    m_tiles = _ceil_div(gemm.M, tile.tile_m)
    k_tiles = _ceil_div(gemm.K, tile.tile_k)
    n_tiles = _ceil_div(gemm.N, tile.tile_n)
    return m_tiles, k_tiles, n_tiles


def _partial_sum_spill_bytes(gemm: GEMMShape, k_tiles: int, hw: HardwareConfig) -> int:
    if k_tiles <= 1:
        return 0

    # 如果 tile_k 不能覆盖完整 K 维度，一个 C 元素需要多轮 K tile 累加。
    # 当 dataflow 不让 C partial sum 常驻片上时，中间 partial sum 需要：
    # - 前 k_tiles - 1 轮之后写回一次 INT32 partial sum；
    # - 后 k_tiles - 1 轮之前读回一次 INT32 partial sum。
    # 所以中间 partial sum traffic = 2 * (k_tiles - 1) * M * N * acc_bytes。
    return 2 * (k_tiles - 1) * gemm.M * gemm.N * hw.acc_bytes()


def _make_result(
    dataflow: str,
    input_read_bytes: int,
    weight_read_bytes: int,
    psum_read_write_bytes: int,
    output_write_bytes: int,
    hw: HardwareConfig,
    explanation: str,
    psum_in_sram: bool = True,
    num_c_tiles: int = 1,
) -> DataflowResult:
    total = (
        input_read_bytes
        + weight_read_bytes
        + psum_read_write_bytes
        + output_write_bytes
    )
    return DataflowResult(
        dataflow=dataflow,
        dram_bytes=total,
        memory_cycles=memory_cycles(total, hw),
        explanation=explanation,
        input_read_bytes=input_read_bytes,
        weight_read_bytes=weight_read_bytes,
        psum_read_write_bytes=psum_read_write_bytes,
        output_write_bytes=output_write_bytes,
        total_dram_bytes=total,
        psum_in_sram=psum_in_sram,
        num_c_tiles=num_c_tiles,
    )


def output_stationary(
    gemm: GEMMShape, tile: GEMMTile, hw: HardwareConfig
) -> DataflowResult:
    """估计 output-stationary：尽量让输出 partial sum 常驻片上。"""

    m_tiles, _, n_tiles = _tile_counts(gemm, tile)
    data_bytes = hw.data_bytes()

    # output_stationary 的复用对象是 C_tile / partial sum。
    # 对同一个 M/N 输出 tile，tile_k 多轮累加时尽量把 INT32 partial sum 留在片上，
    # 因此避免把中间 partial sum 写到 DRAM 再读回来。最终 C 只按 data_width 写回一次。
    # A 会随着不同 N tile 重复读取；B 会随着不同 M tile 重复读取。
    a_bytes = gemm.M * gemm.K * data_bytes * n_tiles
    b_bytes = gemm.K * gemm.N * data_bytes * m_tiles
    c_write_bytes = gemm.M * gemm.N * data_bytes

    return _make_result(
        dataflow="output_stationary",
        input_read_bytes=a_bytes,
        weight_read_bytes=b_bytes,
        psum_read_write_bytes=0,
        output_write_bytes=c_write_bytes,
        hw=hw,
        explanation=(
            "C partial sum 尽量跨 K tile 留在片上；A/B 按 tile 流式读取，"
            "最终 C 只写回一次。"
        ),
    )


def weight_stationary(
    gemm: GEMMShape, tile: GEMMTile, hw: HardwareConfig
) -> DataflowResult:
    """估计 weight-stationary：尽量让权重 tile 常驻片上。"""

    _, k_tiles, n_tiles = _tile_counts(gemm, tile)
    data_bytes = hw.data_bytes()

    # weight_stationary 的复用对象是 B_tile / 权重 tile。
    # 对同一组输出通道 tile，尽量让权重留在片上，然后遍历多个 M tile。
    # 因此 B 的 DRAM 读取按完整 B 矩阵读一次估算，不再乘以 m_tiles。
    # A 仍会随不同 N tile 重复读取。由于优先保权重，本简化模型假设 C partial
    # sum 可能在 K tile 之间发生片外读写。
    a_bytes = gemm.M * gemm.K * data_bytes * n_tiles
    b_bytes = gemm.K * gemm.N * data_bytes
    c_bytes = gemm.M * gemm.N * data_bytes
    partial_sum_bytes = _partial_sum_spill_bytes(gemm, k_tiles, hw)

    return _make_result(
        dataflow="weight_stationary",
        input_read_bytes=a_bytes,
        weight_read_bytes=b_bytes,
        psum_read_write_bytes=partial_sum_bytes,
        output_write_bytes=c_bytes,
        hw=hw,
        explanation=(
            "B/weight tile 跨多个 M tile 复用，因此减少权重 DRAM 读取；"
            "但 C partial sum 可能在 K tile 之间发生片外读写。"
        ),
        psum_in_sram=k_tiles <= 1,
        num_c_tiles=k_tiles,
    )


def input_stationary(
    gemm: GEMMShape, tile: GEMMTile, hw: HardwareConfig
) -> DataflowResult:
    """估计 input-stationary：尽量让输入 tile 常驻片上。"""

    m_tiles, k_tiles, _ = _tile_counts(gemm, tile)
    data_bytes = hw.data_bytes()

    # input_stationary 的复用对象是 A_tile / 输入 tile。
    # 对同一组输入位置，尽量让 A 留在片上，然后遍历多个输出通道 N tile。
    # 因此 A 的 DRAM 读取按完整 A 矩阵读一次估算，不再乘以 n_tiles。
    # B 仍会随不同 M tile 重复读取。由于优先保输入，本简化模型假设 C partial
    # sum 可能在 K tile 之间发生片外读写。
    a_bytes = gemm.M * gemm.K * data_bytes
    b_bytes = gemm.K * gemm.N * data_bytes * m_tiles
    c_bytes = gemm.M * gemm.N * data_bytes
    partial_sum_bytes = _partial_sum_spill_bytes(gemm, k_tiles, hw)

    return _make_result(
        dataflow="input_stationary",
        input_read_bytes=a_bytes,
        weight_read_bytes=b_bytes,
        psum_read_write_bytes=partial_sum_bytes,
        output_write_bytes=c_bytes,
        hw=hw,
        explanation=(
            "A/input tile 跨多个 N tile 复用，因此减少输入 DRAM 读取；"
            "但 C partial sum 可能在 K tile 之间发生片外读写。"
        ),
        psum_in_sram=k_tiles <= 1,
        num_c_tiles=k_tiles,
    )


def compare_dataflows(
    gemm: GEMMShape, tile: GEMMTile, hw: HardwareConfig
) -> list[DataflowResult]:
    """同时返回 OS/WS/IS 三种 dataflow 的估计结果。"""

    return [
        output_stationary(gemm, tile, hw),
        weight_stationary(gemm, tile, hw),
        input_stationary(gemm, tile, hw),
    ]


def _split_count(total: int, tile_size: int) -> int:
    return _ceil_div(total, tile_size)


def _conv_spatial_input_windows(
    layer: Conv2DLayer,
    tile: ConvTile,
) -> tuple[int, int, int]:
    out_h, out_w = layer.output_hw()
    p_tiles = _split_count(out_h, tile.tp)
    q_tiles = _split_count(out_w, tile.tq)

    input_window_sum = 0
    for p_index in range(p_tiles):
        p_size = min(tile.tp, out_h - p_index * tile.tp)
        input_h = (p_size - 1) * layer.stride + layer.kernel_h
        for q_index in range(q_tiles):
            q_size = min(tile.tq, out_w - q_index * tile.tq)
            input_w = (q_size - 1) * layer.stride + layer.kernel_w
            input_window_sum += input_h * input_w

    return p_tiles, q_tiles, input_window_sum


def _conv_base_bytes(
    layer: Conv2DLayer,
    tile: ConvTile,
    hw: HardwareConfig,
) -> tuple[int, int, int, int, int, int]:
    out_h, out_w = layer.output_hw()
    b_tiles = _split_count(layer.batch, tile.tb)
    m_tiles = _split_count(layer.out_channels, tile.tm)
    c_tiles = _split_count(layer.in_channels, tile.tc)
    p_tiles, q_tiles, input_window_sum = _conv_spatial_input_windows(layer, tile)

    input_once = (
        layer.batch
        * layer.in_channels
        * input_window_sum
        * hw.data_bytes()
    )
    weight_once = (
        layer.out_channels
        * layer.in_channels
        * layer.kernel_h
        * layer.kernel_w
        * hw.data_bytes()
    )
    output_write = (
        layer.batch
        * layer.out_channels
        * out_h
        * out_w
        * hw.acc_bytes()
    )
    output_tile_count = b_tiles * m_tiles * p_tiles * q_tiles
    spatial_tile_count = b_tiles * p_tiles * q_tiles
    return (
        input_once,
        weight_once,
        output_write,
        output_tile_count,
        spatial_tile_count,
        c_tiles,
    )


def _conv_psum_extra_bytes(
    output_write_bytes: int,
    num_c_tiles: int,
    psum_in_sram: bool,
) -> int:
    if psum_in_sram or num_c_tiles <= 1:
        return 0

    # 如果 Tc 没覆盖所有输入通道，一个输出 tile 需要跨多个 input-channel tile 累加。
    # 当 partial sum 不能留在 SRAM 时，每两个 Tc tile 的边界都需要：
    # 先写出中间 INT32 psum，下一轮再读回来继续累加。
    # 最终输出写回在 output_write_bytes 中单独统计。
    return 2 * (num_c_tiles - 1) * output_write_bytes


def output_stationary_conv(
    layer: Conv2DLayer,
    tile: ConvTile,
    hw: HardwareConfig,
    psum_in_sram: bool = True,
) -> DataflowResult:
    """估计 Conv 的 output-stationary DRAM traffic。"""

    (
        input_once,
        weight_once,
        output_write,
        _,
        spatial_tile_count,
        c_tiles,
    ) = _conv_base_bytes(layer, tile, hw)
    m_tiles = _split_count(layer.out_channels, tile.tm)

    input_bytes = input_once * m_tiles
    weight_bytes = weight_once * spatial_tile_count
    psum_extra = _conv_psum_extra_bytes(output_write, c_tiles, psum_in_sram)

    return _make_result(
        dataflow="output_stationary",
        input_read_bytes=input_bytes,
        weight_read_bytes=weight_bytes,
        psum_read_write_bytes=psum_extra,
        output_write_bytes=output_write,
        hw=hw,
        explanation=(
            "Conv output-stationary 尽量让输出 partial sum 留在片上，"
            "因此 Tc 切分不一定带来 psum DRAM traffic。"
        ),
        psum_in_sram=psum_in_sram,
        num_c_tiles=c_tiles,
    )


def weight_stationary_conv(
    layer: Conv2DLayer,
    tile: ConvTile,
    hw: HardwareConfig,
    psum_in_sram: bool = False,
) -> DataflowResult:
    """估计 Conv 的 weight-stationary DRAM traffic。"""

    (
        input_once,
        weight_once,
        output_write,
        _,
        _,
        c_tiles,
    ) = _conv_base_bytes(layer, tile, hw)
    m_tiles = _split_count(layer.out_channels, tile.tm)

    input_bytes = input_once * m_tiles
    weight_bytes = weight_once
    psum_extra = _conv_psum_extra_bytes(output_write, c_tiles, psum_in_sram)

    return _make_result(
        dataflow="weight_stationary",
        input_read_bytes=input_bytes,
        weight_read_bytes=weight_bytes,
        psum_read_write_bytes=psum_extra,
        output_write_bytes=output_write,
        hw=hw,
        explanation=(
            "Conv weight-stationary 让权重 tile 跨多个空间 tile 复用，"
            "从而减少重复加载权重。"
        ),
        psum_in_sram=psum_in_sram,
        num_c_tiles=c_tiles,
    )


def input_stationary_conv(
    layer: Conv2DLayer,
    tile: ConvTile,
    hw: HardwareConfig,
    psum_in_sram: bool = False,
) -> DataflowResult:
    """估计 Conv 的 input-stationary DRAM traffic。"""

    (
        input_once,
        weight_once,
        output_write,
        _,
        spatial_tile_count,
        c_tiles,
    ) = _conv_base_bytes(layer, tile, hw)

    input_bytes = input_once
    weight_bytes = weight_once * spatial_tile_count
    psum_extra = _conv_psum_extra_bytes(output_write, c_tiles, psum_in_sram)

    return _make_result(
        dataflow="input_stationary",
        input_read_bytes=input_bytes,
        weight_read_bytes=weight_bytes,
        psum_read_write_bytes=psum_extra,
        output_write_bytes=output_write,
        hw=hw,
        explanation=(
            "Conv input-stationary 让输入激活 tile 跨多个输出通道 tile 复用，"
            "从而减少重复加载输入。"
        ),
        psum_in_sram=psum_in_sram,
        num_c_tiles=c_tiles,
    )


def compare_conv_dataflows(
    layer: Conv2DLayer,
    tile: ConvTile,
    hw: HardwareConfig,
) -> list[DataflowResult]:
    """同时返回 Conv OS/WS/IS 三种 dataflow 的估计结果。"""

    return [
        output_stationary_conv(layer, tile, hw, psum_in_sram=True),
        weight_stationary_conv(layer, tile, hw, psum_in_sram=False),
        input_stationary_conv(layer, tile, hw, psum_in_sram=False),
    ]
