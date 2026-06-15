"""Simple tile enumeration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from tinydlp.compute_model import TileComputeResult, estimate_compute, estimate_tile_compute
from tinydlp.dataflow import (
    DataflowResult,
    input_stationary_conv,
    input_stationary,
    output_stationary_conv,
    output_stationary,
    weight_stationary_conv,
    weight_stationary,
)
from tinydlp.gemm import GEMMShape
from tinydlp.hardware import HardwareConfig
from tinydlp.layer import Conv2DLayer
from tinydlp.tile import (
    ConvTile,
    ConvTileSRAMUsage,
    GEMMTile,
    conv_tile_sram_usage_for_hw,
    conv_tile_to_gemm_tile,
    is_valid_tile,
    total_sram_bytes,
)


TILE_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
DATAFLOW_NAMES = ("output_stationary", "weight_stationary", "input_stationary")
DataflowFn = Callable[[GEMMShape, GEMMTile, HardwareConfig], DataflowResult]
ConvDataflowFn = Callable[
    [Conv2DLayer, ConvTile, HardwareConfig, bool], DataflowResult
]


@dataclass(frozen=True)
class ScheduleResult:
    """Combined compute and memory estimate for one GEMM schedule."""

    layer_name: str
    gemm: GEMMShape
    tile: GEMMTile
    dataflow: str
    macs: int
    ideal_compute_cycles: int
    array_compute_cycles: int
    systolic_compute_cycles: int
    pe_utilization: float
    dram_bytes: int
    memory_cycles: int
    no_overlap_cycles: int
    ideal_overlap_cycles: int
    bottleneck: str


@dataclass(frozen=True)
class ConvComputeSummary:
    """Compute summary for a fully tiled Conv schedule."""

    macs: int
    ideal_cycles: int
    array_aware_cycles: int
    systolic_cycles: int
    pe_utilization: float
    provided_mac_slots: int


@dataclass(frozen=True)
class ConvScheduleResult:
    """Combined compute, SRAM, and DRAM estimate for one Conv-native schedule."""

    layer_name: str
    layer: Conv2DLayer
    conv_tile: ConvTile
    gemm_tile: GEMMTile
    sram_usage: ConvTileSRAMUsage
    dataflow: str
    dram_traffic: DataflowResult
    macs: int
    ideal_compute_cycles: int
    array_compute_cycles: int
    systolic_compute_cycles: int
    pe_utilization: float
    memory_cycles: int
    no_overlap_cycles: int
    ideal_overlap_cycles: int
    total_cycles: int
    bottleneck: str
    psum_in_sram: bool


def _candidates_up_to(limit: int) -> list[int]:
    return [value for value in TILE_CANDIDATES if value <= limit]


def _unique_positive(values: list[int], limit: int | None = None) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value <= 0:
            continue
        if limit is not None and value > limit:
            continue
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _split_sizes(total: int, tile_size: int) -> list[int]:
    sizes = [tile_size] * (total // tile_size)
    remainder = total % tile_size
    if remainder:
        sizes.append(remainder)
    return sizes


def _size_counts(total: int, tile_size: int) -> dict[int, int]:
    counts: dict[int, int] = {}
    for size in _split_sizes(total, tile_size):
        counts[size] = counts.get(size, 0) + 1
    return counts


def enumerate_gemm_tiles(gemm: GEMMShape, hw: HardwareConfig) -> list[GEMMTile]:
    """Enumerate SRAM-valid GEMM tiles, sorted by SRAM footprint."""

    valid_tiles: list[GEMMTile] = []
    for tile_m in _candidates_up_to(gemm.M):
        for tile_k in _candidates_up_to(gemm.K):
            for tile_n in _candidates_up_to(gemm.N):
                tile = GEMMTile(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n)
                if is_valid_tile(tile, hw):
                    valid_tiles.append(tile)

    valid_tiles.sort(key=lambda tile: total_sram_bytes(tile, hw))
    return valid_tiles


def _tm_candidates(out_channels: int, array_n: int) -> list[int]:
    return _unique_positive(
        [array_n, 2 * array_n, 4 * array_n, out_channels],
        limit=out_channels,
    )


def _tc_candidates(in_channels: int) -> list[int]:
    return _unique_positive(
        [
            in_channels,
            in_channels // 2,
            in_channels // 4,
            32,
            16,
            8,
            4,
            2,
            1,
        ],
        limit=in_channels,
    )


def _spatial_candidates(limit: int) -> list[int]:
    return _unique_positive([1, 2, 4, 8, 16, 32, limit], limit=limit)


def generate_conv_tile_candidates(
    layer: Conv2DLayer,
    hw: HardwareConfig,
) -> list[ConvTile]:
    """Generate Conv-native tile candidates before SRAM filtering."""

    out_h, out_w = layer.output_hw()
    candidates: list[ConvTile] = []
    for tb in [1]:
        if tb > layer.batch:
            continue
        for tm in _tm_candidates(layer.out_channels, hw.array_n):
            for tc in _tc_candidates(layer.in_channels):
                for tp in _spatial_candidates(out_h):
                    for tq in _spatial_candidates(out_w):
                        candidates.append(
                            ConvTile(tb=tb, tm=tm, tc=tc, tp=tp, tq=tq)
                        )
    return candidates


def enumerate_conv_tiles(
    layer: Conv2DLayer,
    hw: HardwareConfig,
    double_buffer: bool = False,
) -> list[ConvTile]:
    """Enumerate SRAM-valid Conv-native tiles, sorted by SRAM footprint."""

    valid_tiles: list[ConvTile] = []
    for tile in generate_conv_tile_candidates(layer, hw):
        sram = conv_tile_sram_usage_for_hw(
            tile=tile,
            kernel_h=layer.kernel_h,
            kernel_w=layer.kernel_w,
            stride=layer.stride,
            hw=hw,
            double_buffer=double_buffer,
        )
        if sram.fits_sram:
            valid_tiles.append(tile)

    valid_tiles.sort(
        key=lambda tile: conv_tile_sram_usage_for_hw(
            tile=tile,
            kernel_h=layer.kernel_h,
            kernel_w=layer.kernel_w,
            stride=layer.stride,
            hw=hw,
            double_buffer=double_buffer,
        ).total_sram_bytes
    )
    return valid_tiles


def _dataflow_fn(dataflow_name: str) -> DataflowFn:
    dataflows: dict[str, DataflowFn] = {
        "output_stationary": output_stationary,
        "weight_stationary": weight_stationary,
        "input_stationary": input_stationary,
    }
    try:
        return dataflows[dataflow_name]
    except KeyError as exc:
        names = ", ".join(sorted(dataflows))
        raise ValueError(
            f"unknown dataflow {dataflow_name!r}; expected one of {names}"
        ) from exc


def _conv_dataflow_fn(dataflow_name: str) -> ConvDataflowFn:
    dataflows: dict[str, ConvDataflowFn] = {
        "output_stationary": output_stationary_conv,
        "weight_stationary": weight_stationary_conv,
        "input_stationary": input_stationary_conv,
    }
    try:
        return dataflows[dataflow_name]
    except KeyError as exc:
        names = ", ".join(sorted(dataflows))
        raise ValueError(
            f"unknown dataflow {dataflow_name!r}; expected one of {names}"
        ) from exc


def _default_conv_psum_in_sram(dataflow_name: str) -> bool:
    return dataflow_name == "output_stationary"


def _objective_cycles(result: ScheduleResult, overlap_mode: str) -> int:
    if overlap_mode == "none":
        return result.no_overlap_cycles
    if overlap_mode == "ideal":
        return result.ideal_overlap_cycles
    raise ValueError("overlap_mode must be 'none' or 'ideal'")


def evaluate_schedule(
    gemm: GEMMShape,
    hw: HardwareConfig,
    tile: GEMMTile,
    dataflow_name: str,
) -> ScheduleResult:
    """Evaluate compute, memory, and bottleneck estimates for one schedule."""

    compute = estimate_compute(gemm, hw)
    memory = _dataflow_fn(dataflow_name)(gemm, tile, hw)

    # 本调度评估默认使用 systolic_compute_cycles 作为实际 compute_cycles，
    # 因为它比 ideal 和 array-aware 模型多考虑了简化的 fill/drain 开销。
    compute_cycles = compute.systolic_cycles

    # no_overlap 表示搬运和计算完全串行执行：
    # 总周期 = 计算周期 + 内存搬运周期。
    no_overlap_cycles = compute_cycles + memory.memory_cycles

    # ideal_overlap 表示通过双缓冲等方式，把搬运和计算做理想重叠：
    # 总周期 = max(计算周期, 内存周期)。
    # 这是一个性能下界，不是真实硬件上的精确执行时间。
    ideal_overlap_cycles = max(compute_cycles, memory.memory_cycles)

    bottleneck = (
        "compute-bound"
        if compute_cycles > memory.memory_cycles
        else "memory-bound"
    )

    return ScheduleResult(
        layer_name=gemm.name,
        gemm=gemm,
        tile=tile,
        dataflow=memory.dataflow,
        macs=compute.macs,
        ideal_compute_cycles=compute.ideal_cycles,
        array_compute_cycles=compute.array_aware_cycles,
        systolic_compute_cycles=compute.systolic_cycles,
        pe_utilization=compute.pe_utilization_systolic,
        dram_bytes=memory.dram_bytes,
        memory_cycles=memory.memory_cycles,
        no_overlap_cycles=no_overlap_cycles,
        ideal_overlap_cycles=ideal_overlap_cycles,
        bottleneck=bottleneck,
    )


def _estimate_tiled_conv_compute(
    layer: Conv2DLayer,
    tile: ConvTile,
    hw: HardwareConfig,
) -> ConvComputeSummary:
    out_h, out_w = layer.output_hw()
    b_counts = _size_counts(layer.batch, tile.tb)
    p_counts = _size_counts(out_h, tile.tp)
    q_counts = _size_counts(out_w, tile.tq)
    m_counts = _size_counts(layer.out_channels, tile.tm)
    c_counts = _size_counts(layer.in_channels, tile.tc)

    ideal_cycles = 0
    array_aware_cycles = 0
    systolic_cycles = 0
    useful_macs = 0
    provided_slots = 0

    for b_size, b_count in b_counts.items():
        for p_size, p_count in p_counts.items():
            for q_size, q_count in q_counts.items():
                M_tile = b_size * p_size * q_size
                spatial_count = b_count * p_count * q_count
                for m_size, m_count in m_counts.items():
                    N_tile = m_size
                    for c_size, c_count in c_counts.items():
                        K_tile = c_size * layer.kernel_h * layer.kernel_w
                        multiplicity = spatial_count * m_count * c_count
                        compute = estimate_tile_compute(
                            M_tile=M_tile,
                            N_tile=N_tile,
                            K_tile=K_tile,
                            array_m=hw.array_m,
                            array_n=hw.array_n,
                            mac_per_pe_per_cycle=hw.mac_per_pe_per_cycle,
                        )
                        ideal_cycles += compute.ideal_cycles * multiplicity
                        array_aware_cycles += (
                            compute.array_aware_cycles * multiplicity
                        )
                        systolic_cycles += compute.systolic_cycles * multiplicity
                        useful_macs += compute.useful_macs * multiplicity
                        provided_slots += compute.provided_mac_slots * multiplicity

    pe_utilization = useful_macs / provided_slots if provided_slots else 0.0
    return ConvComputeSummary(
        macs=useful_macs,
        ideal_cycles=ideal_cycles,
        array_aware_cycles=array_aware_cycles,
        systolic_cycles=systolic_cycles,
        pe_utilization=pe_utilization,
        provided_mac_slots=provided_slots,
    )


def evaluate_conv_schedule(
    layer: Conv2DLayer,
    hw: HardwareConfig,
    tile: ConvTile,
    dataflow_name: str,
    overlap_mode: str = "ideal",
    psum_in_sram: bool | None = None,
    double_buffer: bool = False,
) -> ConvScheduleResult:
    """Evaluate compute, SRAM, and DRAM estimates for one Conv schedule."""

    sram = conv_tile_sram_usage_for_hw(
        tile=tile,
        kernel_h=layer.kernel_h,
        kernel_w=layer.kernel_w,
        stride=layer.stride,
        hw=hw,
        double_buffer=double_buffer,
    )
    gemm_tile = conv_tile_to_gemm_tile(tile, layer.kernel_h, layer.kernel_w)
    compute = _estimate_tiled_conv_compute(layer, tile, hw)

    keep_psum = (
        _default_conv_psum_in_sram(dataflow_name)
        if psum_in_sram is None
        else psum_in_sram
    )
    memory = _conv_dataflow_fn(dataflow_name)(layer, tile, hw, keep_psum)

    compute_cycles = compute.systolic_cycles
    no_overlap_cycles = compute_cycles + memory.memory_cycles
    ideal_overlap_cycles = max(compute_cycles, memory.memory_cycles)
    bottleneck = (
        "compute-bound"
        if compute_cycles > memory.memory_cycles
        else "memory-bound"
    )
    total_cycles = (
        no_overlap_cycles if overlap_mode == "none" else ideal_overlap_cycles
    )
    if overlap_mode not in {"none", "ideal"}:
        raise ValueError("overlap_mode must be 'none' or 'ideal'")

    return ConvScheduleResult(
        layer_name=layer.name,
        layer=layer,
        conv_tile=tile,
        gemm_tile=gemm_tile,
        sram_usage=sram,
        dataflow=memory.dataflow,
        dram_traffic=memory,
        macs=compute.macs,
        ideal_compute_cycles=compute.ideal_cycles,
        array_compute_cycles=compute.array_aware_cycles,
        systolic_compute_cycles=compute.systolic_cycles,
        pe_utilization=compute.pe_utilization,
        memory_cycles=memory.memory_cycles,
        no_overlap_cycles=no_overlap_cycles,
        ideal_overlap_cycles=ideal_overlap_cycles,
        total_cycles=total_cycles,
        bottleneck=bottleneck,
        psum_in_sram=keep_psum,
    )


def search_topk_conv_schedules(
    layer: Conv2DLayer,
    hw: HardwareConfig,
    k: int = 10,
    overlap_mode: str = "ideal",
    double_buffer: bool = False,
) -> list[ConvScheduleResult]:
    """Search legal Conv tiles and dataflows, returning the top-k schedules."""

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if overlap_mode not in {"none", "ideal"}:
        raise ValueError("overlap_mode must be 'none' or 'ideal'")

    tiles = enumerate_conv_tiles(layer, hw, double_buffer=double_buffer)
    if not tiles:
        raise ValueError(
            "no legal Conv tile found; SRAM may be too small or the tile "
            "candidate set may not contain a suitable tile"
        )

    results: list[ConvScheduleResult] = []
    for tile in tiles:
        for dataflow_name in DATAFLOW_NAMES:
            results.append(
                evaluate_conv_schedule(
                    layer=layer,
                    hw=hw,
                    tile=tile,
                    dataflow_name=dataflow_name,
                    overlap_mode=overlap_mode,
                    double_buffer=double_buffer,
                )
            )

    results.sort(
        key=lambda result: (
            result.total_cycles,
            result.dram_traffic.total_dram_bytes,
            result.systolic_compute_cycles,
            result.sram_usage.total_sram_bytes,
            result.dataflow,
        )
    )
    return results[:k]


def search_best_conv_schedule(
    layer: Conv2DLayer,
    hw: HardwareConfig,
    overlap_mode: str = "ideal",
    double_buffer: bool = False,
) -> ConvScheduleResult:
    """Return the best Conv-native schedule under the selected objective."""

    return search_topk_conv_schedules(
        layer=layer,
        hw=hw,
        k=1,
        overlap_mode=overlap_mode,
        double_buffer=double_buffer,
    )[0]


def search_topk_schedules(
    gemm: GEMMShape,
    hw: HardwareConfig,
    k: int = 10,
    overlap_mode: str = "ideal",
) -> list[ScheduleResult]:
    """Search legal tiles and dataflows, returning the top-k schedules."""

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    tiles = enumerate_gemm_tiles(gemm, hw)
    if not tiles:
        raise ValueError(
            "no legal GEMM tile found; SRAM may be too small or the tile "
            "candidate set may not contain a suitable tile"
        )

    results: list[ScheduleResult] = []
    for tile in tiles:
        for dataflow_name in DATAFLOW_NAMES:
            results.append(evaluate_schedule(gemm, hw, tile, dataflow_name))

    results.sort(
        key=lambda result: (
            _objective_cycles(result, overlap_mode),
            result.dram_bytes,
            result.systolic_compute_cycles,
            total_sram_bytes(result.tile, hw),
            result.dataflow,
        )
    )
    return results[:k]


def search_best_schedule(
    gemm: GEMMShape,
    hw: HardwareConfig,
    overlap_mode: str = "ideal",
) -> ScheduleResult:
    """Return the best schedule under the selected overlap objective."""

    return search_topk_schedules(
        gemm=gemm,
        hw=hw,
        k=1,
        overlap_mode=overlap_mode,
    )[0]


def pretty_print_schedule(result: ScheduleResult) -> str:
    """Print and return a human-readable schedule summary."""

    text = "\n".join(
        [
            "ScheduleResult",
            f"  GEMM shape: M={result.gemm.M}, K={result.gemm.K}, N={result.gemm.N}",
            (
                "  tile: "
                f"tile_m={result.tile.tile_m}, "
                f"tile_k={result.tile.tile_k}, "
                f"tile_n={result.tile.tile_n}"
            ),
            f"  dataflow: {result.dataflow}",
            f"  MACs: {result.macs}",
            f"  ideal compute cycles: {result.ideal_compute_cycles}",
            f"  array-aware compute cycles: {result.array_compute_cycles}",
            f"  systolic compute cycles: {result.systolic_compute_cycles}",
            f"  PE utilization: {result.pe_utilization:.4f}",
            f"  DRAM traffic: {result.dram_bytes} bytes",
            f"  memory cycles: {result.memory_cycles}",
            f"  no-overlap cycles: {result.no_overlap_cycles}",
            f"  ideal-overlap cycles: {result.ideal_overlap_cycles}",
            f"  bottleneck: {result.bottleneck}",
        ]
    )
    print(text)
    return text


def format_conv_tile(tile: ConvTile) -> str:
    """Return a compact Conv tile string."""

    return (
        f"Tb={tile.tb},Tm={tile.tm},Tc={tile.tc},"
        f"Tp={tile.tp},Tq={tile.tq}"
    )


def pretty_print_conv_schedule(result: ConvScheduleResult) -> str:
    """Print and return a human-readable Conv schedule summary."""

    traffic = result.dram_traffic
    sram = result.sram_usage
    text = "\n".join(
        [
            "ConvScheduleResult",
            f"  Conv tile: {format_conv_tile(result.conv_tile)}",
            (
                "  GEMM tile: "
                f"M_tile={result.gemm_tile.tile_m}, "
                f"K_tile={result.gemm_tile.tile_k}, "
                f"N_tile={result.gemm_tile.tile_n}"
            ),
            (
                "  SRAM: "
                f"input={sram.input_bytes}, weight={sram.weight_bytes}, "
                f"psum={sram.psum_bytes}, total={sram.total_sram_bytes}, "
                f"fits={sram.fits_sram}"
            ),
            f"  dataflow: {result.dataflow}",
            f"  psum in SRAM: {result.psum_in_sram}",
            f"  MACs: {result.macs}",
            f"  ideal compute cycles: {result.ideal_compute_cycles}",
            f"  array-aware compute cycles: {result.array_compute_cycles}",
            f"  systolic compute cycles: {result.systolic_compute_cycles}",
            f"  PE utilization: {result.pe_utilization:.4f}",
            (
                "  DRAM traffic: "
                f"input={traffic.input_read_bytes}, "
                f"weight={traffic.weight_read_bytes}, "
                f"psum_extra={traffic.psum_read_write_bytes}, "
                f"output={traffic.output_write_bytes}, "
                f"total={traffic.total_dram_bytes}"
            ),
            f"  memory cycles: {result.memory_cycles}",
            f"  no-overlap cycles: {result.no_overlap_cycles}",
            f"  ideal-overlap cycles: {result.ideal_overlap_cycles}",
            f"  bottleneck: {result.bottleneck}",
        ]
    )
    print(text)
    return text


def demo() -> None:
    """Print a small tile-enumeration demo."""

    gemm = GEMMShape(M=1024, K=256, N=256, name="demo_gemm")
    hw = HardwareConfig(name="dlp_16x16", array_m=16, array_n=16, sram_kb=64)
    tiles = enumerate_gemm_tiles(gemm, hw)

    print(f"GEMM: {gemm.pretty_summary()}")
    print(f"Hardware: {hw.pretty_summary()}")
    print(f"valid tile count: {len(tiles)}")
    print("first 10 tiles sorted by SRAM bytes:")
    for tile in tiles[:10]:
        print(f"  {tile} -> {total_sram_bytes(tile, hw)} bytes")


if __name__ == "__main__":
    demo()
