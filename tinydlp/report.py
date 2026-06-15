"""CSV and Markdown report generation."""

from __future__ import annotations

import csv
from pathlib import Path

from tinydlp.hardware import HardwareConfig
from tinydlp.layer import Conv2DLayer, FullyConnectedLayer
from tinydlp.scheduler import ScheduleResult


Layer = Conv2DLayer | FullyConnectedLayer


CSV_FIELDS = [
    "layer_name",
    "layer_type",
    "conv_batch",
    "conv_C",
    "conv_H",
    "conv_W",
    "conv_Kout",
    "conv_R",
    "conv_S",
    "conv_stride",
    "conv_padding",
    "conv_output_P",
    "conv_output_Q",
    "fc_batch",
    "fc_in_features",
    "fc_out_features",
    "M",
    "K",
    "N",
    "gemm_M_output_positions",
    "gemm_K_reduction_CRS",
    "gemm_N_output_channels",
    "MACs",
    "tile_m",
    "tile_k",
    "tile_n",
    "tile_m_meaning",
    "tile_k_meaning",
    "tile_n_meaning",
    "dataflow",
    "ideal_compute_cycles",
    "array_compute_cycles",
    "systolic_compute_cycles",
    "pe_utilization",
    "dram_bytes",
    "memory_cycles",
    "no_overlap_cycles",
    "ideal_overlap_cycles",
    "bottleneck",
]


def _blank_layer_fields() -> dict[str, object]:
    return {
        "layer_type": "",
        "conv_batch": "",
        "conv_C": "",
        "conv_H": "",
        "conv_W": "",
        "conv_Kout": "",
        "conv_R": "",
        "conv_S": "",
        "conv_stride": "",
        "conv_padding": "",
        "conv_output_P": "",
        "conv_output_Q": "",
        "fc_batch": "",
        "fc_in_features": "",
        "fc_out_features": "",
        "tile_m_meaning": "GEMM output rows",
        "tile_k_meaning": "GEMM reduction slice",
        "tile_n_meaning": "GEMM output columns",
    }


def _layer_fields(layer: Layer | None) -> dict[str, object]:
    fields = _blank_layer_fields()
    if isinstance(layer, Conv2DLayer):
        out_h, out_w = layer.output_hw()
        fields.update(
            {
                "layer_type": "conv2d",
                "conv_batch": layer.batch,
                "conv_C": layer.in_channels,
                "conv_H": layer.in_h,
                "conv_W": layer.in_w,
                "conv_Kout": layer.out_channels,
                "conv_R": layer.kernel_h,
                "conv_S": layer.kernel_w,
                "conv_stride": layer.stride,
                "conv_padding": layer.padding,
                "conv_output_P": out_h,
                "conv_output_Q": out_w,
                "tile_m_meaning": "output positions: batch*Tp*Tq",
                "tile_k_meaning": "reduction: Tc*R*S",
                "tile_n_meaning": "output channels: Tm",
            }
        )
    elif isinstance(layer, FullyConnectedLayer):
        fields.update(
            {
                "layer_type": "fc",
                "fc_batch": layer.batch,
                "fc_in_features": layer.in_features,
                "fc_out_features": layer.out_features,
                "tile_m_meaning": "FC batch rows",
                "tile_k_meaning": "input feature reduction",
                "tile_n_meaning": "output features",
            }
        )
    return fields


def _result_row(result: ScheduleResult, layer: Layer | None = None) -> dict[str, object]:
    row = {
        "layer_name": result.layer_name,
        "M": result.gemm.M,
        "K": result.gemm.K,
        "N": result.gemm.N,
        "gemm_M_output_positions": result.gemm.M,
        "gemm_K_reduction_CRS": result.gemm.K,
        "gemm_N_output_channels": result.gemm.N,
        "MACs": result.macs,
        "tile_m": result.tile.tile_m,
        "tile_k": result.tile.tile_k,
        "tile_n": result.tile.tile_n,
        "dataflow": result.dataflow,
        "ideal_compute_cycles": result.ideal_compute_cycles,
        "array_compute_cycles": result.array_compute_cycles,
        "systolic_compute_cycles": result.systolic_compute_cycles,
        "pe_utilization": result.pe_utilization,
        "dram_bytes": result.dram_bytes,
        "memory_cycles": result.memory_cycles,
        "no_overlap_cycles": result.no_overlap_cycles,
        "ideal_overlap_cycles": result.ideal_overlap_cycles,
        "bottleneck": result.bottleneck,
    }
    row.update(_layer_fields(layer))
    return row


def write_result_csv(
    results: list[ScheduleResult],
    output_dir: str | Path,
    layers: list[Layer] | None = None,
) -> Path:
    """Write per-layer schedule results to result.csv."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "result.csv"

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for index, result in enumerate(results):
            layer = layers[index] if layers is not None and index < len(layers) else None
            writer.writerow(_result_row(result, layer))

    return csv_path


def _hardware_markdown(hw: HardwareConfig | None) -> str:
    if hw is None:
        return "未提供硬件配置对象。\n"

    return "\n".join(
        [
            f"- name: `{hw.name}`",
            f"- array: `{hw.array_m} x {hw.array_n}`",
            f"- MAC/PE/cycle: `{hw.mac_per_pe_per_cycle}`",
            f"- frequency: `{hw.frequency_mhz} MHz`",
            f"- SRAM: `{hw.sram_kb} KB`",
            f"- DRAM bandwidth: `{hw.dram_bandwidth_gb_s} GB/s`",
            f"- data width: `{hw.data_width_bits} bits`",
            f"- accumulator width: `{hw.acc_width_bits} bits`",
            "",
        ]
    )


def _layer_shape(layer: Layer | None) -> str:
    if isinstance(layer, Conv2DLayer):
        out_h, out_w = layer.output_hw()
        return (
            f"Conv N={layer.batch}, C={layer.in_channels}, "
            f"HxW={layer.in_h}x{layer.in_w}, Kout={layer.out_channels}, "
            f"RxS={layer.kernel_h}x{layer.kernel_w}, P/Q={out_h}x{out_w}"
        )
    if isinstance(layer, FullyConnectedLayer):
        return (
            f"FC batch={layer.batch}, in={layer.in_features}, "
            f"out={layer.out_features}"
        )
    return ""


def _layer_table(
    results: list[ScheduleResult],
    layers: list[Layer] | None = None,
) -> str:
    header = (
        "| layer | original shape | GEMM M/K/N | MACs | tile | dataflow | PE util | "
        "DRAM bytes | no-overlap | ideal-overlap | bottleneck |"
    )
    separator = "|---|---|---:|---:|---|---|---:|---:|---:|---:|---|"
    rows = [header, separator]
    for index, result in enumerate(results):
        layer = layers[index] if layers is not None and index < len(layers) else None
        tile = f"{result.tile.tile_m}x{result.tile.tile_k}x{result.tile.tile_n}"
        gemm = f"{result.gemm.M}/{result.gemm.K}/{result.gemm.N}"
        rows.append(
            "| "
            f"{result.layer_name} | "
            f"{_layer_shape(layer)} | "
            f"{gemm} | "
            f"{result.macs} | "
            f"{tile} | "
            f"{result.dataflow} | "
            f"{result.pe_utilization:.4f} | "
            f"{result.dram_bytes} | "
            f"{result.no_overlap_cycles} | "
            f"{result.ideal_overlap_cycles} | "
            f"{result.bottleneck} |"
        )
    return "\n".join(rows)


def write_summary_md(
    results: list[ScheduleResult],
    output_dir: str | Path,
    hw: HardwareConfig | None = None,
    layers: list[Layer] | None = None,
) -> Path:
    """Write a Markdown summary report."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    md_path = output_path / "summary.md"

    total_macs = sum(result.macs for result in results)
    total_dram_bytes = sum(result.dram_bytes for result in results)
    total_no_overlap_cycles = sum(result.no_overlap_cycles for result in results)
    total_ideal_overlap_cycles = sum(result.ideal_overlap_cycles for result in results)

    text = "\n".join(
        [
            "# TinyDLP-Scheduler Report",
            "",
            "## 项目说明",
            "",
            "TinyDLP-Scheduler 用于学习 Conv/GEMM 在简化深度学习处理器上的映射、"
            "tiling、dataflow、PE 利用率和瓶颈分析。该项目是分析模型，"
            "不是 RTL 级 cycle-accurate 仿真。",
            "",
            "## 硬件配置",
            "",
            _hardware_markdown(hw),
            "## 每层结果",
            "",
            _layer_table(results, layers),
            "",
            "## Network Summary",
            "",
            f"- 总 MACs: `{total_macs}`",
            f"- 总 DRAM traffic: `{total_dram_bytes}` bytes",
            f"- 总 no-overlap cycles: `{total_no_overlap_cycles}`",
            f"- 总 ideal-overlap cycles: `{total_ideal_overlap_cycles}`",
            "",
            "## Bottleneck 解释",
            "",
            "- `compute-bound`: systolic compute cycles 大于 memory cycles，"
            "说明当前估计下主要受计算阵列执行时间限制。",
            "- `memory-bound`: memory cycles 大于或等于 systolic compute cycles，"
            "说明当前估计下主要受 DRAM 搬运时间限制。",
            "",
            "## Overlap 解释",
            "",
            "- `no-overlap`: 搬运和计算完全串行，"
            "`cycles = compute_cycles + memory_cycles`。",
            "- `ideal-overlap`: 假设通过双缓冲等方式理想重叠搬运和计算，"
            "`cycles = max(compute_cycles, memory_cycles)`。这是性能下界，"
            "不是真实硬件精确时间。",
            "",
        ]
    )

    md_path.write_text(text, encoding="utf-8")
    return md_path


def generate_reports(
    results: list[ScheduleResult],
    output_dir: str | Path = "reports",
    hw: HardwareConfig | None = None,
    layers: list[Layer] | None = None,
) -> tuple[Path, Path]:
    """Generate result.csv and summary.md."""

    csv_path = write_result_csv(results, output_dir, layers)
    md_path = write_summary_md(results, output_dir, hw, layers)
    return csv_path, md_path
