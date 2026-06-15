"""Show how SRAM capacity changes Conv tiling, dataflow, and DRAM traffic.

Run from the project root:

    python examples/sram_dataflow_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinydlp.hardware import HardwareConfig
from tinydlp.layer import Conv2DLayer
from tinydlp.plot_style import COLORS, DATAFLOW_COLORS, plt, save_figure, style_axes
from tinydlp.scheduler import (
    ConvScheduleResult,
    DATAFLOW_NAMES,
    enumerate_conv_tiles,
    evaluate_conv_schedule,
    format_conv_tile,
)


SRAM_KB_VALUES = [8, 16, 32, 64, 128]
FIG_PATH = PROJECT_ROOT / "figs" / "sram_dataflow_demo.png"
DATAFLOW_LABELS = {
    "output_stationary": "output-stationary (keep psum)",
    "weight_stationary": "weight-stationary (reuse weights)",
    "input_stationary": "input-stationary (reuse inputs)",
}


def build_conv_layer() -> Conv2DLayer:
    return Conv2DLayer(
        name="sram_dataflow_conv",
        batch=1,
        in_channels=64,
        in_h=32,
        in_w=32,
        out_channels=128,
        kernel_h=3,
        kernel_w=3,
        stride=1,
        padding=1,
    )


def build_hardware(sram_kb: int) -> HardwareConfig:
    return HardwareConfig(
        name=f"TinyDLP-16x16-{sram_kb}KB",
        array_m=16,
        array_n=16,
        frequency_mhz=500.0,
        sram_kb=sram_kb,
        dram_bandwidth_gb_s=12.8,
        data_width_bits=8,
        acc_width_bits=32,
    )


def _best_for_dataflow(
    layer: Conv2DLayer,
    hw: HardwareConfig,
    dataflow: str,
) -> ConvScheduleResult:
    tiles = enumerate_conv_tiles(layer, hw)
    results = [
        evaluate_conv_schedule(
            layer=layer,
            hw=hw,
            tile=tile,
            dataflow_name=dataflow,
            overlap_mode="ideal",
        )
        for tile in tiles
    ]
    return min(
        results,
        key=lambda result: (
            result.total_cycles,
            result.dram_traffic.total_dram_bytes,
            result.systolic_compute_cycles,
        ),
    )


def evaluate_sweep() -> list[tuple[int, dict[str, ConvScheduleResult]]]:
    layer = build_conv_layer()
    rows: list[tuple[int, dict[str, ConvScheduleResult]]] = []
    for sram_kb in SRAM_KB_VALUES:
        hw = build_hardware(sram_kb)
        per_dataflow = {
            dataflow: _best_for_dataflow(layer, hw, dataflow)
            for dataflow in DATAFLOW_NAMES
        }
        rows.append((sram_kb, per_dataflow))
    return rows


def _kb(value: int) -> float:
    return value / 1024


def _best_overall(
    per_dataflow: dict[str, ConvScheduleResult],
) -> ConvScheduleResult:
    return min(
        per_dataflow.values(),
        key=lambda result: (
            result.total_cycles,
            result.dram_traffic.total_dram_bytes,
            result.systolic_compute_cycles,
        ),
    )


def print_results(rows: list[tuple[int, dict[str, ConvScheduleResult]]]) -> None:
    header = (
        f"{'SRAM':>8} {'dataflow':>20} {'best Conv tile':>34} "
        f"{'SRAM use':>12} {'DRAM KB':>12} {'compute':>12} "
        f"{'memory':>12} {'total':>12}"
    )
    print("Table: SRAM sweep with per-dataflow best Conv tiles")
    print("Question: when SRAM is smaller, how much extra DRAM traffic does each reuse policy create?")
    print("Conv example: input N=1,C=64,H=W=32; Kout=128; kernel R=S=3; stride=1; padding=1")
    print("Hardware: 16x16 PE array, int8 activations/weights, int32 partial sums")
    print()
    print(header)
    print("-" * len(header))
    for sram_kb, per_dataflow in rows:
        best = _best_overall(per_dataflow)
        for dataflow in DATAFLOW_NAMES:
            result = per_dataflow[dataflow]
            marker = "*" if result is best else " "
            print(
                f"{str(sram_kb) + 'KB':>8} "
                f"{(marker + ' ' + dataflow):>20} "
                f"{format_conv_tile(result.conv_tile):>34} "
                f"{_kb(result.sram_usage.total_sram_bytes):>10.1f}KB "
                f"{_kb(result.dram_traffic.total_dram_bytes):>12.1f} "
                f"{result.systolic_compute_cycles:>12} "
                f"{result.memory_cycles:>12} "
                f"{result.total_cycles:>12}"
            )
        print("-" * len(header))
    print("* marks the lowest-cycle schedule; ties are broken by lower DRAM traffic.")


def plot_results(rows: list[tuple[int, dict[str, ConvScheduleResult]]]) -> None:
    sram_values = [sram_kb for sram_kb, _ in rows]
    labels = [f"{value}KB" for value in sram_values]

    fig, (ax_cycles, ax_dram) = plt.subplots(
        2,
        1,
        figsize=(9.5, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.15]},
    )
    fig.suptitle("SRAM Sweep: Best Conv Tile per Dataflow", fontsize=16, fontweight="bold")

    for dataflow in DATAFLOW_NAMES:
        color = DATAFLOW_COLORS.get(dataflow, COLORS["gray"])
        cycles = [
            per_dataflow[dataflow].total_cycles for _, per_dataflow in rows
        ]
        dram_mb = [
            per_dataflow[dataflow].dram_traffic.total_dram_bytes / (1024 * 1024)
            for _, per_dataflow in rows
        ]
        label = DATAFLOW_LABELS[dataflow]
        ax_cycles.plot(
            sram_values,
            cycles,
            color=color,
            marker="o",
            linewidth=2.4,
            label=label,
        )
        ax_dram.plot(
            sram_values,
            dram_mb,
            color=color,
            marker="o",
            linewidth=2.4,
            label=label,
        )

    ax_cycles.set_ylabel("total cycles")
    ax_cycles.set_title("Compute-bound here: cycles are close, traffic tells the reuse story")
    style_axes(ax_cycles, grid_axis="both")
    ax_cycles.legend(loc="upper right")

    ax_dram.set_xlabel("SRAM capacity")
    ax_dram.set_ylabel("DRAM traffic (MB)")
    ax_dram.set_xticks(sram_values)
    ax_dram.set_xticklabels(labels)
    style_axes(ax_dram, grid_axis="both")
    ax_dram.legend(loc="upper right")

    save_figure(fig, FIG_PATH)
    print()
    print(f"Saved figure: {FIG_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    rows = evaluate_sweep()
    print_results(rows)
    plot_results(rows)


if __name__ == "__main__":
    main()
