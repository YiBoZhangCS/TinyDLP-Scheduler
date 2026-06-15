"""SRAM sensitivity demo for TinyDLP schedule search.

Run from the project root:

    python examples/sram_sensitivity_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinydlp.hardware import HardwareConfig
from tinydlp.layer import Conv2DLayer
from tinydlp.plot_style import COLORS, plt, save_figure, style_axes
from tinydlp.scheduler import ScheduleResult, search_best_schedule


SRAM_KB_VALUES = [16, 32, 64, 128, 256, 512]
FIG_PATH = PROJECT_ROOT / "figs" / "sram_sensitivity.png"


def build_conv_layer() -> Conv2DLayer:
    return Conv2DLayer(
        name="sram_sensitivity_conv",
        batch=1,
        in_channels=32,
        in_h=32,
        in_w=32,
        out_channels=64,
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
    )


def evaluate_sram_sweep() -> list[tuple[int, ScheduleResult]]:
    layer = build_conv_layer()
    gemm = layer.to_gemm_shape()
    results: list[tuple[int, ScheduleResult]] = []

    for sram_kb in SRAM_KB_VALUES:
        hw = build_hardware(sram_kb)
        schedule = search_best_schedule(gemm=gemm, hw=hw, overlap_mode="ideal")
        results.append((sram_kb, schedule))

    return results


def format_tile(result: ScheduleResult) -> str:
    return (
        f"{result.tile.tile_m}x"
        f"{result.tile.tile_k}x"
        f"{result.tile.tile_n}"
    )


def print_results(results: list[tuple[int, ScheduleResult]]) -> None:
    header = (
        f"{'SRAM KB':>8} {'best tile':>14} {'dataflow':>20} "
        f"{'dram_bytes':>14} {'ideal_overlap':>16} "
        f"{'PE util':>10} {'bottleneck':>14}"
    )

    print("SRAM sensitivity demo")
    print(
        "This experiment shows that larger on-chip SRAM can hold larger tiles, "
        "which may reduce DRAM traffic or improve the selected schedule."
    )
    print()
    print(header)
    print("-" * len(header))
    for sram_kb, result in results:
        print(
            f"{sram_kb:>8} "
            f"{format_tile(result):>14} "
            f"{result.dataflow:>20} "
            f"{result.dram_bytes:>14} "
            f"{result.ideal_overlap_cycles:>16} "
            f"{result.pe_utilization:>10.4f} "
            f"{result.bottleneck:>14}"
        )

    if len({result.ideal_overlap_cycles for _, result in results}) == 1:
        print()
        print(
            "Note: ideal-overlap cycles are flat because all selected schedules "
            "are compute-bound; SRAM still reduces DRAM traffic, shown in the plot."
        )


def plot_results(results: list[tuple[int, ScheduleResult]]) -> None:
    sram_values = [sram_kb for sram_kb, _ in results]
    cycles = [result.ideal_overlap_cycles for _, result in results]
    dram_kb = [result.dram_bytes / 1024 for _, result in results]

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(
        sram_values,
        cycles,
        marker="o",
        linewidth=2.4,
        color=COLORS["blue"],
    )
    axes[0].set_title("SRAM Sensitivity")
    axes[0].set_ylabel("ideal-overlap cycles")
    style_axes(axes[0], grid_axis="both")

    axes[1].plot(
        sram_values,
        dram_kb,
        marker="o",
        linewidth=2.4,
        color=COLORS["orange"],
    )
    axes[1].set_xlabel("SRAM capacity (KB)")
    axes[1].set_ylabel("DRAM traffic (KB)")
    axes[1].set_xticks(sram_values)
    axes[1].tick_params(axis="x", labelrotation=35)
    style_axes(axes[1], grid_axis="both")

    save_figure(fig, FIG_PATH)

    print()
    print(f"Saved figure: {FIG_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    results = evaluate_sram_sweep()
    print_results(results)
    plot_results(results)


if __name__ == "__main__":
    main()
