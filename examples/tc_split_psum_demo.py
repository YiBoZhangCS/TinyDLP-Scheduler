"""Show partial-sum traffic when input channels are split by Tc.

Run from the project root:

    python examples/tc_split_psum_demo.py
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
from tinydlp.scheduler import ConvScheduleResult, evaluate_conv_schedule
from tinydlp.tile import ConvTile


FIG_PATH = PROJECT_ROOT / "figs" / "tc_split_psum_demo.png"


def build_conv_layer() -> Conv2DLayer:
    return Conv2DLayer(
        name="tc_split_psum_conv",
        batch=1,
        in_channels=256,
        in_h=14,
        in_w=14,
        out_channels=64,
        kernel_h=3,
        kernel_w=3,
        stride=1,
        padding=1,
    )


def build_hardware() -> HardwareConfig:
    return HardwareConfig(
        name="TinyDLP-16x16-512KB",
        array_m=16,
        array_n=16,
        frequency_mhz=500.0,
        sram_kb=512,
        dram_bandwidth_gb_s=12.8,
        data_width_bits=8,
        acc_width_bits=32,
    )


def evaluate_cases() -> list[tuple[str, ConvScheduleResult]]:
    layer = build_conv_layer()
    hw = build_hardware()
    out_h, out_w = layer.output_hw()
    cases = [
        (
            "A. Tc=256",
            ConvTile(tb=1, tm=64, tc=256, tp=out_h, tq=out_w),
            True,
        ),
        (
            "B. Tc=64, psum SRAM",
            ConvTile(tb=1, tm=64, tc=64, tp=out_h, tq=out_w),
            True,
        ),
        (
            "C. Tc=64, psum DRAM",
            ConvTile(tb=1, tm=64, tc=64, tp=out_h, tq=out_w),
            False,
        ),
    ]

    results: list[tuple[str, ConvScheduleResult]] = []
    for label, tile, psum_in_sram in cases:
        result = evaluate_conv_schedule(
            layer=layer,
            hw=hw,
            tile=tile,
            dataflow_name="output_stationary",
            overlap_mode="ideal",
            psum_in_sram=psum_in_sram,
        )
        results.append((label, result))
    return results


def _mb(value: int) -> float:
    return value / (1024 * 1024)


def print_results(rows: list[tuple[str, ConvScheduleResult]]) -> None:
    header = (
        f"{'strategy':<24} {'Tc':>6} {'c_tiles':>8} {'psum SRAM':>10} "
        f"{'DRAM MB':>10} {'memory':>10} {'total':>10}"
    )
    print("Table: Input-channel split and partial-sum spill")
    print("Question: what happens when Tc<C and partial sums cannot stay in SRAM?")
    print("Conv example: input N=1,C=256,H=W=14; Kout=64; kernel R=S=3; stride=1; padding=1")
    print("Hardware: 16x16 PE array, int8 activations/weights, int32 partial sums")
    print()
    print(header)
    print("-" * len(header))
    for label, result in rows:
        print(
            f"{label:<24} "
            f"{result.conv_tile.tc:>6} "
            f"{result.dram_traffic.num_c_tiles:>8} "
            f"{str(result.psum_in_sram):>10} "
            f"{_mb(result.dram_traffic.total_dram_bytes):>10.2f} "
            f"{result.memory_cycles:>10} "
            f"{result.total_cycles:>10}"
        )
        if not result.sram_usage.fits_sram:
            print(
                f"  note: tile SRAM use is "
                f"{result.sram_usage.total_sram_bytes / 1024:.1f}KB, "
                "so this strategy needs larger SRAM."
            )


def plot_results(rows: list[tuple[str, ConvScheduleResult]]) -> None:
    labels = [label.replace(", ", "\n").replace(". ", ".\n") for label, _ in rows]
    traffic_mb = [_mb(result.dram_traffic.total_dram_bytes) for _, result in rows]
    psum_extra_mb = [_mb(result.dram_traffic.psum_read_write_bytes) for _, result in rows]
    colors = [COLORS["blue"], COLORS["green"], COLORS["red"]]
    x = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, traffic_mb, color=colors, width=0.58)
    ax.set_title("Tc Split: Keeping vs Spilling Partial Sums")
    ax.set_ylabel("DRAM traffic (MB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    style_axes(ax)

    for bar, total, extra in zip(bars, traffic_mb, psum_extra_mb):
        label = f"{total:.2f} MB"
        if extra > 0:
            label += f"\n+{extra:.2f} MB psum"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.025,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
        )

    ax.text(
        1.98,
        traffic_mb[-1] * 0.55,
        "read + write\nintermediate psums",
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["red"],
    )

    save_figure(fig, FIG_PATH)
    print()
    print(f"Saved figure: {FIG_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    rows = evaluate_cases()
    print_results(rows)
    plot_results(rows)


if __name__ == "__main__":
    main()
