"""Demonstrate modulus effects on GEMM PE utilization.

Run from the project root:

    python examples/mod_effect_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinydlp.compute_model import ComputeResult, estimate_compute
from tinydlp.gemm import GEMMShape
from tinydlp.hardware import HardwareConfig
from tinydlp.plot_style import COLORS, plt, save_figure, style_axes


K_REDUCTION = 256
ARRAY_M = 128
ARRAY_N = 128
FIG_PATH = PROJECT_ROOT / "figs" / "mod_effect_utilization.png"


def build_hardware() -> HardwareConfig:
    return HardwareConfig(
        name="dlp_128x128",
        array_m=ARRAY_M,
        array_n=ARRAY_N,
    )


def format_result(gemm: GEMMShape, result: ComputeResult) -> str:
    return (
        f"{gemm.M:>4} {gemm.N:>4} "
        f"{result.macs:>12} "
        f"{result.ideal_cycles:>12} "
        f"{result.array_aware_cycles:>18} "
        f"{result.systolic_cycles:>16} "
        f"{result.pe_utilization_array_aware:>22.4f} "
        f"{result.pe_utilization_systolic:>22.4f}"
    )


def print_cases(hw: HardwareConfig) -> None:
    cases = [(128, 128), (129, 129), (256, 256), (257, 257)]
    header = (
        f"{'M':>4} {'N':>4} {'MACs':>12} {'ideal':>12} "
        f"{'array_aware':>18} {'systolic':>16} "
        f"{'util_array_aware':>22} {'util_systolic':>22}"
    )

    print("Modulus effect demo")
    print(f"Hardware: {hw.name}, array={hw.array_m}x{hw.array_n}, K={K_REDUCTION}")
    print()
    print(header)
    print("-" * len(header))
    for m, n in cases:
        gemm = GEMMShape(M=m, K=K_REDUCTION, N=n, name=f"gemm_{m}x{n}")
        result = estimate_compute(gemm, hw)
        print(format_result(gemm, result))


def plot_utilization(hw: HardwareConfig) -> None:
    sizes = list(range(64, 513))
    array_utils: list[float] = []
    systolic_utils: list[float] = []

    for size in sizes:
        gemm = GEMMShape(M=size, K=K_REDUCTION, N=size, name=f"gemm_{size}")
        result = estimate_compute(gemm, hw)
        array_utils.append(result.pe_utilization_array_aware)
        systolic_utils.append(result.pe_utilization_systolic)

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        sizes,
        array_utils,
        label="array-aware utilization",
        linewidth=2.4,
        color=COLORS["blue"],
    )
    ax.plot(
        sizes,
        systolic_utils,
        label="systolic utilization",
        linewidth=2.4,
        color=COLORS["orange"],
    )

    for multiple in (128, 256, 384, 512):
        ax.axvline(multiple, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(
            multiple,
            0.04,
            str(multiple),
            rotation=90,
            ha="right",
            va="bottom",
            color="gray",
            fontsize=8,
        )

    ax.set_title("PE Utilization vs GEMM Size on a 128x128 Array")
    ax.set_xlabel("matrix size, M = N = size")
    ax.set_ylabel("PE utilization")
    ax.set_xlim(64, 512)
    ax.set_ylim(0, 1.05)
    style_axes(ax, grid_axis="both")
    ax.legend()
    save_figure(fig, FIG_PATH)

    print()
    print(f"Saved figure: {FIG_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    hw = build_hardware()
    print_cases(hw)
    plot_utilization(hw)


if __name__ == "__main__":
    main()
