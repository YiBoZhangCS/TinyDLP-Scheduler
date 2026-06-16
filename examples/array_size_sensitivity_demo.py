"""阵列尺寸敏感性 demo：固定 GEMM，观察不同 PE 阵列规模下的利用率。

从项目根目录运行：

    python examples/array_size_sensitivity_demo.py
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


ARRAY_SIZES = [4, 8, 16, 32, 64]
FIG_PATH = PROJECT_ROOT / "figs" / "array_size_sensitivity.png"


def build_gemm() -> GEMMShape:
    # 这个 GEMM 对应 Conv：输入 1x3x32x32，输出通道 16，卷积核 3x3。
    # 映射后 M = batch * P * Q = 1024，K = C * R * S = 27，N = Kout = 16。
    return GEMMShape(M=1024, K=27, N=16, name="conv1_gemm")


def build_hardware(array_size: int) -> HardwareConfig:
    return HardwareConfig(
        name=f"TinyDLP-{array_size}x{array_size}",
        array_m=array_size,
        array_n=array_size,
        frequency_mhz=500.0,
        dram_bandwidth_gb_s=12.8,
    )


def evaluate_array_sweep() -> list[tuple[int, ComputeResult]]:
    gemm = build_gemm()
    results: list[tuple[int, ComputeResult]] = []

    for array_size in ARRAY_SIZES:
        hw = build_hardware(array_size)
        result = estimate_compute(gemm, hw)
        results.append((array_size, result))

    return results


def print_results(results: list[tuple[int, ComputeResult]]) -> None:
    header = (
        f"{'array':>8} {'ideal_cycles':>14} "
        f"{'systolic_cycles':>16} {'PE utilization':>16}"
    )

    print("Array size sensitivity demo")
    print("Fixed GEMM: M=1024, K=27, N=16")
    print()
    print(header)
    print("-" * len(header))
    for array_size, result in results:
        print(
            f"{array_size}x{array_size: <5} "
            f"{result.ideal_cycles:>14} "
            f"{result.systolic_cycles:>16} "
            f"{result.pe_utilization_systolic:>16.4f}"
        )


def plot_results(results: list[tuple[int, ComputeResult]]) -> None:
    labels = [f"{array_size}x{array_size}" for array_size, _ in results]
    utilizations = [result.pe_utilization_systolic for _, result in results]

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, utilizations, color=COLORS["green"])
    ax.set_title("Array Size Sensitivity")
    ax.set_xlabel("array size")
    ax.set_ylabel("PE utilization")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", labelrotation=35)
    style_axes(ax)
    save_figure(fig, FIG_PATH)

    print()
    print(f"Saved figure: {FIG_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    results = evaluate_array_sweep()
    print_results(results)
    plot_results(results)


if __name__ == "__main__":
    main()
