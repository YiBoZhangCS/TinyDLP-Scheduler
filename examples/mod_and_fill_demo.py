"""PPT demo 1：展示 Conv->GEMM 后的模数效应和脉动阵列 fill/drain 开销。

从项目根目录运行：

    python examples/mod_and_fill_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinydlp.compute_model import estimate_compute
from tinydlp.hardware import HardwareConfig
from tinydlp.layer import Conv2DLayer
from tinydlp.plot_style import COLORS, plt, save_figure, style_axes


FIG_PATH = PROJECT_ROOT / "figs" / "mod_and_fill_demo.png"


def build_hardware() -> HardwareConfig:
    return HardwareConfig(name="TinyDLP-16x16", array_m=16, array_n=16)


def build_cases() -> list[tuple[str, Conv2DLayer]]:
    return [
        (
            "A aligned: 32x32, Kout=16",
            Conv2DLayer(
                name="aligned_32x32_16ch",
                batch=1,
                in_channels=3,
                in_h=32,
                in_w=32,
                out_channels=16,
                kernel_h=3,
                kernel_w=3,
                stride=1,
                padding=1,
            ),
        ),
        (
            "B misaligned: 31x31, Kout=17",
            Conv2DLayer(
                name="misaligned_31x31_17ch",
                batch=1,
                in_channels=3,
                in_h=31,
                in_w=31,
                out_channels=17,
                kernel_h=3,
                kernel_w=3,
                stride=1,
                padding=1,
            ),
        ),
        (
            "C channel-aligned: 32x32, Kout=32",
            Conv2DLayer(
                name="aligned_32x32_32ch",
                batch=1,
                in_channels=3,
                in_h=32,
                in_w=32,
                out_channels=32,
                kernel_h=3,
                kernel_w=3,
                stride=1,
                padding=1,
            ),
        ),
    ]


def evaluate_cases() -> list[dict[str, object]]:
    hw = build_hardware()
    rows: list[dict[str, object]] = []
    for case_name, layer in build_cases():
        gemm = layer.to_gemm_shape()
        compute = estimate_compute(gemm, hw)
        rows.append(
            {
                "case": case_name,
                "M": gemm.M,
                "K": gemm.K,
                "N": gemm.N,
                "ideal_cycles": compute.ideal_cycles,
                "systolic_cycles": compute.systolic_cycles,
                "pe_utilization": compute.pe_utilization_systolic,
            }
        )
    return rows


def print_results(rows: list[dict[str, object]]) -> None:
    header = (
        f"{'Conv case':<34} {'GEMM M,K,N':>18} {'ideal_cycles':>14} "
        f"{'systolic_cycles':>16} {'PE utilization':>16}"
    )
    print("Table: Conv-to-GEMM modulus and systolic fill/drain")
    print("Question: why can two similar Conv layers get very different PE utilization?")
    print("Hardware: 16x16 PE array. GEMM view: M=batch*P*Q, K=C*R*S, N=Kout.")
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        shape = f"{row['M']},{row['K']},{row['N']}"
        print(
            f"{row['case']:<34} {shape:>18} "
            f"{row['ideal_cycles']:>14} "
            f"{row['systolic_cycles']:>16} "
            f"{row['pe_utilization']:>16.4f}"
        )


def plot_results(rows: list[dict[str, object]]) -> None:
    labels = [
        str(row["case"]).replace(": ", "\n").replace(", ", "\n")
        for row in rows
    ]
    x = list(range(len(labels)))
    cycles = [int(row["systolic_cycles"]) for row in rows]
    utils = [float(row["pe_utilization"]) for row in rows]

    fig, ax_cycles = plt.subplots(figsize=(9, 5))
    bars = ax_cycles.bar(
        x,
        cycles,
        width=0.58,
        color=COLORS["blue"],
        label="systolic cycles",
    )
    ax_cycles.set_title("Conv Shapes Mapped to GEMM: Modulus + Fill/Drain")
    ax_cycles.set_ylabel("systolic cycles")
    ax_cycles.set_xticks(x)
    ax_cycles.set_xticklabels(labels)
    style_axes(ax_cycles)

    ax_util = ax_cycles.twinx()
    ax_util.plot(
        x,
        utils,
        color=COLORS["orange"],
        marker="o",
        linewidth=2.6,
        label="PE utilization",
    )
    ax_util.set_ylabel("PE utilization")
    ax_util.set_ylim(0, 1.05)
    ax_util.spines["top"].set_visible(False)

    for index, (bar, row) in enumerate(zip(bars, rows)):
        ax_cycles.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"M={row['M']}\nK={row['K']}\nN={row['N']}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
        )
        ax_util.text(
            index,
            utils[index] + 0.035,
            f"{utils[index]:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLORS["orange"],
        )

    handles_1, labels_1 = ax_cycles.get_legend_handles_labels()
    handles_2, labels_2 = ax_util.get_legend_handles_labels()
    ax_cycles.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper left")

    save_figure(fig, FIG_PATH)
    print()
    print(f"Saved figure: {FIG_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    rows = evaluate_cases()
    print_results(rows)
    plot_results(rows)


if __name__ == "__main__":
    main()
