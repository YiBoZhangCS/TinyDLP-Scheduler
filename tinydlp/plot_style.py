"""统一绘图风格：让所有 TinyDLP 图表适合 README 和 PPT 展示。"""

from __future__ import annotations

import os
from pathlib import Path

MPL_CONFIG_DIR = Path("/tmp/tinydlp-matplotlib")
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#54A24B",
    "red": "#E45756",
    "gray": "#7F7F7F",
}

DATAFLOW_COLORS = {
    "output_stationary": COLORS["blue"],
    "weight_stationary": COLORS["orange"],
    "input_stationary": COLORS["green"],
}

DPI = 300


def apply_plot_style() -> None:
    """应用白底、清晰字体、色盲友好配色的 matplotlib 风格。"""

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.size": 11,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "savefig.facecolor": "white",
        }
    )


def style_axes(ax, grid_axis: str = "y") -> None:
    """去掉多余边框并添加浅色网格。"""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis)
        ax.set_axisbelow(True)


def save_figure(fig, path: str | Path) -> Path:
    """以固定 300 dpi 和紧凑布局保存图表。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


apply_plot_style()
