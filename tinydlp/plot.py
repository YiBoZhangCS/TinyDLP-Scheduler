"""Matplotlib plots generated from TinyDLP CSV reports."""

from __future__ import annotations

import csv
from pathlib import Path

from tinydlp.plot_style import COLORS, plt, save_figure, style_axes


def _read_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"CSV report has no data rows: {csv_path}")
    return rows


def _int_column(rows: list[dict[str, str]], name: str) -> list[int]:
    return [int(float(row[name])) for row in rows]


def _float_column(rows: list[dict[str, str]], name: str) -> list[float]:
    return [float(row[name]) for row in rows]


def _finish_bar_plot(
    fig,
    ax,
    output_path: Path,
    title: str,
    ylabel: str,
) -> Path:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    style_axes(ax)
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    return save_figure(fig, output_path)


def plot_layer_latency(rows: list[dict[str, str]], output_dir: Path) -> Path:
    labels = [row["layer_name"] for row in rows]
    values = _int_column(rows, "ideal_overlap_cycles")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=COLORS["blue"])
    return _finish_bar_plot(
        fig,
        ax,
        output_dir / "layer_latency.png",
        "Layer Latency",
        "ideal-overlap cycles",
    )


def plot_compute_vs_memory(rows: list[dict[str, str]], output_dir: Path) -> Path:
    labels = [row["layer_name"] for row in rows]
    compute = _int_column(rows, "systolic_compute_cycles")
    memory = _int_column(rows, "memory_cycles")
    positions = list(range(len(labels)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        [pos - width / 2 for pos in positions],
        compute,
        width=width,
        label="compute",
        color=COLORS["blue"],
    )
    ax.bar(
        [pos + width / 2 for pos in positions],
        memory,
        width=width,
        label="memory",
        color=COLORS["orange"],
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.legend()
    return _finish_bar_plot(
        fig,
        ax,
        output_dir / "compute_vs_memory.png",
        "Compute vs Memory Cycles",
        "cycles",
    )


def plot_pe_utilization(rows: list[dict[str, str]], output_dir: Path) -> Path:
    labels = [row["layer_name"] for row in rows]
    values = _float_column(rows, "pe_utilization")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=COLORS["green"])
    ax.set_ylim(0, 1.05)
    return _finish_bar_plot(
        fig,
        ax,
        output_dir / "pe_utilization.png",
        "PE Utilization",
        "utilization",
    )


def plot_dram_traffic(rows: list[dict[str, str]], output_dir: Path) -> Path:
    labels = [row["layer_name"] for row in rows]
    values = _int_column(rows, "dram_bytes")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=COLORS["red"])
    return _finish_bar_plot(
        fig,
        ax,
        output_dir / "dram_traffic.png",
        "DRAM Traffic",
        "bytes",
    )


def generate_plots(
    csv_path: str | Path = "reports/result.csv",
    output_dir: str | Path = "figs",
) -> list[Path]:
    """Generate all standard plots from result.csv."""

    rows = _read_rows(csv_path)
    fig_dir = Path(output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    return [
        plot_layer_latency(rows, fig_dir),
        plot_compute_vs_memory(rows, fig_dir),
        plot_pe_utilization(rows, fig_dir),
        plot_dram_traffic(rows, fig_dir),
    ]
