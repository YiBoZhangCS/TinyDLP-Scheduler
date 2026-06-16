"""命令行入口：读取模型/硬件 JSON，运行调度搜索，并生成报告和图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tinydlp.hardware import HardwareConfig
from tinydlp.layer import Conv2DLayer, FullyConnectedLayer
from tinydlp.plot import generate_plots
from tinydlp.report import generate_reports
from tinydlp.scheduler import (
    ScheduleResult,
    pretty_print_schedule,
    search_best_schedule,
)

Layer = Conv2DLayer | FullyConnectedLayer


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def load_hardware(path: str | Path) -> HardwareConfig:
    """从硬件 JSON 文件读取 HardwareConfig。"""

    data = _read_json(path)
    return HardwareConfig(
        name=str(data["name"]),
        array_m=int(data["array_m"]),
        array_n=int(data["array_n"]),
        mac_per_pe_per_cycle=int(data.get("mac_per_pe_per_cycle", 1)),
        frequency_mhz=float(data.get("frequency_mhz", 500.0)),
        sram_kb=int(data.get("sram_kb", 64)),
        dram_bandwidth_gb_s=float(data.get("dram_bandwidth_gb_s", 12.8)),
        data_width_bits=int(data.get("data_width_bits", 8)),
        acc_width_bits=int(data.get("acc_width_bits", 32)),
    )


def _get_int(data: dict[str, Any], *names: str) -> int:
    for name in names:
        if name in data:
            return int(data[name])
    joined = " or ".join(names)
    raise ValueError(f"missing required field: {joined}")


def _load_layer(data: dict[str, Any]) -> Layer:
    layer_type = str(data.get("type", "")).lower()
    if layer_type in {"conv", "conv2d"}:
        return Conv2DLayer(
            name=str(data["name"]),
            batch=_get_int(data, "batch"),
            in_channels=_get_int(data, "in_channels", "C"),
            in_h=_get_int(data, "in_h", "H"),
            in_w=_get_int(data, "in_w", "W"),
            out_channels=_get_int(data, "out_channels", "K"),
            kernel_h=_get_int(data, "kernel_h", "R"),
            kernel_w=_get_int(data, "kernel_w", "S"),
            stride=_get_int(data, "stride"),
            padding=_get_int(data, "padding"),
            data_width_bits=int(data.get("data_width_bits", 8)),
            acc_width_bits=int(data.get("acc_width_bits", 32)),
        )
    if layer_type in {"fc", "fully_connected", "linear"}:
        return FullyConnectedLayer(
            name=str(data["name"]),
            batch=_get_int(data, "batch"),
            in_features=_get_int(data, "in_features"),
            out_features=_get_int(data, "out_features"),
            data_width_bits=int(data.get("data_width_bits", 8)),
            acc_width_bits=int(data.get("acc_width_bits", 32)),
        )
    raise ValueError(f"unknown layer type {layer_type!r} for layer {data.get('name')!r}")


def load_model_layers(path: str | Path) -> list[Layer]:
    """从模型 JSON 文件读取 Conv2D 和全连接层。"""

    data = _read_json(path)
    raw_layers = data.get("layers")
    if not isinstance(raw_layers, list):
        raise ValueError("model JSON must contain a 'layers' list")
    return [_load_layer(layer) for layer in raw_layers]


def run_model(
    model_path: str | Path,
    hw_path: str | Path,
    overlap: str,
    output_dir: str | Path = "reports",
    plot: bool = False,
    fig_dir: str | Path = "figs",
) -> list[ScheduleResult]:
    """评估模型 JSON 中的每一层，并生成可选报告/图表。"""

    hw = load_hardware(hw_path)
    layers = load_model_layers(model_path)
    results: list[ScheduleResult] = []

    print("TinyDLP-Scheduler")
    print(f"Hardware: {hw.name}")
    print(f"Model: {Path(model_path)}")
    print(f"Overlap mode: {overlap}")

    for layer in layers:
        gemm = layer.to_gemm_shape()
        result = search_best_schedule(gemm=gemm, hw=hw, overlap_mode=overlap)
        results.append(result)
        print()
        print(f"Layer: {layer.name}")
        pretty_print_schedule(result)

    total_macs = sum(result.macs for result in results)
    total_dram_bytes = sum(result.dram_bytes for result in results)
    total_no_overlap_cycles = sum(result.no_overlap_cycles for result in results)
    total_ideal_overlap_cycles = sum(result.ideal_overlap_cycles for result in results)

    print()
    print("Network summary")
    print(f"  total MACs: {total_macs}")
    print(f"  total DRAM bytes: {total_dram_bytes}")
    print(f"  total no-overlap cycles: {total_no_overlap_cycles}")
    print(f"  total ideal-overlap cycles: {total_ideal_overlap_cycles}")

    csv_path, md_path = generate_reports(
        results,
        output_dir=output_dir,
        hw=hw,
        layers=layers,
    )
    print()
    print("Reports")
    print(f"  CSV: {csv_path}")
    print(f"  Markdown: {md_path}")

    if plot:
        plot_paths = generate_plots(csv_path=csv_path, output_dir=fig_dir)
        print()
        print("Figures")
        for path in plot_paths:
            print(f"  {path}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TinyDLP-Scheduler analytical runner")
    parser.add_argument("--model", help="Path to model JSON, for example examples/lenet.json")
    parser.add_argument("--hw", help="Path to hardware JSON, for example examples/dlp_16x16.json")
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for generated result.csv and summary.md",
    )
    parser.add_argument(
        "--fig-dir",
        default="figs",
        help="Directory for generated plot PNG files",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots from the CSV report",
    )
    parser.add_argument(
        "--overlap",
        choices=("ideal", "none"),
        default="ideal",
        help="Schedule objective: ideal overlap or no overlap",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model is None and args.hw is None:
        print("TinyDLP-Scheduler")
        print("Analytical Conv/GEMM mapping and bottleneck exploration for a simplified DLP.")
        print("Try: python run.py --model examples/lenet.json --hw examples/dlp_16x16.json")
        return
    if args.model is None or args.hw is None:
        raise SystemExit("--model and --hw must be provided together")

    run_model(
        model_path=args.model,
        hw_path=args.hw,
        overlap=args.overlap,
        output_dir=args.output_dir,
        plot=args.plot,
        fig_dir=args.fig_dir,
    )


if __name__ == "__main__":
    main()
