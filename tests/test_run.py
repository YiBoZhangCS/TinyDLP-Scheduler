"""测试 run.py：验证 JSON 加载、模型运行和报告/图表生成。"""

from pathlib import Path

from run import load_hardware, load_model_layers, run_model
from tinydlp.layer import Conv2DLayer, FullyConnectedLayer


def test_load_hardware_from_json() -> None:
    hw = load_hardware(Path("examples/dlp_16x16.json"))

    assert hw.name == "TinyDLP-16x16"
    assert hw.array_m == 16
    assert hw.array_n == 16
    assert hw.sram_kb == 64


def test_load_model_layers_from_json() -> None:
    layers = load_model_layers(Path("examples/lenet.json"))

    assert len(layers) == 3
    assert isinstance(layers[0], Conv2DLayer)
    assert isinstance(layers[1], Conv2DLayer)
    assert isinstance(layers[2], FullyConnectedLayer)
    assert layers[0].to_gemm_shape().M == 1024
    assert layers[0].to_gemm_shape().K == 27
    assert layers[0].to_gemm_shape().N == 16
    assert layers[1].kernel_h == 5
    assert layers[1].kernel_w == 5
    assert layers[1].padding == 2
    assert layers[1].to_gemm_shape().K == 16 * 5 * 5


def test_run_model_returns_one_schedule_per_layer(capsys, tmp_path) -> None:
    results = run_model(
        model_path=Path("examples/lenet.json"),
        hw_path=Path("examples/dlp_16x16.json"),
        overlap="ideal",
        output_dir=tmp_path / "reports",
        plot=True,
        fig_dir=tmp_path / "figs",
    )
    printed = capsys.readouterr().out

    assert len(results) == 3
    assert all(result.macs > 0 for result in results)
    assert "Layer: conv1" in printed
    assert "Layer: conv2" in printed
    assert "Layer: fc1" in printed
    assert "Network summary" in printed
    assert "Reports" in printed
    assert "Figures" in printed
    assert (tmp_path / "reports" / "result.csv").exists()
    assert (tmp_path / "reports" / "summary.md").exists()
    assert (tmp_path / "figs" / "layer_latency.png").exists()
