"""测试 plot：验证 result.csv 可以生成标准图表文件。"""

from tinydlp.gemm import GEMMShape
from tinydlp.hardware import HardwareConfig
from tinydlp.plot import generate_plots
from tinydlp.report import generate_reports
from tinydlp.scheduler import evaluate_schedule
from tinydlp.tile import GEMMTile


def test_generate_plots_from_result_csv(tmp_path) -> None:
    gemm = GEMMShape(M=16, K=16, N=16, name="toy")
    hw = HardwareConfig(name="dlp_16x16", array_m=16, array_n=16)
    tile = GEMMTile(tile_m=16, tile_k=16, tile_n=16)
    result = evaluate_schedule(gemm, hw, tile, "output_stationary")
    csv_path, _ = generate_reports([result], output_dir=tmp_path / "reports", hw=hw)

    paths = generate_plots(csv_path=csv_path, output_dir=tmp_path / "figs")

    assert [path.name for path in paths] == [
        "layer_latency.png",
        "compute_vs_memory.png",
        "pe_utilization.png",
        "dram_traffic.png",
    ]
    assert all(path.exists() for path in paths)
    assert all(path.stat().st_size > 0 for path in paths)
