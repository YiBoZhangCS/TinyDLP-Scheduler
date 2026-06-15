import csv

from tinydlp.gemm import GEMMShape
from tinydlp.hardware import HardwareConfig
from tinydlp.report import CSV_FIELDS, generate_reports
from tinydlp.scheduler import evaluate_schedule
from tinydlp.tile import GEMMTile


def test_generate_reports_writes_csv_and_markdown(tmp_path) -> None:
    gemm = GEMMShape(M=16, K=16, N=16, name="toy")
    hw = HardwareConfig(name="dlp_16x16", array_m=16, array_n=16)
    tile = GEMMTile(tile_m=16, tile_k=16, tile_n=16)
    result = evaluate_schedule(gemm, hw, tile, "output_stationary")

    csv_path, md_path = generate_reports([result], output_dir=tmp_path, hw=hw)

    assert csv_path == tmp_path / "result.csv"
    assert md_path == tmp_path / "summary.md"

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows
    assert list(rows[0].keys()) == CSV_FIELDS
    assert rows[0]["layer_name"] == "toy"
    assert rows[0]["M"] == "16"
    assert rows[0]["K"] == "16"
    assert rows[0]["N"] == "16"

    summary = md_path.read_text(encoding="utf-8")
    assert "TinyDLP-Scheduler Report" in summary
    assert "硬件配置" in summary
    assert "每层结果" in summary
    assert "总 MACs" in summary
    assert "compute-bound" in summary
    assert "no-overlap" in summary
    assert "ideal-overlap" in summary
