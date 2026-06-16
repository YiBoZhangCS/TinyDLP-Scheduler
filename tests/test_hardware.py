"""测试 hardware：验证 PE 数、峰值吞吐、SRAM 和 DRAM 带宽换算。"""

from tinydlp.hardware import HardwareConfig


def test_hardware_peak_throughput() -> None:
    hw = HardwareConfig(
        name="dlp_16x16",
        array_m=16,
        array_n=16,
        frequency_mhz=500.0,
    )

    assert hw.num_pe() == 256
    assert hw.peak_macs_per_cycle() == 256
    assert hw.peak_gmacs_per_second() == 128.0


def test_hardware_memory_and_summary() -> None:
    hw = HardwareConfig(
        name="dlp_16x16",
        array_m=16,
        array_n=16,
        sram_kb=64,
        dram_bandwidth_gb_s=12.8,
        data_width_bits=8,
        acc_width_bits=32,
    )

    assert hw.sram_bytes() == 64 * 1024
    assert hw.dram_bytes_per_cycle() == 25.6
    assert hw.data_bytes() == 1
    assert hw.acc_bytes() == 4
    assert "dlp_16x16" in hw.pretty_summary()
