"""硬件配置：描述简化 DLP 的 PE 阵列、SRAM、频率和 DRAM 带宽。"""

from __future__ import annotations

from dataclasses import dataclass


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_positive_float(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True)
class HardwareConfig:
    """简化 DLP 硬件参数。"""

    name: str
    array_m: int
    array_n: int
    mac_per_pe_per_cycle: int = 1
    frequency_mhz: float = 500.0
    sram_kb: int = 64
    dram_bandwidth_gb_s: float = 12.8
    data_width_bits: int = 8
    acc_width_bits: int = 32

    def __post_init__(self) -> None:
        for field_name in (
            "array_m",
            "array_n",
            "mac_per_pe_per_cycle",
            "sram_kb",
            "data_width_bits",
            "acc_width_bits",
        ):
            _validate_positive_int(field_name, getattr(self, field_name))
        _validate_positive_float("frequency_mhz", self.frequency_mhz)
        _validate_positive_float("dram_bandwidth_gb_s", self.dram_bandwidth_gb_s)

    def num_pe(self) -> int:
        """返回二维 PE 阵列中的处理单元数量。"""

        # 阵列有 array_m 行和 array_n 列，所以 PE 数量是二者相乘。
        return self.array_m * self.array_n

    def peak_macs_per_cycle(self) -> int:
        """返回每周期理论峰值 MAC 吞吐。"""

        # 每个 PE 每周期可以发起 mac_per_pe_per_cycle 次 MAC。
        # 峰值 MACs/cycle = PE 数量 * 每个 PE 每周期 MAC 数。
        return self.num_pe() * self.mac_per_pe_per_cycle

    def peak_gmacs_per_second(self) -> float:
        """返回以 GMAC/s 表示的理论峰值吞吐。"""

        # frequency_mhz 表示每秒百万周期。
        # GMAC/s = MACs/cycle * frequency_mhz * 1e6 / 1e9
        #        = MACs/cycle * frequency_mhz / 1000。
        return self.peak_macs_per_cycle() * self.frequency_mhz / 1000

    def dram_bytes_per_cycle(self) -> float:
        """返回折算到每周期的片外 DRAM 带宽，单位是 bytes/cycle。"""

        # 这里 dram_bandwidth_gb_s 使用十进制 GB/s，
        # bytes/s = GB/s * 1e9，cycles/s = frequency_mhz * 1e6。
        # bytes/cycle = bytes/s / cycles/s。
        return self.dram_bandwidth_gb_s * 1e9 / (self.frequency_mhz * 1e6)

    def sram_bytes(self) -> int:
        """返回片上 SRAM 容量，单位是字节。"""

        return self.sram_kb * 1024

    def data_bytes(self) -> int:
        """返回输入激活和权重标量的存储字节数。"""

        return self.data_width_bits // 8

    def acc_bytes(self) -> int:
        """返回累加器/partial sum 标量的存储字节数。"""

        return self.acc_width_bits // 8

    def pretty_summary(self) -> str:
        return (
            f"HardwareConfig(name={self.name}, array={self.array_m}x{self.array_n}, "
            f"num_pe={self.num_pe()}, peak_macs_per_cycle={self.peak_macs_per_cycle()}, "
            f"peak_gmacs_per_second={self.peak_gmacs_per_second()}, "
            f"sram_bytes={self.sram_bytes()}, "
            f"dram_bytes_per_cycle={self.dram_bytes_per_cycle()})"
        )
