"""Hardware configuration for simplified TinyDLP analytical models."""

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
    """Simplified DLP hardware parameters."""

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
        """Return number of processing elements in the 2D array."""

        # The array has array_m rows and array_n columns, so PE count is M * N.
        return self.array_m * self.array_n

    def peak_macs_per_cycle(self) -> int:
        """Return peak MAC throughput per cycle."""

        # Every PE can issue mac_per_pe_per_cycle MACs each cycle.
        # Peak MACs/cycle = number of PEs * MACs per PE per cycle.
        return self.num_pe() * self.mac_per_pe_per_cycle

    def peak_gmacs_per_second(self) -> float:
        """Return peak throughput in GMAC/s."""

        # frequency_mhz means million cycles/second.
        # GMAC/s = MACs/cycle * frequency_mhz * 1e6 / 1e9
        #        = MACs/cycle * frequency_mhz / 1000.
        return self.peak_macs_per_cycle() * self.frequency_mhz / 1000

    def dram_bytes_per_cycle(self) -> float:
        """Return off-chip DRAM bandwidth normalized to bytes per cycle."""

        # dram_bandwidth_gb_s is decimal GB/s here, so bytes/s = GB/s * 1e9.
        # cycles/s = frequency_mhz * 1e6.
        # bytes/cycle = bytes/s / cycles/s.
        return self.dram_bandwidth_gb_s * 1e9 / (self.frequency_mhz * 1e6)

    def sram_bytes(self) -> int:
        """Return on-chip SRAM capacity in bytes."""

        return self.sram_kb * 1024

    def data_bytes(self) -> int:
        """Return storage bytes for input/weight data scalars."""

        return self.data_width_bits // 8

    def acc_bytes(self) -> int:
        """Return storage bytes for accumulator scalars."""

        return self.acc_width_bits // 8

    def pretty_summary(self) -> str:
        return (
            f"HardwareConfig(name={self.name}, array={self.array_m}x{self.array_n}, "
            f"num_pe={self.num_pe()}, peak_macs_per_cycle={self.peak_macs_per_cycle()}, "
            f"peak_gmacs_per_second={self.peak_gmacs_per_second()}, "
            f"sram_bytes={self.sram_bytes()}, "
            f"dram_bytes_per_cycle={self.dram_bytes_per_cycle()})"
        )
