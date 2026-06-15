"""GEMM shape utilities for TinyDLP analytical models."""

from __future__ import annotations

from dataclasses import dataclass


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True)
class GEMMShape:
    """Shape of C = A x B, where A is M x K and B is K x N."""

    M: int
    K: int
    N: int
    name: str = ""

    def __post_init__(self) -> None:
        _validate_positive("M", self.M)
        _validate_positive("K", self.K)
        _validate_positive("N", self.N)

    def macs(self) -> int:
        """Return total multiply-accumulate operations."""

        # GEMM C = A x B:
        # A has shape M x K, B has shape K x N, C has shape M x N.
        # Each C element accumulates K products, so total MACs = M * K * N.
        return self.M * self.K * self.N

    def output_elements(self) -> int:
        """Return number of elements in output matrix C."""

        # C has shape M x N, so it contains M * N output elements.
        return self.M * self.N

    def pretty_summary(self) -> str:
        return (
            f"GEMMShape(name={self.name}, A=({self.M}, {self.K}), "
            f"B=({self.K}, {self.N}), C=({self.M}, {self.N}), "
            f"macs={self.macs()})"
        )
