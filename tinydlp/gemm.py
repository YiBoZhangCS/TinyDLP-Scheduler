"""GEMM 形状工具：描述矩阵乘法 C = A x B 的 M/K/N 维度。"""

from __future__ import annotations

from dataclasses import dataclass


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True)
class GEMMShape:
    """矩阵乘法形状：A 是 M x K，B 是 K x N，输出 C 是 M x N。"""

    M: int
    K: int
    N: int
    name: str = ""

    def __post_init__(self) -> None:
        _validate_positive("M", self.M)
        _validate_positive("K", self.K)
        _validate_positive("N", self.N)

    def macs(self) -> int:
        """返回总 MAC 数，也就是所有输出元素需要的乘加次数。"""

        # GEMM C = A x B：
        # A 的形状是 M x K，B 的形状是 K x N，C 的形状是 M x N。
        # 每个 C 元素累加 K 个乘积，所以总 MACs = M * K * N。
        return self.M * self.K * self.N

    def output_elements(self) -> int:
        """返回输出矩阵 C 的元素个数。"""

        # C 的形状是 M x N，因此共有 M * N 个输出元素。
        return self.M * self.N

    def pretty_summary(self) -> str:
        return (
            f"GEMMShape(name={self.name}, A=({self.M}, {self.K}), "
            f"B=({self.K}, {self.N}), C=({self.M}, {self.N}), "
            f"macs={self.macs()})"
        )
