"""SRAM tile 工具：计算 GEMM/Conv tile 的片上存储占用。"""

from __future__ import annotations

from dataclasses import dataclass

from tinydlp.hardware import HardwareConfig


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True)
class GEMMTile:
    """GEMM 矩阵乘法 C = A x B 的一个 tile。

    tile_m：一次处理的输出矩阵 C 的行数。对卷积来说，对应一组输出空间位置。
    tile_n：一次处理的输出矩阵 C 的列数。对卷积来说，对应一组输出通道。
    tile_k：reduction 维度切片。对卷积来说，对应 C_in * R * S 的一段。
    """

    tile_m: int
    tile_k: int
    tile_n: int

    def __post_init__(self) -> None:
        _validate_positive("tile_m", self.tile_m)
        _validate_positive("tile_k", self.tile_k)
        _validate_positive("tile_n", self.tile_n)


@dataclass(frozen=True)
class ConvTile:
    """Conv2D 原生循环维度上的 tile。

    这些名字沿用常见加速器 tiling 记法：

    - Tb：batch 方向切分
    - Tm：输出通道方向切分
    - Tc：输入通道方向切分
    - Tp：输出高度方向切分
    - Tq：输出宽度方向切分
    """

    tb: int
    tm: int
    tc: int
    tp: int
    tq: int

    def __post_init__(self) -> None:
        _validate_positive("tb", self.tb)
        _validate_positive("tm", self.tm)
        _validate_positive("tc", self.tc)
        _validate_positive("tp", self.tp)
        _validate_positive("tq", self.tq)

    def to_gemm_tile(self, kernel_h: int, kernel_w: int) -> GEMMTile:
        """返回该 Conv tile 对应的 GEMM tile。"""

        return conv_tile_to_gemm_tile(self, kernel_h, kernel_w)


@dataclass(frozen=True)
class ConvTileSRAMUsage:
    """一个 Conv tile 的 SRAM 占用拆解。"""

    input_h_tile: int
    input_w_tile: int
    input_bytes: int
    weight_bytes: int
    psum_bytes: int
    total_sram_bytes: int
    sram_capacity_bytes: int
    double_buffer: bool

    @property
    def fits_sram(self) -> bool:
        """返回该 tile 是否能放入给定 SRAM 容量。"""

        return self.total_sram_bytes <= self.sram_capacity_bytes


def a_tile_bytes(tile: GEMMTile, hw: HardwareConfig) -> int:
    """返回 A_tile 的字节数，形状为 tile_m x tile_k。"""

    # A_tile 存输入激活或 im2col 后展开的输入，使用 data_width_bits。
    return tile.tile_m * tile.tile_k * hw.data_bytes()


def b_tile_bytes(tile: GEMMTile, hw: HardwareConfig) -> int:
    """返回 B_tile 的字节数，形状为 tile_k x tile_n。"""

    # B_tile 存权重，使用 data_width_bits。
    return tile.tile_k * tile.tile_n * hw.data_bytes()


def c_tile_bytes(tile: GEMMTile, hw: HardwareConfig) -> int:
    """返回 C_tile / partial sum 的字节数，形状为 tile_m x tile_n。"""

    # C_tile 存输出 partial sum，所以使用 acc_width_bits，而不是 data_width_bits。
    # 例如 INT8 x INT8 通常累加到 INT32，避免 reduction 过程中溢出。
    #
    # 当完整 GEMM 的 K 维度大于 tile_k 时，调度器要多次遍历 K tile，
    # 并把结果持续累加到同一个 C_tile partial sum。
    return tile.tile_m * tile.tile_n * hw.acc_bytes()


def total_sram_bytes(tile: GEMMTile, hw: HardwareConfig) -> int:
    """返回 A/B/C 三个 tile 总共需要的 SRAM 字节数。"""

    return a_tile_bytes(tile, hw) + b_tile_bytes(tile, hw) + c_tile_bytes(tile, hw)


def is_valid_tile(tile: GEMMTile, hw: HardwareConfig) -> bool:
    """返回该 GEMM tile 是否满足 SRAM 容量约束。"""

    return total_sram_bytes(tile, hw) <= hw.sram_bytes()


def conv_tile_to_gemm_tile(
    tile: ConvTile,
    kernel_h: int,
    kernel_w: int,
) -> GEMMTile:
    """把 Conv 原生 tile 映射成等价的 GEMM tile。

    M_tile 是 tile 内输出位置数量，N_tile 是输出通道数量，
    K_tile 是 C * R * S reduction 维度上的切片。
    """

    _validate_positive("kernel_h", kernel_h)
    _validate_positive("kernel_w", kernel_w)
    return GEMMTile(
        tile_m=tile.tb * tile.tp * tile.tq,
        tile_k=tile.tc * kernel_h * kernel_w,
        tile_n=tile.tm,
    )


def conv_tile_sram_usage(
    tile: ConvTile,
    kernel_h: int,
    kernel_w: int,
    stride: int,
    act_bytes: int,
    weight_bytes_per_elem: int,
    acc_bytes: int,
    sram_capacity_bytes: int,
    double_buffer: bool = False,
) -> ConvTileSRAMUsage:
    """返回一个 Conv tile 的 SRAM 占用，并显式考虑空间 halo。

    一个 Tp x Tq 输出 tile 需要比 Tp x Tq 更大的输入窗口，因为边界输出点
    仍然要覆盖完整卷积核。设 stride 为 u，则输入窗口是：

    input_h_tile = (Tp - 1) * u + R
    input_w_tile = (Tq - 1) * u + S
    """

    for name, value in (
        ("kernel_h", kernel_h),
        ("kernel_w", kernel_w),
        ("stride", stride),
        ("act_bytes", act_bytes),
        ("weight_bytes_per_elem", weight_bytes_per_elem),
        ("acc_bytes", acc_bytes),
        ("sram_capacity_bytes", sram_capacity_bytes),
    ):
        _validate_positive(name, value)

    input_h_tile = (tile.tp - 1) * stride + kernel_h
    input_w_tile = (tile.tq - 1) * stride + kernel_w

    input_bytes = tile.tb * tile.tc * input_h_tile * input_w_tile * act_bytes
    weight_bytes = (
        tile.tm * tile.tc * kernel_h * kernel_w * weight_bytes_per_elem
    )
    psum_bytes = tile.tb * tile.tm * tile.tp * tile.tq * acc_bytes

    if double_buffer:
        total_bytes = 2 * (input_bytes + weight_bytes) + psum_bytes
    else:
        total_bytes = input_bytes + weight_bytes + psum_bytes

    return ConvTileSRAMUsage(
        input_h_tile=input_h_tile,
        input_w_tile=input_w_tile,
        input_bytes=input_bytes,
        weight_bytes=weight_bytes,
        psum_bytes=psum_bytes,
        total_sram_bytes=total_bytes,
        sram_capacity_bytes=sram_capacity_bytes,
        double_buffer=double_buffer,
    )


def conv_tile_sram_usage_for_hw(
    tile: ConvTile,
    kernel_h: int,
    kernel_w: int,
    stride: int,
    hw: HardwareConfig,
    double_buffer: bool = False,
) -> ConvTileSRAMUsage:
    """使用 HardwareConfig 中的数据位宽计算 Conv tile 的 SRAM 占用。"""

    return conv_tile_sram_usage(
        tile=tile,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride=stride,
        act_bytes=hw.data_bytes(),
        weight_bytes_per_elem=hw.data_bytes(),
        acc_bytes=hw.acc_bytes(),
        sram_capacity_bytes=hw.sram_bytes(),
        double_buffer=double_buffer,
    )
