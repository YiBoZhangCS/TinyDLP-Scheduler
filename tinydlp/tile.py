"""SRAM tile helpers for GEMM and Conv-native mappings."""

from __future__ import annotations

from dataclasses import dataclass

from tinydlp.hardware import HardwareConfig


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True)
class GEMMTile:
    """A GEMM tile over C = A x B.

    tile_m: rows of output matrix C processed at once. For convolution, this
        corresponds to a group of output spatial positions.
    tile_n: columns of output matrix C processed at once. For convolution, this
        corresponds to a group of output channels / filters.
    tile_k: reduction dimension tile. For convolution, this corresponds to a
        slice of C_in * R * S.
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
    """A Conv2D tile in NCHW-style loop dimensions.

    The public names follow the common accelerator tiling notation:

    - Tb: batch tile
    - Tm: output-channel tile
    - Tc: input-channel tile
    - Tp: output-height tile
    - Tq: output-width tile
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
        """Return the equivalent GEMM tile for this convolution tile."""

        return conv_tile_to_gemm_tile(self, kernel_h, kernel_w)


@dataclass(frozen=True)
class ConvTileSRAMUsage:
    """SRAM breakdown for one Conv tile."""

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
        """Return whether this tile fits in the configured SRAM capacity."""

        return self.total_sram_bytes <= self.sram_capacity_bytes


def a_tile_bytes(tile: GEMMTile, hw: HardwareConfig) -> int:
    """Return bytes for A_tile with shape tile_m x tile_k."""

    # A_tile stores input activations/unfolded inputs and uses data_width_bits.
    return tile.tile_m * tile.tile_k * hw.data_bytes()


def b_tile_bytes(tile: GEMMTile, hw: HardwareConfig) -> int:
    """Return bytes for B_tile with shape tile_k x tile_n."""

    # B_tile stores weights and uses data_width_bits.
    return tile.tile_k * tile.tile_n * hw.data_bytes()


def c_tile_bytes(tile: GEMMTile, hw: HardwareConfig) -> int:
    """Return bytes for C_tile / partial sums with shape tile_m x tile_n."""

    # C_tile stores output partial sums, so it uses acc_width_bits rather than
    # data_width_bits. For example, INT8 x INT8 products are commonly accumulated
    # into INT32 partial sums to avoid overflow during the reduction.
    #
    # When the full GEMM K dimension is larger than tile_k, the scheduler must
    # visit the K dimension multiple times and keep accumulating into the same
    # C_tile partial sums.
    return tile.tile_m * tile.tile_n * hw.acc_bytes()


def total_sram_bytes(tile: GEMMTile, hw: HardwareConfig) -> int:
    """Return total SRAM bytes needed by A, B, and C tiles."""

    return a_tile_bytes(tile, hw) + b_tile_bytes(tile, hw) + c_tile_bytes(tile, hw)


def is_valid_tile(tile: GEMMTile, hw: HardwareConfig) -> bool:
    """Return whether the tile fits in the configured SRAM capacity."""

    return total_sram_bytes(tile, hw) <= hw.sram_bytes()


def conv_tile_to_gemm_tile(
    tile: ConvTile,
    kernel_h: int,
    kernel_w: int,
) -> GEMMTile:
    """Map a Conv-native tile to the equivalent GEMM tile.

    M_tile is the number of output positions in the tile, N_tile is the number
    of output channels, and K_tile is the reduction slice over C * R * S.
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
    """Return SRAM usage for one Conv tile, including spatial halo.

    A Tp x Tq output tile needs a larger input window because neighboring
    output points share kernel footprints. With stride `u`, the input window is:

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
    """Return Conv tile SRAM usage using byte widths from a HardwareConfig."""

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
