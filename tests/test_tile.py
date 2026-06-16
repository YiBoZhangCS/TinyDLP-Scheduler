"""测试 tile：验证 GEMM/Conv tile 的 SRAM 占用和 halo 计算。"""

from tinydlp.hardware import HardwareConfig
from tinydlp.gemm import GEMMShape
from tinydlp.scheduler import enumerate_gemm_tiles
from tinydlp.tile import (
    ConvTile,
    GEMMTile,
    a_tile_bytes,
    b_tile_bytes,
    c_tile_bytes,
    conv_tile_sram_usage,
    conv_tile_to_gemm_tile,
    is_valid_tile,
    total_sram_bytes,
)


def test_tile_module_imports() -> None:
    assert GEMMTile is not None


def test_gemm_tile_sram_bytes() -> None:
    hw = HardwareConfig(name="dlp_16x16", array_m=16, array_n=16, sram_kb=64)
    tile = GEMMTile(tile_m=16, tile_k=32, tile_n=16)

    assert a_tile_bytes(tile, hw) == 16 * 32
    assert b_tile_bytes(tile, hw) == 32 * 16
    assert c_tile_bytes(tile, hw) == 16 * 16 * 4
    assert total_sram_bytes(tile, hw) == 16 * 32 + 32 * 16 + 16 * 16 * 4
    assert is_valid_tile(tile, hw)


def test_gemm_tile_can_exceed_sram_capacity() -> None:
    hw = HardwareConfig(name="tiny_sram", array_m=16, array_n=16, sram_kb=1)
    tile = GEMMTile(tile_m=64, tile_k=64, tile_n=64)

    assert not is_valid_tile(tile, hw)


def test_enumerate_gemm_tiles_returns_only_sram_valid_tiles() -> None:
    gemm = GEMMShape(M=64, K=32, N=64, name="small_gemm")
    hw = HardwareConfig(name="dlp_16x16", array_m=16, array_n=16, sram_kb=64)

    tiles = enumerate_gemm_tiles(gemm, hw)
    sram_sizes = [total_sram_bytes(tile, hw) for tile in tiles]

    assert tiles
    assert sram_sizes == sorted(sram_sizes)
    assert all(tile.tile_m <= gemm.M for tile in tiles)
    assert all(tile.tile_k <= gemm.K for tile in tiles)
    assert all(tile.tile_n <= gemm.N for tile in tiles)
    assert all(is_valid_tile(tile, hw) for tile in tiles)


def test_conv_tile_maps_to_gemm_tile() -> None:
    tile = ConvTile(tb=1, tm=32, tc=16, tp=8, tq=4)

    gemm_tile = conv_tile_to_gemm_tile(tile, kernel_h=3, kernel_w=3)

    assert gemm_tile.tile_m == 1 * 8 * 4
    assert gemm_tile.tile_n == 32
    assert gemm_tile.tile_k == 16 * 3 * 3


def test_conv_tile_sram_usage_includes_halo_and_double_buffer() -> None:
    tile = ConvTile(tb=1, tm=16, tc=8, tp=4, tq=4)

    usage = conv_tile_sram_usage(
        tile=tile,
        kernel_h=3,
        kernel_w=3,
        stride=1,
        act_bytes=1,
        weight_bytes_per_elem=1,
        acc_bytes=4,
        sram_capacity_bytes=8 * 1024,
        double_buffer=True,
    )

    assert usage.input_h_tile == 6
    assert usage.input_w_tile == 6
    assert usage.input_bytes == 1 * 8 * 6 * 6
    assert usage.weight_bytes == 16 * 8 * 3 * 3
    assert usage.psum_bytes == 1 * 16 * 4 * 4 * 4
    assert usage.total_sram_bytes == (
        2 * (usage.input_bytes + usage.weight_bytes) + usage.psum_bytes
    )
    assert usage.fits_sram
