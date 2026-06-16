"""测试 gemm：验证 GEMM 形状、MAC 数和输出元素统计。"""

from tinydlp.gemm import GEMMShape
from tinydlp.layer import Conv2DLayer, FullyConnectedLayer


def test_gemm_module_imports() -> None:
    assert GEMMShape is not None


def test_gemm_shape_macs_and_output_elements() -> None:
    shape = GEMMShape(M=4, K=8, N=16, name="toy")

    assert shape.macs() == 4 * 8 * 16
    assert shape.output_elements() == 4 * 16
    assert "toy" in shape.pretty_summary()


def test_conv2d_to_gemm_shape() -> None:
    conv = Conv2DLayer(
        name="conv1",
        batch=1,
        in_channels=3,
        in_h=32,
        in_w=32,
        out_channels=16,
        kernel_h=3,
        kernel_w=3,
        stride=1,
        padding=1,
    )

    shape = conv.to_gemm_shape()

    assert shape == GEMMShape(M=1024, K=27, N=16, name="conv1")
    assert shape.macs() == conv.macs()


def test_fully_connected_to_gemm_shape() -> None:
    fc = FullyConnectedLayer(
        name="fc1",
        batch=2,
        in_features=128,
        out_features=10,
    )

    shape = fc.to_gemm_shape()

    assert shape == GEMMShape(M=2, K=128, N=10, name="fc1")
    assert shape.macs() == fc.macs()
