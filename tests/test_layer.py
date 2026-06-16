"""测试 layer：验证 Conv/FC 参数、输出尺寸和 Conv->GEMM 映射。"""

from tinydlp.layer import Conv2DLayer, FullyConnectedLayer


def test_layer_module_imports() -> None:
    assert Conv2DLayer is not None
    assert FullyConnectedLayer is not None


def test_conv2d_output_shape_and_macs() -> None:
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

    assert conv.output_hw() == (32, 32)
    assert conv.output_shape() == (1, 16, 32, 32)
    assert conv.macs() == 442_368


def test_conv2d_tensor_bytes() -> None:
    conv = Conv2DLayer(
        name="conv_bytes",
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

    assert conv.input_bytes() == 1 * 3 * 32 * 32
    assert conv.weight_bytes() == 16 * 3 * 3 * 3
    assert conv.output_bytes() == 1 * 16 * 32 * 32 * 4
    assert "conv_bytes" in conv.pretty_summary()


def test_fully_connected_macs_and_tensor_bytes() -> None:
    fc = FullyConnectedLayer(
        name="fc1",
        batch=2,
        in_features=128,
        out_features=10,
    )

    assert fc.macs() == 2 * 128 * 10
    assert fc.input_bytes() == 2 * 128
    assert fc.weight_bytes() == 128 * 10
    assert fc.output_bytes() == 2 * 10 * 4
    assert "fc1" in fc.pretty_summary()
