"""层定义：保存 Conv/FC 原始参数，并负责把算子映射到 GEMM 形状。"""

from __future__ import annotations

from dataclasses import dataclass

from tinydlp.gemm import GEMMShape


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _bits_to_storage_bytes(bits: int) -> int:
    if bits <= 0:
        raise ValueError("bit width must be positive")
    # 每个标量需要的存储字节数。这里向上取整，所以 int4 等亚字节格式
    # 也能用字节寻址内存做保守估算。
    return _ceil_div(bits, 8)


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True)
class Conv2DLayer:
    """NCHW 布局的 2D 卷积层。

    batch 对应输入 N，in_channels 对应输入 C，in_h/in_w 对应 H/W，
    out_channels 对应 Kout，kernel_h/kernel_w 对应 R/S。
    """

    name: str
    batch: int
    in_channels: int
    in_h: int
    in_w: int
    out_channels: int
    kernel_h: int
    kernel_w: int
    stride: int
    padding: int
    data_width_bits: int = 8
    acc_width_bits: int = 32

    def __post_init__(self) -> None:
        for field_name in (
            "batch",
            "in_channels",
            "in_h",
            "in_w",
            "out_channels",
            "kernel_h",
            "kernel_w",
            "stride",
        ):
            _validate_positive(field_name, getattr(self, field_name))
        if self.padding < 0:
            raise ValueError(f"padding must be non-negative, got {self.padding}")
        _bits_to_storage_bytes(self.data_width_bits)
        _bits_to_storage_bytes(self.acc_width_bits)
        self.output_hw()

    def output_hw(self) -> tuple[int, int]:
        """返回输出特征图的空间尺寸 P/Q。"""

        # 卷积输出尺寸公式：
        # P = floor((H + 2 * padding - R) / stride) + 1。
        # Q = floor((W + 2 * padding - S) / stride) + 1。
        # H/W 是输入高宽，R/S 是卷积核高宽。
        out_h = (self.in_h + 2 * self.padding - self.kernel_h) // self.stride + 1
        out_w = (self.in_w + 2 * self.padding - self.kernel_w) // self.stride + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                "convolution output spatial shape must be positive, "
                f"got ({out_h}, {out_w})"
            )
        return out_h, out_w

    def output_shape(self) -> tuple[int, int, int, int]:
        """返回 NCHW 布局下的输出张量形状。"""

        out_h, out_w = self.output_hw()
        return self.batch, self.out_channels, out_h, out_w

    def macs(self) -> int:
        """返回该卷积层的总 MAC 数。"""

        out_h, out_w = self.output_hw()
        # 每个输出元素需要累加 C * R * S 个乘积。
        # 总 MACs = N * Kout * P * Q * C * R * S，其中：
        # N=batch，Kout=out_channels，P/Q=输出高宽，
        # C=in_channels，R/S=卷积核高宽。
        return (
            self.batch
            * self.out_channels
            * out_h
            * out_w
            * self.in_channels
            * self.kernel_h
            * self.kernel_w
        )

    def input_bytes(self) -> int:
        """返回存储输入 feature map 需要的字节数。"""

        # 输入激活按 data_width_bits 存储，例如 int8 对应 1 字节。
        num_elements = self.batch * self.in_channels * self.in_h * self.in_w
        return num_elements * _bits_to_storage_bytes(self.data_width_bits)

    def weight_bytes(self) -> int:
        """返回存储卷积核权重需要的字节数。"""

        # 权重按 data_width_bits 存储，形状是 Kout x C x R x S。
        num_elements = (
            self.out_channels * self.in_channels * self.kernel_h * self.kernel_w
        )
        return num_elements * _bits_to_storage_bytes(self.data_width_bits)

    def output_bytes(self) -> int:
        """返回存储输出/partial sum 张量需要的字节数。"""

        out_h, out_w = self.output_hw()
        # 本模型把输出按累加器精度统计，因为这里关心 MAC 之后的
        # partial sum / full sum，而不是后续量化后的低精度输出。
        num_elements = self.batch * self.out_channels * out_h * out_w
        return num_elements * _bits_to_storage_bytes(self.acc_width_bits)

    def to_gemm_shape(self) -> GEMMShape:
        """返回该卷积层对应的 GEMM 形状。"""

        out_h, out_w = self.output_hw()
        # Conv-to-GEMM 映射，也就是 im2col 之后的矩阵乘法视角：
        # A 是展开后的输入矩阵，形状为 M x K。
        #   M = batch * P * Q，每一行对应一个输出空间位置。
        #   K = in_channels * kernel_h * kernel_w，对应 C * R * S。
        # B 是权重矩阵，形状为 K x N。
        #   N = out_channels，也就是 Kout，每一列对应一个输出通道。
        # C 是输出矩阵，形状为 M x N。
        return GEMMShape(
            M=self.batch * out_h * out_w,
            K=self.in_channels * self.kernel_h * self.kernel_w,
            N=self.out_channels,
            name=self.name,
        )

    def pretty_summary(self) -> str:
        out_shape = self.output_shape()
        return (
            f"Conv2DLayer(name={self.name}, output_shape={out_shape}, "
            f"macs={self.macs()}, input_bytes={self.input_bytes()}, "
            f"weight_bytes={self.weight_bytes()}, output_bytes={self.output_bytes()})"
        )


@dataclass(frozen=True)
class FullyConnectedLayer:
    """全连接层：用一个批量矩阵乘法表示。"""

    name: str
    batch: int
    in_features: int
    out_features: int
    data_width_bits: int = 8
    acc_width_bits: int = 32

    def __post_init__(self) -> None:
        for field_name in ("batch", "in_features", "out_features"):
            _validate_positive(field_name, getattr(self, field_name))
        _bits_to_storage_bytes(self.data_width_bits)
        _bits_to_storage_bytes(self.acc_width_bits)

    def macs(self) -> int:
        """返回全连接层的总 MAC 数。"""

        # 矩阵视角：输入是 batch x in_features，
        # 权重是 in_features x out_features，输出是 batch x out_features。
        # 每个输出元素累加 in_features 个乘积。
        return self.batch * self.in_features * self.out_features

    def input_bytes(self) -> int:
        """返回存储输入特征矩阵需要的字节数。"""

        # 输入矩阵形状是 batch x in_features。
        num_elements = self.batch * self.in_features
        return num_elements * _bits_to_storage_bytes(self.data_width_bits)

    def weight_bytes(self) -> int:
        """返回存储全连接权重矩阵需要的字节数。"""

        # 权重矩阵形状是 in_features x out_features。
        num_elements = self.in_features * self.out_features
        return num_elements * _bits_to_storage_bytes(self.data_width_bits)

    def output_bytes(self) -> int:
        """返回存储输出特征矩阵需要的字节数。"""

        # 输出矩阵形状是 batch x out_features，并按累加器精度统计。
        num_elements = self.batch * self.out_features
        return num_elements * _bits_to_storage_bytes(self.acc_width_bits)

    def to_gemm_shape(self) -> GEMMShape:
        """返回全连接层对应的 GEMM 形状。"""

        # FC 本身就是 GEMM：
        # A 是输入激活矩阵：batch x in_features。
        # B 是权重矩阵：in_features x out_features。
        # C 是输出激活矩阵：batch x out_features。
        return GEMMShape(
            M=self.batch,
            K=self.in_features,
            N=self.out_features,
            name=self.name,
        )

    def pretty_summary(self) -> str:
        output_shape = (self.batch, self.out_features)
        return (
            f"FullyConnectedLayer(name={self.name}, output_shape={output_shape}, "
            f"macs={self.macs()}, input_bytes={self.input_bytes()}, "
            f"weight_bytes={self.weight_bytes()}, output_bytes={self.output_bytes()})"
        )
