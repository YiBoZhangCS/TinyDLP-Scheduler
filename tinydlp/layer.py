"""Layer descriptions used by the TinyDLP analytical models."""

from __future__ import annotations

from dataclasses import dataclass

from tinydlp.gemm import GEMMShape


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _bits_to_storage_bytes(bits: int) -> int:
    if bits <= 0:
        raise ValueError("bit width must be positive")
    # Storage bytes per scalar. This rounds up so sub-byte formats, such as
    # int4, can still be represented conservatively in byte-addressed memory.
    return _ceil_div(bits, 8)


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True)
class Conv2DLayer:
    """2D convolution layer in NCHW layout."""

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
        """Return output spatial shape (P, Q)."""

        # P = floor((H + 2 * padding - R) / stride) + 1.
        # Q = floor((W + 2 * padding - S) / stride) + 1.
        # H/W are input height/width, R/S are kernel height/width.
        out_h = (self.in_h + 2 * self.padding - self.kernel_h) // self.stride + 1
        out_w = (self.in_w + 2 * self.padding - self.kernel_w) // self.stride + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                "convolution output spatial shape must be positive, "
                f"got ({out_h}, {out_w})"
            )
        return out_h, out_w

    def output_shape(self) -> tuple[int, int, int, int]:
        """Return output tensor shape in NCHW layout."""

        out_h, out_w = self.output_hw()
        return self.batch, self.out_channels, out_h, out_w

    def macs(self) -> int:
        """Return total multiply-accumulate operations."""

        out_h, out_w = self.output_hw()
        # Each output element accumulates C * R * S products.
        # Total MACs = N * K * P * Q * C * R * S, where:
        # N=batch, K=out_channels, P/Q=output height/width,
        # C=in_channels, R/S=kernel height/width.
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
        """Return bytes needed to store the input activation tensor."""

        # Input activations use data_width_bits per scalar.
        num_elements = self.batch * self.in_channels * self.in_h * self.in_w
        return num_elements * _bits_to_storage_bytes(self.data_width_bits)

    def weight_bytes(self) -> int:
        """Return bytes needed to store the convolution weights."""

        # Weights use data_width_bits per scalar and have shape K x C x R x S.
        num_elements = (
            self.out_channels * self.in_channels * self.kernel_h * self.kernel_w
        )
        return num_elements * _bits_to_storage_bytes(self.data_width_bits)

    def output_bytes(self) -> int:
        """Return bytes needed to store the output activation tensor."""

        out_h, out_w = self.output_hw()
        # Outputs are counted at accumulator precision because this model tracks
        # post-MAC partial/full sums before any later quantization pass.
        num_elements = self.batch * self.out_channels * out_h * out_w
        return num_elements * _bits_to_storage_bytes(self.acc_width_bits)

    def to_gemm_shape(self) -> GEMMShape:
        """Return the equivalent GEMM shape for this convolution."""

        out_h, out_w = self.output_hw()
        # Conv-to-GEMM view after im2col:
        # A is the unfolded input matrix with shape M x K.
        #   M = batch * P * Q, one row for each output spatial position.
        #   K = in_channels * kernel_h * kernel_w, one column per filter tap.
        # B is the weight matrix with shape K x N.
        #   N = out_channels, one column per output channel/filter.
        # C is the output matrix with shape M x N.
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
    """Fully connected layer represented as a batched matrix multiply."""

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
        """Return total multiply-accumulate operations."""

        # Matrix view: input is N x M, weights are M x K, output is N x K.
        # Each output element accumulates M products, so MACs = N * M * K.
        return self.batch * self.in_features * self.out_features

    def input_bytes(self) -> int:
        """Return bytes needed to store the input feature matrix."""

        # Input matrix shape is batch x in_features.
        num_elements = self.batch * self.in_features
        return num_elements * _bits_to_storage_bytes(self.data_width_bits)

    def weight_bytes(self) -> int:
        """Return bytes needed to store the weight matrix."""

        # Weight matrix shape is in_features x out_features.
        num_elements = self.in_features * self.out_features
        return num_elements * _bits_to_storage_bytes(self.data_width_bits)

    def output_bytes(self) -> int:
        """Return bytes needed to store the output feature matrix."""

        # Output matrix shape is batch x out_features and is counted at
        # accumulator precision before any later quantization pass.
        num_elements = self.batch * self.out_features
        return num_elements * _bits_to_storage_bytes(self.acc_width_bits)

    def to_gemm_shape(self) -> GEMMShape:
        """Return the GEMM shape for this fully connected layer."""

        # FC as GEMM:
        # A is the input activation matrix: batch x in_features.
        # B is the weight matrix: in_features x out_features.
        # C is the output activation matrix: batch x out_features.
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
