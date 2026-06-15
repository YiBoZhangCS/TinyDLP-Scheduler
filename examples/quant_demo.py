"""Run the TinyDLP int8 quantized matmul demo."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinydlp.quant import int8_matmul_demo


def main() -> None:
    result = int8_matmul_demo()
    print("INT8 matmul demo")
    print(f"scale_A: {result.scale_a}")
    print(f"zero_point_A: {result.zero_point_a}")
    print(f"scale_B: {result.scale_b}")
    print(f"zero_point_B: {result.zero_point_b}")
    print(f"MAE: {result.mae}")
    print(f"max error: {result.max_error}")


if __name__ == "__main__":
    main()
