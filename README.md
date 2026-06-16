# TinyDLP-Scheduler

TinyDLP-Scheduler 是一个面向深度学习处理器的轻量级算子映射与性能建模项目。它从 Conv/FC 层参数出发，把算子映射到 GEMM，结合片上 SRAM、PE 阵列规模、DRAM 带宽和 dataflow，估计计算周期、访存量、PE 利用率和瓶颈。

这个项目的定位是教学和分析模型，不是 RTL 仿真器，也不是完整编译器后端。

## Quick Start

安装依赖：

```bash
pip install -r requirements.txt
```

运行主系统，也就是读取模型 JSON 和硬件 JSON，搜索每层调度方案并生成报告：

```bash
python run.py --model examples/lenet.json --hw examples/dlp_16x16.json --plot
```

运行后主要看这几个输出：

```text
reports/result.csv      每层详细结果，适合用表格查看
reports/summary.md      Markdown 摘要报告
figs/layer_latency.png  每层延迟图
figs/compute_vs_memory.png
figs/pe_utilization.png
figs/dram_traffic.png
```

如果要跑测试：

```bash
conda run -n tinydlp python -m pytest
```

## 输入怎么改

模型输入在 `examples/lenet.json`。一个 Conv2D 层写成：

```json
{
  "type": "conv2d",
  "name": "my_conv",
  "batch": 1,
  "C": 64,
  "H": 32,
  "W": 32,
  "K": 128,
  "R": 5,
  "S": 5,
  "stride": 1,
  "padding": 2
}
```

字段含义：

- `batch`：输入 batch，也就是 N
- `C/H/W`：输入 feature map 的通道、高、宽
- `K`：输出通道数，也就是 `Kout`
- `R/S`：卷积核高、宽
- `stride/padding`：卷积步长和 padding

硬件输入在 `examples/dlp_16x16.json`。常改的字段是：

```json
{
  "array_m": 16,
  "array_n": 16,
  "sram_kb": 64,
  "dram_bandwidth_gb_s": 12.8,
  "data_width_bits": 8,
  "acc_width_bits": 32
}
```

如果要测试新的硬件配置，复制一份 JSON，例如 `examples/dlp_24x16.json`，然后运行：

```bash
python run.py --model examples/lenet.json --hw examples/dlp_24x16.json --plot
```

## 输出怎么看

`reports/result.csv` 同时保留 Conv 原始参数和 GEMM 映射结果。

Conv 原始参数列：

```text
conv_C, conv_H, conv_W, conv_Kout, conv_R, conv_S,
conv_stride, conv_padding, conv_output_P, conv_output_Q
```

GEMM 映射列：

```text
gemm_M_output_positions = batch * P * Q
gemm_K_reduction_CRS    = C * R * S
gemm_N_output_channels  = Kout
```

调度和性能列：

```text
tile_m, tile_k, tile_n
dataflow
systolic_compute_cycles
pe_utilization
dram_bytes
memory_cycles
ideal_overlap_cycles
bottleneck
```

注意：`lenet.json` 里的 `K` 表示卷积输出通道 `Kout`；而 GEMM 里的 `K` 表示 reduction 维度 `C*R*S`。CSV 中已经用解释性列名把这两个概念分开。

## PPT Demo

下面三个脚本是专门给 README/PPT 展示用的，不是主系统入口。它们用固定实验突出某一个硬件映射现象。

### 1. 模数效应与 fill/drain

```bash
python examples/mod_and_fill_demo.py
```

输出：

```text
figs/mod_and_fill_demo.png
```

这个 demo 比较几组相近 Conv/GEMM shape，展示 `M/N` 是否对齐 PE 阵列会明显影响 PE utilization；同时说明当 `K` 较小时，脉动阵列 fill/drain overhead 占比很高。

### 2. SRAM 容量与 dataflow

```bash
python examples/sram_dataflow_demo.py
```

输出：

```text
figs/sram_dataflow_demo.png
```

这个 demo 扫描不同 SRAM 容量，分别展示 `output_stationary`、`weight_stationary`、`input_stationary` 三种 dataflow 的最佳 tile 和 DRAM traffic。它的重点是说明 SRAM 越小，tile 越碎，halo 和重复加载越明显。

### 3. Tc 切分与 partial sum

```bash
python examples/tc_split_psum_demo.py
```

输出：

```text
figs/tc_split_psum_demo.png
```

这个 demo 比较 `Tc=256` 和 `Tc=64`，并区分 partial sum 能否留在 SRAM。它展示的是：当 `Tc < C` 且 partial sum 留不住时，中间 psum 的读写会显著增加 DRAM traffic。

其他辅助 demo：

```bash
python examples/mod_effect_demo.py
python examples/sram_sensitivity_demo.py
python examples/array_size_sensitivity_demo.py
python examples/quant_demo.py
```

这些是较早的小实验，主要用于单独观察模数效应、SRAM 敏感性、阵列大小敏感性和 INT8 量化误差。

## 核心建模逻辑

### Conv 到 GEMM

Conv 层输出空间尺寸为 `P/Q`，映射到 GEMM：

```text
M = batch * P * Q
K = C * R * S
N = Kout
```

Conv tile 使用原生维度描述：

```text
Tb: batch tile
Tm: output channel tile
Tc: input channel tile
Tp: output height tile
Tq: output width tile
```

对应 GEMM tile：

```text
M_tile = Tb * Tp * Tq
N_tile = Tm
K_tile = Tc * R * S
```

PE 阵列的 M 方向对应输出位置数量，所以搜索器会优先选择：

```text
Tb * Tp * Tq % array_m == 0
```

这比单独要求 `Tp` 或 `Tq` 是阵列倍数更准确。

### SRAM 占用

Conv tile 的输入窗口要考虑 halo：

```text
input_h_tile = (Tp - 1) * stride + R
input_w_tile = (Tq - 1) * stride + S
```

SRAM 占用：

```text
input_bytes  = Tb * Tc * input_h_tile * input_w_tile * act_bytes
weight_bytes = Tm * Tc * R * S * weight_bytes
psum_bytes   = Tb * Tm * Tp * Tq * acc_bytes
total_sram   = input_bytes + weight_bytes + psum_bytes
```

空间切分越碎，halo 重叠越多，输入重复加载也越明显。

### 脉动阵列周期

简化 systolic 周期模型：

```text
m_blocks = ceil(M / array_m)
n_blocks = ceil(N / array_n)
cycles = m_blocks * n_blocks * (K + array_m + array_n - 2)
```

`ceil` 体现模数效应：尾块即使没有填满阵列，也会占用完整 block。  
`array_m + array_n - 2` 是简化的 fill/drain overhead。

### Dataflow

项目比较三种简化 dataflow：

- `output_stationary`：尽量让 partial sum 留在片上，减少 psum 读写
- `weight_stationary`：尽量复用权重 tile，减少 weight 重复加载
- `input_stationary`：尽量复用输入 tile，减少 input 重复加载

如果 `Tc < C`，一个 output tile 需要跨多个 input-channel tile 累加。若 psum 不能留在 SRAM，模型会加入中间 partial sum 的读写 traffic。

## 代码结构

```text
run.py                  主入口：读取 JSON，运行调度，生成报告和图
tinydlp/layer.py        Conv/FC 层定义，输出尺寸和 Conv->GEMM 映射
tinydlp/hardware.py     硬件参数：PE 阵列、SRAM、DRAM 带宽、位宽
tinydlp/tile.py         GEMM/Conv tile 和 SRAM 占用计算
tinydlp/compute_model.py  理想/阵列感知/脉动阵列 compute cycles
tinydlp/dataflow.py     OS/WS/IS 的 DRAM traffic 估算
tinydlp/scheduler.py    tile/dataflow 搜索器
tinydlp/report.py       CSV 和 Markdown 报告生成
tinydlp/plot.py         标准图表生成
tinydlp/plot_style.py   统一图表风格
examples/*.py           展示 demo
tests/*.py              单元测试
```

## Limitations

- 不是 RTL 级仿真器。
- 不是 cycle-accurate 硬件模型。
- DRAM traffic 是简化解析估算。
- 没有真实 DMA、NoC、SRAM bank conflict 或 cache replacement 模型。
- 没有 TVM、MLIR、代码生成或真实编译器后端。
- systolic fill/drain 是简化模型，没有覆盖所有微架构流水细节。
