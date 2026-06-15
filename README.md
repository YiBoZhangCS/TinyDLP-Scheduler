# TinyDLP-Scheduler

## Project Overview

TinyDLP-Scheduler is a lightweight operator mapping, performance modeling, and
schedule search tool for deep-learning processors.

It maps Conv2D and fully connected layers to GEMM shapes, estimates compute and
memory costs on a simplified systolic-array-like DLP, searches SRAM-constrained
tiling/dataflow choices, and generates reports and visualizations for
bottleneck analysis.

The goal is educational and analytical: to make accelerator performance effects
visible with small, readable Python models rather than a heavyweight compiler or
RTL simulator.

## Motivation

Counting MACs alone is not enough to understand accelerator performance.

- Peak compute throughput is not the same as delivered performance.
- Modulus effects can reduce PE utilization when GEMM dimensions do not divide
  cleanly into the array shape.
- DRAM bandwidth and SRAM capacity can dominate or reshape the best schedule.
- Dataflow determines whether inputs, weights, or output partial sums are
  reused efficiently.
- A layer with fewer MACs can still be slow if it has poor utilization or heavy
  memory traffic.

TinyDLP-Scheduler is built to expose these effects explicitly.

## Supported Features

- Conv2D / FC to GEMM mapping
- MACs statistics
- array-aware compute model
- systolic fill/drain overhead
- PE utilization estimation
- SRAM-constrained tiling
- output-stationary / weight-stationary / input-stationary dataflows
- no-overlap / ideal-overlap latency estimates
- CSV and Markdown reports
- matplotlib visualizations
- INT8 per-tensor affine quantization demo

## Key Concepts

### Conv-native tiling

For Conv2D, TinyDLP can describe tiles in the original convolution loop space:

- `Tb`: batch tile
- `Tm`: output channel tile
- `Tc`: input channel tile
- `Tp`: output height tile
- `Tq`: output width tile

This makes the schedule easier to reason about than a GEMM-only tile. For
example, reducing `Tp`/`Tq` means the spatial tile is smaller, while reducing
`Tc` means one output tile must be accumulated across multiple input-channel
tiles.

### Mapping to GEMM

A Conv tile maps to a GEMM tile as:

```text
M_tile = Tb * Tp * Tq
N_tile = Tm
K_tile = Tc * R * S
```

`M_tile` is the number of output positions, `N_tile` is the number of output
channels, and `K_tile` is the reduction dimension over input channels and kernel
taps.

### SRAM capacity model

Conv tile SRAM usage is modeled with input, weight, and partial-sum storage:

```text
input_h_tile = (Tp - 1) * stride + R
input_w_tile = (Tq - 1) * stride + S

input_bytes  = Tb * Tc * input_h_tile * input_w_tile * act_bytes
weight_bytes = Tm * Tc * R * S * weight_bytes
psum_bytes   = Tb * Tm * Tp * Tq * acc_bytes
total_sram   = input_bytes + weight_bytes + psum_bytes
```

With double buffering, input and weight tiles are counted twice:

```text
total_sram = 2 * (input_bytes + weight_bytes) + psum_bytes
```

Spatial tiling creates halo because a `Tp x Tq` output tile still needs the
kernel footprint around its border. Neighboring output tiles therefore load
overlapping input regions, especially for larger kernels or small spatial tiles.

### Systolic array model

The simplified systolic model uses the array shape explicitly:

```text
m_blocks = ceil(M / array_m)
n_blocks = ceil(N / array_n)
cycles = m_blocks * n_blocks * (K + array_m + array_n - 2)
```

The `ceil` terms expose modulus effects: tail blocks that do not fill the array
still reserve a full `array_m x array_n` block. The `array_m + array_n - 2` term
is a simplified fill/drain overhead. When `K` is small, this overhead is a large
fraction of the total tile time, so PE utilization can drop even if the MAC
count looks modest.

### Dataflow model

TinyDLP estimates DRAM traffic for three simplified dataflows:

- `output_stationary`: keeps output partial sums on chip when possible, reducing
  intermediate psum read/write traffic.
- `weight_stationary`: keeps weight tiles on chip across multiple spatial
  tiles, reducing repeated weight loads.
- `input_stationary`: keeps input activation tiles on chip across output-channel
  tiles, reducing repeated input loads.

If `Tc < C`, each output tile needs multiple input-channel tiles:

```text
num_c_tiles = ceil(C / Tc)
```

When partial sums stay in SRAM, the final output is written once. If partial
sums spill to DRAM, the model adds intermediate read/write traffic:

```text
extra_psum_traffic = 2 * (num_c_tiles - 1) * psum_tile_bytes * num_output_tiles
```

This is still an analytical estimate, not a DMA-accurate memory trace.

### Bottleneck and overlap

- **Bottleneck**: a schedule is compute-bound when systolic compute cycles
  exceed memory cycles, otherwise it is memory-bound.
- **No-overlap vs ideal-overlap**: no-overlap assumes memory movement and
  compute are serialized; ideal-overlap assumes perfect overlap, such as an
  optimistic double-buffering lower bound.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the LeNet-like example and generate reports plus figures:

```bash
python run.py --model examples/lenet.json --hw examples/dlp_16x16.json --plot
```

Run standalone demos:

```bash
python examples/mod_effect_demo.py
python examples/sram_sensitivity_demo.py
python examples/array_size_sensitivity_demo.py
python examples/quant_demo.py
python examples/mod_and_fill_demo.py
python examples/sram_dataflow_demo.py
python examples/tc_split_psum_demo.py
```

Generated outputs:

- `reports/result.csv`
- `reports/summary.md`
- `figs/layer_latency.png`
- `figs/compute_vs_memory.png`
- `figs/pe_utilization.png`
- `figs/dram_traffic.png`

## Add and Test a New Conv Layer

Edit `examples/lenet.json` or create a new model JSON. A Conv2D layer uses:

```json
{
  "type": "conv2d",
  "name": "my_conv",
  "batch": 1,
  "C": 64,
  "H": 32,
  "W": 32,
  "K": 128,
  "R": 3,
  "S": 3,
  "stride": 1,
  "padding": 1
}
```

Here `C/H/W` describe the input feature map, `K` means `Kout` output channels,
and `R/S` describe the kernel height/width. The report keeps both views:

- original Conv fields: `conv_C`, `conv_H`, `conv_W`, `conv_Kout`, `conv_R`,
  `conv_S`, `conv_output_P`, `conv_output_Q`
- GEMM fields: `gemm_M_output_positions`, `gemm_K_reduction_CRS`,
  `gemm_N_output_channels`

Run a custom model and hardware pair with:

```bash
python run.py --model examples/lenet.json --hw examples/dlp_16x16.json --plot
```

To test a different SRAM size or PE array, copy `examples/dlp_16x16.json` and
change `sram_kb`, `array_m`, `array_n`, or `dram_bandwidth_gb_s`, then pass the
new hardware JSON with `--hw`.

## Demo Results

The newer Conv-native demos are designed for README and PPT figures:

```bash
python examples/mod_and_fill_demo.py
python examples/sram_dataflow_demo.py
python examples/tc_split_psum_demo.py
```

They generate:

- `figs/mod_and_fill_demo.png`: compares nearby Conv/GEMM shapes and shows how
  M/N alignment plus fill/drain overhead changes cycles and PE utilization.
- `figs/sram_dataflow_demo.png`: sweeps SRAM capacity and shows how smaller
  SRAM forces smaller tiles, higher DRAM traffic, and a dataflow-dependent best
  schedule.
- `figs/tc_split_psum_demo.png`: compares `Tc=256` and `Tc=64`, with and
  without keeping partial sums in SRAM.

## Example Output

Example command:

```bash
python run.py --model examples/lenet.json --hw examples/dlp_16x16.json --plot
```

Representative output:

```text
TinyDLP-Scheduler
Hardware: TinyDLP-16x16
Model: examples/lenet.json
Overlap mode: ideal

Layer: conv1
ScheduleResult
  GEMM shape: M=1024, K=27, N=16
  tile: tile_m=512, tile_k=1, tile_n=16
  dataflow: output_stationary
  MACs: 442368
  ideal compute cycles: 1728
  array-aware compute cycles: 1728
  systolic compute cycles: 3648
  PE utilization: 0.4737
  DRAM traffic: 44896 bytes
  memory cycles: 1754
  no-overlap cycles: 5402
  ideal-overlap cycles: 3648
  bottleneck: compute-bound

Network summary
  total MACs: 5283840
  total DRAM bytes: 368536
  total no-overlap cycles: 48749
  total ideal-overlap cycles: 34352
```

## Limitations

- This is not an RTL-level simulator.
- This is not a cycle-accurate model of a real accelerator.
- DRAM traffic is a simplified analytical estimate.
- There is no real DMA schedule, NoC model, SRAM bank conflict model, or cache
  replacement model.
- There is no real TVM, MLIR, code generation, or compiler backend integration.
- The systolic model uses simplified fill/drain overhead and does not model
  every microarchitectural pipeline detail.
- The quantization demo is for numerical intuition, not production calibration.

## Future Work

- ONNX or `torch.fx` frontend import
- More precise systolic-array simulator
- Double-buffering pipeline model
- NoC and SRAM bank conflict model
- Sparse tensor and mixed-precision support
- Verilog NPU backend or FPGA prototype
