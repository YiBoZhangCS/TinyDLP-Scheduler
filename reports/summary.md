# TinyDLP-Scheduler Report

## 项目说明

TinyDLP-Scheduler 用于学习 Conv/GEMM 在简化深度学习处理器上的映射、tiling、dataflow、PE 利用率和瓶颈分析。该项目是分析模型，不是 RTL 级 cycle-accurate 仿真。

## 硬件配置

- name: `TinyDLP-16x16`
- array: `16 x 16`
- MAC/PE/cycle: `1`
- frequency: `500.0 MHz`
- SRAM: `64 KB`
- DRAM bandwidth: `12.8 GB/s`
- data width: `8 bits`
- accumulator width: `32 bits`

## 每层结果

| layer | original shape | GEMM M/K/N | MACs | tile | dataflow | PE util | DRAM bytes | no-overlap | ideal-overlap | bottleneck |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---|
| conv1 | Conv N=1, C=3, HxW=32x32, Kout=16, RxS=3x3, P/Q=32x32 | 1024/27/16 | 442368 | 512x1x16 | output_stationary | 0.4737 | 44896 | 5402 | 3648 | compute-bound |
| conv2 | Conv N=1, C=16, HxW=32x32, Kout=32, RxS=3x3, P/Q=32x32 | 1024/144/32 | 4718592 | 256x1x32 | output_stationary | 0.8276 | 198656 | 30032 | 22272 | compute-bound |
| fc1 | FC batch=1, in=1024, out=120 | 1/1024/120 | 122880 | 1x512x1 | input_stationary | 0.0569 | 124984 | 13315 | 8432 | compute-bound |

## Network Summary

- 总 MACs: `5283840`
- 总 DRAM traffic: `368536` bytes
- 总 no-overlap cycles: `48749`
- 总 ideal-overlap cycles: `34352`

## Bottleneck 解释

- `compute-bound`: systolic compute cycles 大于 memory cycles，说明当前估计下主要受计算阵列执行时间限制。
- `memory-bound`: memory cycles 大于或等于 systolic compute cycles，说明当前估计下主要受 DRAM 搬运时间限制。

## Overlap 解释

- `no-overlap`: 搬运和计算完全串行，`cycles = compute_cycles + memory_cycles`。
- `ideal-overlap`: 假设通过双缓冲等方式理想重叠搬运和计算，`cycles = max(compute_cycles, memory_cycles)`。这是性能下界，不是真实硬件精确时间。
