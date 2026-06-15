# 第 1 页：项目主线

## TinyDLP-Scheduler：面向深度学习处理器的算子映射性能建模

```text
Conv layer
    ↓
Conv-native tiling
    ↓
GEMM mapping
    ↓
Systolic array model
    ↓
SRAM/dataflow traffic
    ↓
Bottleneck analysis
```

- 从 Conv 原生维度 `Tb/Tm/Tc/Tp/Tq` 描述 tile。
- 将 Conv tile 映射为 GEMM 的 `M_tile/N_tile/K_tile`。
- 用阵列尺寸、模数效应和 fill/drain overhead 估计 compute cycles。
- 用 SRAM capacity、halo、dataflow 和 partial sum 策略估计 DRAM traffic。
- 输出 compute-bound / memory-bound 判断和适合展示的图表。

# 第 2 页：三个展示实验

## 实验 1：模数效应与 fill/drain

- 图：`figs/mod_and_fill_demo.png`
- 表格标题：Conv-to-GEMM modulus and systolic fill/drain
- 结论：同样在相近 MACs 规模下，`M/N` 是否对齐 PE 阵列会明显影响 PE utilization；当 `K_tile` 较小时，fill/drain overhead 占比更高。

## 实验 2：SRAM 容量敏感性

- 图：`figs/sram_dataflow_demo.png`
- 表格标题：SRAM sweep with per-dataflow best Conv tiles
- 结论：SRAM 越小，tile 越碎，halo 和重复加载更明显，DRAM traffic 上升；图中分别展示 OS/WS/IS 三种复用策略，避免只看一个 best dataflow 而看不出差异。

## 实验 3：Tc 切分与 partial sum

- 图：`figs/tc_split_psum_demo.png`
- 表格标题：Input-channel split and partial-sum spill
- 结论：当 `Tc < C` 时，一个 output tile 需要跨多个 input-channel tile 累加；如果 partial sum 不能留在 SRAM，中间 psum 的读写会显著增加访存。

# 第 3 页：项目收获

## 中文讲稿

通过 TinyDLP-Scheduler，我把卷积层从原始 Conv 维度映射到 GEMM 计算，并进一步观察它在脉动阵列上的执行行为。这个项目让我更直观地理解了 PE 阵列利用率不仅取决于 MACs，还会受到 M/N 模数效应、K 维度 fill/drain overhead、SRAM tiling、halo 重叠和 dataflow 选择的影响。通过对 output/weight/input-stationary 的访存估算，以及 compute cycles 和 memory cycles 的对比，我也更清楚地理解了 compute-bound 与 memory-bound 的形成原因。
