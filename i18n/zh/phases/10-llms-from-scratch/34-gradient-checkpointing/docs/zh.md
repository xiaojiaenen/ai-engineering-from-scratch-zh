# 梯度检查点与激活重计算

> 反向传播会保存所有中间激活值。在70B参数和128K上下文的规模下，每个rank的激活值占用高达3TB。检查点机制通过计算换内存：重新计算而非保存。问题在于应该丢弃哪些片段，而答案不是"全部丢弃"。

**类型：** 构建
**语言：** Python（含numpy，可选torch）
**前置条件：** 第10阶段第04课（预训练Mini-GPT）、第10阶段第05课（扩展与分布式）
**时长：** 约70分钟

## 问题所在

训练Transformer时，每一层都会存储反向传播中需要求导的所有操作的输入：注意力输入、Q/K/V投影、softmax输出、FFN输入、归一化输出以及残差流。对于隐藏维度为 `d`、序列长度 `L`、批次大小 `B` 的层，每层需要存储约 `12 * B * L * d` 个浮点数。

当 `d=8192, L=8192, B=1` 时，在BF16精度下每层占800MB。一个64层的模型需要51GB的激活值内存——这还是没乘以微批次大小、没加入注意力softmax中间值（每个头 `L^2`）、没考虑张量并行的部分副本之前的数据。

双面账单：BF16权重加上优化器状态可能能放进80GB显存，但激活值会超出限制。梯度检查点（又称激活重计算）是标准解决方案。丢弃大部分激活值，在反向传播期间重做前向传播来获取它们。代价：额外的FLOPs。收益：内存按检查点片段数与总层数的比例下降。

简单粗暴地做，每步的检查点开销约为前向传播FLOPs的33%。做得好——按照Korthikanti等人的"智能选择"进行选择性检查点——你可以在不到5%的FLOP开销下节省5倍内存。结合FP8矩阵乘法、FSDP卸载和专家并行MoE，这真的很重要：你既不能承受内存溢出，也不能容忍浪费的计算。

## 概念解析

### 反向传播实际需要什么

`output = layer(input)`。反向传播需要 `grad_input` 和 `grad_params`。为了计算它们，它需要：

- `input`（用于线性层的 `grad_params = input.T @ grad_output`）
- 一些激活导数中间值（ReLU/GELU/softmax的导数依赖于激活值本身）

前向传播会自动在自动微分图中存储这些内容。每个 `tensor.retain_grad()` 和每个需要其输入的操作都会保留引用。

### 朴素的全量检查点

将网络分成 `N` 个片段。前向传播时，只存储每个片段的*输入*。当反向传播需要中间值时，重新运行该片段的前向传播来生成它们，然后进行微分。

示例：将32层Transformer分成32个各含1层的片段。

- 内存：32个层输入（较小）对比 32 * (每层激活量)（很大）。
- 额外计算：每个片段额外一次前向传播，即总共约33%更多的前向FLOPs（因为反向传播是前向的2倍，完整步骤从 1 + 2 = 3 变成 1 + 1 + 2 = 4 单位）。

这是Chen等人2016年的原始方案：每隔 `sqrt(L)` 层设置一个检查点以平衡内存和计算。对于L=64，那就是8个检查点。

### 选择性格检查点（Korthikanti 2022）

并非所有激活值的成本都一样。注意力softmax输出是 `B*L*L*heads`，随序列长度*二次方*增长。FFN隐藏激活是 `B*L*4d`，线性增长。对于长序列，softmax占主导地位。

选择性格检查点保留易于存储的激活值（线性投影、残差），只重新计算昂贵的部分（注意力）。你支付最小的FLOPs来重新计算，但节省了O(L^2)的内存。

Megatron-Core将其实现为"selective"激活重计算。在2024年及以后的前沿训练中被广泛使用。

### 卸载

重新计算的替代方案是在前向和反向传播之间将激活值传输到CPU内存。需要PCIe带宽；当空闲带宽超过重新生成的成本时才有利。混合策略很常见：对某些层检查点，对其他层卸载。

FSDP2将卸载作为一等公民选项。当GPU受限于内存但CPU-GPU传输有空间时，卸载表现优异。

### 重计算成本模型

每步使用朴素检查点（每隔 `k` 层，共 `L` 层）的FLOPs：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # 片段中每层额外一次前向传播
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

使用选择性格检查点，你只重新计算注意力内核，而不是整个层：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型

每层激活值体积：`A`。对于 `L` 层，总激活值内存：`L * A`。

全量检查点（片段大小为1）：只存储 `L * input_volume`（对于标准Transformer约为 `L * 1/10 A`）。节省约 `9 * L * A * 1/10`。

每隔 `k` 层检查点：存储 `L/k * A` 加上活跃片段内 `k-1` 层的激活值。

在 `k = sqrt(L)` 时，内存和重计算成本都随 `sqrt(L)` 缩放——这是均匀成本层的最优权衡。

### 何时不需要检查点

- 流水线阶段内部已经处于飞行状态的层。它们无论如何都必须完成。
- 如果首层和末层主导阶段的计算（在Transformer中罕见）。
- 已经使用FlashAttention的注意力内核——Flash已经快速重算了softmax，因此额外的层级别检查点增加的价值有限。

### 实现模式

1. **函数包装器：** 用 `torch.utils.checkpoint.checkpoint(fn, input)` 包装一个片段。PyTorch只存储 `input`，反向传播时重新计算所有内容。

2. **基于装饰器：** 将层标记为可检查点；训练器在配置时决定哪些片段被包装。

3. **手动显式重计算：** 自己编写反向传播，调用自定义的 `recompute_forward`，用存储的输入复制前向传播。

所有三种方法给出相同的功能结果。包装器是标准惯用法。

### 与TP / PP / FP8的交互

- **张量并行：** 检查点输入必须在前向重计算时收集或重新分散；处理通信成本。
- **流水线并行：** 典型模式是对每个流水线阶段的前向传播进行检查点，以便逆序微批次可以复用激活值内存。
- **FP8重计算：** 重计算期间更新的amax历史记录必须与原前向传播匹配，否则FP8缩放会漂移。大多数框架会快照缩放值。

```figure
activation-recompute
```

## 构建它

### 第1步：带片段的玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    # 线性前向传播
    return x @ w + b


def relu(x):
    # ReLU激活函数
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    # 单层前向传播：线性 -> ReLU -> 线性
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    # 模型前向传播，保存所有激活值
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 第2步：朴素反向传播需要所有激活值

```python
def model_backward(grad_output, activations, params):
    # 朴素反向传播：使用所有保存的激活值
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 第3步：每隔k层的检查点内存

```python
def model_forward_checkpointed(x, params, k=4):
    # 前向传播时每隔k层保存一次激活值
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    # 反向传播时按需重计算片段
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 第4步：成本模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    # 计算检查点的成本开销
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    # 选择性检查点的成本模型
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 第5步：内存估算器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    # 估算激活值内存占用（MB）
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    # 检查点后剩余内存估算
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 第6步：最优片段大小

```python
def optimal_segment(n_layers):
    # 使用sqrt(L)规则确定最优片段大小
    return int(round(np.sqrt(n_layers)))
```

### 第7步：选择性检查点决策

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    # 根据层类型和激活值大小决定是否重计算
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 使用它

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint` —— PyTorch中的标准包装器。包装一个函数，仅保存输入，反向传播时重新计算。
- **Megatron-Core激活重计算**：支持 `selective`、`full` 和 `block` 模式。在2024年及以后的前沿训练中是标准配置。
- **FSDP2卸载**：`module.to_empty(device="cpu")` 配合FSDP2中的 `offload_policy` 将激活值卸载到CPU而非重新计算。
- **DeepSpeed ZeRO-Offload**：用于优化器状态和激活值的CPU卸载，与检查点机制互补。

## 交付物

本课程生成 `outputs/prompt-activation-recompute-policy.md`——一个提示文件，接受你的模型配置（层数、隐藏维度、序列长度、批次大小）和可用GPU显存，输出逐层的重计算策略（无/选择性/全量/卸载）。

## 练习

1. **验证正确性**。运行 `model_forward` + `model_backward`（全量激活值）对比 `model_forward_checkpointed` + `model_backward_checkpointed`（分段）。参数梯度必须在机器精度内完全一致。

2. **扫描片段大小 `k`**。从1到 `L` 扫描片段大小，绘制FLOP开销和内存曲线。找到曲线的拐点。

3. **实现选择性检查点**。保存注意力模块的输入但不保存其中间值。针对32层模型在seq=8192时，测量相对于全层检查点的FLOP开销。

4. **添加卸载**。将片段输入保存到模拟的"CPU缓冲区"（一个单独的列表）。以字节/时间衡量"PCIe带宽"，找到卸载与重计算之间的盈亏平衡点。

5. **基准测试**。用和不用 `torch.utils.checkpoint` 基准测试真实的PyTorch Transformer。通过 `torch.cuda.max_memory_allocated` 测量内存和步时间。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 梯度检查点 | "通过重做前向传播来节省内存" | 仅保存片段输入；在反向传播时重新计算中间值以获取支持梯度的张量 |
| 激活重计算 | "与检查点相同" | 高性能计算领域的同名技术 |
| 片段大小 (k) | "每个检查点多少层" | 一起被丢弃并重新生成的层数 |
| 选择性格检查点 | "Korthikanti的技巧" | 只重新计算存储昂贵的激活值（注意力softmax）；保留便宜的 |
| 全量检查点 | "朴素版本" | 在每個片段中重新计算每一层的中间值 |
| 块检查点 | "粗粒度" | 对整个Transformer块进行检查点；最大粒度 |
| FLOP开销 | "计算税" | 每步额外FLOPs = (重计算FLOPs) / (前向+反向FLOPs)；朴素33%，选择性5% |
| 激活值卸载 | "发送到CPU" | 在前向->反向传播间将激活值移动到CPU内存；重计算的替代方案 |
| sqrt-L规则 | "经典最优解" | 对于均匀成本层，最优检查点间距是sqrt(L)层 |
| 注意力softmax体积 | "O(L^2)问题" | L^2 * heads * batch 浮点数；在长上下文时主导激活值内存 |

## 延伸阅读

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) —— 形式化梯度检查点的开创性论文
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) —— 选择性激活重计算及形式化成本分析
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) —— 通过反向模式重生成的替代常数内存方法
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) —— 大规模激活值卸载
- [PyTorch torch.utils.checkpoint 文档](https://pytorch.org/docs/stable/checkpoint.html) —— 标准API
- [Megatron-Core 激活重计算文档](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) —— 选择性、全量和块模式
