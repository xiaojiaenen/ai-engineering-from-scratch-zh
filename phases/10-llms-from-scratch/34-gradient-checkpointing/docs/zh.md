# 梯度检查点与激活重计算

> 反向传播会保留所有中间激活。对于700亿参数和128K上下文，每个秩需要3 TB的激活存储。检查点技术以算力换内存：通过重计算替代存储。关键问题在于选择哪些部分进行丢弃，而答案绝非“全部丢弃”。

**类型:** 构建实践
**语言:** Python（使用numpy，可选torch）
**前置条件:** 第10阶段第04课（预训练Mini-GPT），第10阶段第05课（规模化与分布式）
**时间:** 约70分钟

## 问题本质

训练Transformer时，每一层都会存储所有在反向传播中需要微分的算子输入：注意力机制的输入、Q/K/V投影、softmax输出、FFN输入、归一化层输出以及残差流。对于隐藏层维度为`d`、序列长度为`L`、批次大小为`B`的单层，存储量约为`12 * B * L * d`个浮点数。

当`d=8192, L=8192, B=1`时，BF16格式下每层需要800 MB存储。64层模型需要51 GB激活存储——这还未乘以微批次大小，未加上注意力softmax中间结果（每头`L^2`），也未考虑张量并行的部分副本。

双重成本：BF16权重加优化器状态可能适配80GB显存，但激活存储会导致溢出。梯度检查点（又称激活重计算）是标准解决方案。丢弃大部分激活；在反向传播时重新执行前向计算以恢复它们。代价：额外算力消耗。收益：内存占用降至检查点段数与总层数之比。

粗暴实施的检查点会使每步前向算力消耗增加约33%。若采用Korthikanti等人提出的“智能选择”进行选择性检查点，可节省5倍内存，算力开销低于5%。结合FP8矩阵乘法、FSDP卸载和专家并行MoE技术，这至关重要：无论是内存还是计算资源都不容浪费。

## 核心概念

### 反向传播的实际需求

`output = layer(input)`。反向传播需要`grad_input`和`grad_params`。为计算这些值需要：

- `input`（用于计算线性层的`grad_params = input.T @ grad_output`）
- 某些激活导数中间结果（ReLU/GELU/softmax的导数取决于激活值）

前向传播过程会自动在自动微分图中存储这些信息。每个`tensor.retain_grad()`和每个需要其输入的算子都会保留引用。

### 简单全面检查点

将网络划分为`N`个段。前向传播时仅存储每个段的*输入*。当反向传播需要中间结果时，重新执行该段的前向计算以生成它们，然后进行微分。

示例：将32层Transformer划分为32个段，每段1层。

- 内存占用：32个层输入（很小）对比 32 ×（每层激活量）（巨大）。
- 额外算力：每段多执行1次前向计算，即总前向算力增加约33%（因为反向算力是前向的2倍，完整步骤从1+2=3变为1+1+2=4单位）。

这是Chen等人2016年提出的原始方案：每`sqrt(L)`层设置一个检查点以平衡内存与算力。对于L=64，即设置8个检查点。

### 选择性检查点（Korthikanti 2022）

并非所有激活的存储成本相同。注意力softmax输出为`B*L*L*heads`，随序列长度*平方级*增长。FFN隐藏激活为`B*L*4d`，呈线性增长。对于长序列，softmax占主导地位。

选择性检查点保留存储成本低的激活（线性投影、残差），仅重计算存储成本高的激活（注意力机制）。以最小算力开销重计算，同时节省O(L²)内存。

Megatron-Core将此实现为“选择性”激活重计算。2024年后大多数前沿训练都采用此技术。

### 卸载

重计算的替代方案：在前向与反向传播之间将激活数据传输到CPU内存。需要PCIe带宽；当闲置带宽足以覆盖重计算成本时，此方案更有利。混合策略很常见：对某些层使用检查点，对其他层使用卸载。

FSDP2将卸载作为一级功能。当GPU内存成为瓶颈而CPU-GPU传输存在空闲时，卸载方案表现出色。

### 重计算成本模型

在`L`层中每`k`层进行简单检查点的每步算力：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

选择性检查点仅重计算注意力内核而非整层：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型

每层激活量：`A`。对于`L`层，总激活内存：`L * A`。

完全检查点（段大小为1）：仅存储`L * input_volume`（标准Transformer约`L * 1/10 A`）。节省约`9 * L * A * 1/10`。

每`k`层设置检查点：存储`L/k * A`加上活跃段内`k-1`层的存储。

当`k = sqrt(L)`时，内存与重计算成本均随`sqrt(L)`变化——这是均匀成本层的最优权衡。

### 不应使用检查点的情况

- 已在流水线阶段中执行的内层。它们反正需要完成计算。
- 若首层和末层主导该阶段计算（Transformer中少见）。
- 已使用FlashAttention的注意力内核——Flash已能快速重计算softmax，在此基础上添加层级检查点收益甚微。

### 实现模式

1. **函数包装器：** 用`torch.utils.checkpoint.checkpoint(fn, input)`包装一个段。PyTorch仅存储`input`，反向传播时重计算所有其他内容。

2. **基于装饰器：** 将层标记为可检查点；训练器在配置时决定哪些段被包装。

3. **手动显式重计算：** 自行编写反向传播代码，调用自定义`recompute_forward`函数，使用存储的输入复制前向过程。

三种方式功能等效。包装器是标准实现范式。

### 与TP/PP/FP8的交互

- **张量并行：** 检查点输入需在重计算时进行收集或重新分散；需处理通信开销。
- **流水线并行：** 典型模式是对每个流水线阶段的前向过程设置检查点，使反向顺序的微批次可复用激活内存。
- **FP8重计算：** 重计算期间更新的amax历史记录必须与原始前向过程一致，否则FP8缩放会漂移。大多数框架会快照缩放因子。

## 构建实现

### 步骤1：带有分段的玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 步骤2：需要全部激活的简单反向传播

```python
def model_backward(grad_output, activations, params):
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

### 步骤3：每k层检查点的内存优化

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
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

### 步骤4：成本模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
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

### 步骤5：内存估算器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 步骤6：最优段大小

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 步骤7：选择性检查点决策

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 使用指南

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint` — PyTorch的标准包装器。包装函数；仅存储输入，反向传播时重计算。
- **Megatron-Core激活重计算**：支持`selective`、`full`和`block`模式。2024年后前沿训练标准。
- **FSDP2卸载**：在FSDP2分片中启用`offload_policy`的`module.to_empty(device="cpu")`将激活数据卸载到CPU而非重计算。
- **DeepSpeed ZeRO-Offload**：将优化器状态和激活数据卸载到CPU，与检查点形成互补。

## 产出成果

本课将生成`outputs/prompt-activation-recompute-policy.md` — 一个提示模板，输入您的模型配置（层数、隐藏维度、序列长度、批次大小）和可用GPU内存，输出逐层重计算策略（无/选择性/完全/卸载）。

## 练习

1. 验证正确性。运行`model_forward` + `model_backward`（完整激活）对比`model_forward_checkpointed` + `model_backward_checkpointed`（分段处理）。参数梯度必须在机器精度内完全一致。

2. 从1到`L`扫描段大小`k`。绘制算力开销与内存占用曲线，找到拐点。

3. 实现选择性检查点：存储注意力模块输入但不存储其中间结果。针对32层模型（序列长度8192）测量其与完全层检查点相比的算力开销。

4. 添加卸载功能。将段输入保存到模拟“CPU缓冲区”（单独列表）。将“PCIe带宽”测量为字节数/时间，找到卸载与重计算的盈亏平衡点。

5. 对比有无`torch.utils.checkpoint`的真实PyTorch Transformer进行基准测试。使用`torch.cuda.max_memory_allocated`测量内存占用和步骤耗时。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 梯度检查点 | “通过重新执行前向计算节省内存” | 仅存储段输入；反向传播时重计算中间结果以获得梯度支持张量 |
| 激活重计算 | “与检查点相同” | 同一技术的高性能计算术语 |
| 段大小(k) | “每个检查点包含多少层” | 一起丢弃并重新生成的中间结果所属的层数 |
| 选择性检查点 | “Korthikanti的技巧” | 仅重计算存储成本高的激活（注意力softmax）；保留低成本激活 |
| 完全检查点 | “简单版本” | 重计算每个段内每层的中间结果 |
| 块检查点 | “粗粒度” | 检查整个Transformer块；最大粒度 |
| 算力开销 | “计算税” | 每步额外算力 = (重计算算力) / (前向+反向算力)；简单法33%，选择性法5% |
| 激活卸载 | “传输到CPU” | 在前向→反向期间将激活数据移至CPU内存；重计算的替代方案 |
| sqrt-L规则 | “经典最优解” | 对于均匀成本层，最优检查点间隔为sqrt(L)层 |
| 注意力-softmax体积 | “O(L²)问题” | L² × 头数 × 批次大小浮点数；长上下文时主导激活内存 |

## 延伸阅读

- [Chen等人, 2016 -- "以亚线性内存成本训练深度网络"](https://arxiv.org/abs/1604.06174) -- 形式化梯度检查点的原始论文
- [Korthikanti等人, 2022 -- "减少大型Transformer模型的激活重计算"](https://arxiv.org/abs/2205.05198) -- 选择性激活重计算及正式成本分析
- [Pudipeddi等人, 2020 -- "使用新执行算法以恒定内存训练大型神经网络"](https://arxiv.org/abs/2002.05645) -- 通过反向模式重材料化的替代恒定内存方案
- [Ren等人, 2021 -- "ZeRO-Offload：民主化百亿级模型训练"](https://arxiv.org/abs/2101.06840) -- 大规模激活卸载
- [PyTorch torch.utils.checkpoint文档](https://pytorch.org/docs/stable/checkpoint.html) -- 标准API
- [Megatron-Core激活重计算文档](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- 选择性、完全和块模式