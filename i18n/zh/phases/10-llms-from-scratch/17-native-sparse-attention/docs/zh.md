# Native Sparse Attention (DeepSeek NSA)

> 在 64k token 长度下，注意力机制占据了 70-80% 的解码延迟。每个开源模型实验室都有修复方案。DeepSeek 的 NSA（ACL 2025 最佳论文）是真正落地的方案：三条并行的注意力分支——压缩的粗粒度 token、选择性保留的细粒度 token、以及用于局部上下文的滑动窗口——通过学习的门控机制组合。它与硬件对齐（对 kernel 友好），原生可训练（在预训练阶段使用，而非推理时外挂），在 64k 解码时比 FlashAttention 更快，同时达到或超越全注意力质量。本课将从头实现三条分支，并说明为何稀疏性可以端到端可微分。

**类型：** 构建
**语言：** Python (stdlib)
**前置知识：** Phase 7 · 12 (KV cache、flash-attention)、Phase 7 · 15 (注意力变体)、Phase 10 · 16 (可微分注意力)
**预计时间：** ~60 分钟

## 学习目标

- 阐述 NSA 的三条注意力分支及其各自捕获的信息类型。
- 解释 NSA 为何是"原生可训练"，而此前稀疏注意力方法仅为推理时方案。
- 以压缩块大小 `l` 和 top-k 选择数 `k` 为函数，计算 NSA 相对全注意力在 64k 上下文下的注意力计算节省量。
- 在 stdlib Python 中对短合成序列实现三支合并，并验证门控权重行为合理。

## 问题背景

全注意力在序列长度 $N$ 下需要 `O(N^2)` 时间和每层 `O(N)` 的 KV cache。在 64k token 时，计算量和显存带宽数据极其严重。NSA 论文中的实测理论估算：在 64k 时注意力占总解码延迟的 70-80%。下游所有指标——TTFT、tokens/sec、每百万 token 成本——均由注意力成本主导。

稀疏注意力是显然的答案。此前的尝试可分为两类。固定模式稀疏（滑动窗口、步长跳采、块内局部）会丢弃信息，在长程回忆任务上失败。推理时稀疏（KV cache 剪枝、H2O、StreamingLLM）应用于在密集注意力下预训练的模型，只能回收部分潜在加速，因为模型从未被要求通过稀疏模式路由信息。

Native Sparse Attention（Yuan et al.，DeepSeek + PKU + UW，ACL 2025 最佳论文，arXiv:2502.11089）两者兼顾：模型在预训练期间学习的稀疏模式，以 kernel 对齐的方式实现，推理时真正交付计算节省。两年后，NSA 或其直接后继将成为每个前沿长上下文模型默认使用的注意力。

## 概念

### 三条并行分支

对每个 query，NSA 运行三次注意力，分别作用于 KV cache 的三个不同视图：

1. **压缩分支（Compressed branch）**。token 被分组为大小为 `l` 的块（通常为 32 或 64）。每个块通过一个小 MLP 压缩为单个摘要 token。query 对这些压缩 token 做注意力，获得对整个序列的粗粒度视图。

2. **选择分支（Selected branch）**。利用压缩分支的注意力得分，识别与当前 query 最相关的 top-k 个块。读取这些块中的细粒度（未压缩）token，query 对所有它们做注意力。可以把压缩分支的注意力作为选择的路由信号。

3. **滑动窗口分支（Sliding-window branch）**。query 对最近 `W` 个 token（通常为 512）做注意力以获取局部上下文。该分支捕获结构密集型的短程模式（句法、局部共指），其他两个分支可能遗漏这些模式。

三条分支的输出通过一个学习得到的 per-position 门控组合：

```
out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win
```

`g_cmp, g_sel, g_win` 是 query 上小 MLP 输出的门控权重。它们不需要求和为 1——可以独立加权各分支。

### 为何这是"原生可训练"

选择步骤（top-k 块）是离散的。离散操作会阻断梯度流。此前稀疏注意力工作要么跳过选择步骤的反向传播（限制训练），要么使用不能在实际推理中获得真实稀疏性的连续松弛。

NSA 规避了这个问题：压缩分支的注意力本身就是整个序列上可微分的粗粒度注意力。top-k 操作只是复用压缩分支中的最高注意力得分来选择加载哪些细粒度块。梯度可以通过压缩分支得分流动（同时影响压缩输出和选择逻辑），选中块的贡献到最终输出也是可微的。不可微分的 `top_k` 操作对前向计算图来说相当于无操作——它只控制从内存加载哪些块。

这就是 NSA 可以端到端用于预训练的原因。模型学习到联合通过三条分支路由信息，产生一个在推理时实际交付承诺加速的稀疏模式。

### 硬件对齐 Kernel

NSA 的 kernel 针对现代 GPU 内存层次设计。kernel 按 GQA group（外层循环）加载 query，对内层循环按 group 获取对应的稀疏 KV block，并在 SRAM 中运行注意力。因为每个 query group 看到相同的选中块（选择是按 query-group 而非按 query-head），KV 加载开销摊到组内。算术强度保持在高位。

论文报告 Triton kernel 在 64k 解码时比 FlashAttention 快 9 倍，且加速比随序列长度增长。正向和反向 kernel 均已提供。

### 计算预算

设 `N` 为序列长度，`l` 为压缩块大小，`k` 为 top-k 选择数，`w` 为滑动窗口，`b` 为选中块大小（通常等于 `l`）。

- 压缩分支：每 query `O(N/l)` 个 key，总计 `O(N * N / l)`。
- 选择分支：每 query `O(k * b)` 个 key，总计 `O(N * k * b)`。
- 滑动分支：每 query `O(w)` 个 key，总计 `O(N * w)`。

总计：`O(N * (N/l + k*b + w))`。

取 `N = 64k, l = 64, k = 16, b = 64, w = 512`：每 query 代价为 `1000 + 1024 + 512 = 2536 keys`。全注意力为 `64000 keys`。25 倍计算节省。

取 `N = 128k, l = 64, k = 16, b = 64, w = 512`：每 query 代价为 `2000 + 1024 + 512 = 3536 keys`。全注意力为 `128000 keys`。36 倍节省。收益随序列长度增长，这正是设计的核心目标。

### 对比

| 方法 | 可微分 | 真实推理加速 | 长程回忆 |
|--------|---------------|----------------------|-------------------|
| 仅滑动窗口 | 是 | 是 | 失败 |
| 步长 / 块稀疏 | 是 | 是 | 部分 |
| KV 剪枝 (H2O, StreamingLLM) | 不适用（推理时） | 是 | 部分 |
| MoBA (Moonshot) | 部分 | 是 | 良好 |
| NSA | 是（原生） | 是（64k 时 9x） | 匹配全注意力 |

MoBA（Moonshot，arXiv:2502.13189）同期发表，采用类似的"三比一好"思路，将 MoE 原则应用到注意力块上。NSA 和 MoBA 是 2026 年长上下文预训练必须关注的两种架构。

```figure
sliding-window-attention
```

## 构建

`code/main.py` 在短合成序列上实现三条分支，展示：

- 压缩 MLP（此处为教学清晰使用了简单 mean-pool 基线；真实 NSA 使用学习到的 MLP）。
- 由压缩分支得分驱动的 top-k 块选择。
- 在末尾 `w` 个 token 上的滑动窗口注意力。
- 门控组合。
- 与全注意力的计算计数对比。

### 步骤 1：将 token 压缩成块

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### 步骤 2：压缩分支注意力

用 query 对压缩后的 key 运行 softmax 注意力。压缩分支得分同时充当 top-k 选择的信号。

### 步骤 3：top-k 块选择

选出得分最高的 `k` 个压缩块索引。从这些块中加载原始未压缩 token，并对它们运行注意力。

### 步骤 4：滑动窗口注意力

取最后 `w` 个 token，对其运行标准注意力。

### 步骤 5：门控 + 合并

query 上的小 MLP 产生三个门控权重。最终输出为三个分支输出的加权和。

### 步骤 6：计算计数

打印每个分支每 query 注意的 key 数和总计。与 `N`（全注意力）对比。在 `l = 32, k = 4, w = 128` 的 1024 token 合成数据上，NSA 每 query 看到 `32 + 128 + 128 = 288` keys，而全注意力为 1024 —— 减少 3.5 倍。

## 使用

NSA 已在 DeepSeek 自身的长上下文预训练管线中部署。截至 2026 年 4 月，在公开推理框架中的集成状态：

- **DeepSeek 内部**：原生支持，发布权重的模型使用 NSA 或其后继 DSA（Deepseek Sparse Attention）。
- **vLLM**：为 DeepSeek-V3.x 权重开发实验性 NSA 支持。
- **SGLang**：已发布 NSA benchmark；生产路径跟随 vLLM。
- **llama.cpp / CPU**：不支持；kernel 分解开销在 CPU 吞吐量下不值得。

何时选用 NSA：

- 针对 64k 以上上下文、拥有可观计算预算的预训练或持续训练任务。
- 推理 DeepSeek 自身的长上下文 checkpoint。权重为 NSA 原生。

何时不使用：

- 服务已预训练好的密集注意力模型。不经过持续训练无法 retrofit NSA。
- 上下文低于 16k。三条分支的开销会超过节省。
- Batch-1 交互式对话。延迟敏感场景下虽有收益，但仅限长上下文。

## 交付物

本课产出 `outputs/skill-nsa-integrator.md`。给定一个长上下文预训练任务规格，生成 NSA 集成方案：压缩块大小、top-k、滑动窗口、门控 MLP 宽度、kernel 选择，以及哪些长上下文评测能证明架构变更的合理性。

## 练习

1. 在 1024 token 合成数据上运行 `code/main.py`。对三组预设值扫过 `(l, k, w)` 并打印计算计数。找出在 needle-in-haystack 测试中以 95% 召回率对抗全注意力的最低每 query key 数预设。

2. 将 mean-pool 压缩器替换为一个小型学习 MLP（2 层，隐藏层 32）。在信号为块平均值的合成任务上训练它。衡量其在 held-out 数据上与 mean-pool 基线的 perplexity 差距。

3. 实现门控 MLP。它以 query 为输入输出三个标量。展示门控行为合理：随机 query 上接近均匀加权；当 query 命中一个远端块时，选中分支获得高权重。

4. 计算 128k 上下文下 NSA 使能的 70B 模型的 KV cache 显存预算。KV heads 为 8，head dim 为 128，BF16。与全注意力对比，并与 MLA（Phase 10 · 14 展示了 MLA 数据）对比。找出 NSA 细粒度分支 KV cache 等于全注意力的序列长度临界点。

5. 阅读 NSA 论文第 4 节（arXiv:2502.11089），用三句话解释为何压缩分支的注意力得分被复用于 top-k 选择，而非单独计算路由得分。将答案与梯度流关联。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|------------------------|
| 压缩分支 | "粗粒度视图" | 对块均值 key 的注意力，为每 query 提供 O(N/l) 规模的键以实现全局上下文 |
| 选择分支 | "Top-k 块" | 对 top-k 个得分最高压缩块的细粒度注意力 |
| 滑动窗口 | "局部上下文" | 对最后 W 个 token 的注意力，捕获短程模式 |
| 原生可训练 | "在稀疏模式下预训练" | 稀疏模式在预训练中学习，而非推理时外挂 |
| 压缩块大小 l | "粗粒度组大小" | 多少个 token 合并为一个摘要；通常 32-64 |
| Top-k | "保留的块数" | 选中压缩块数，读取其未压缩 token；通常 16 |
| 滑动窗口 W | "局部注意力半径" | 通常 512；太短损害局部连贯性，太长浪费计算 |
| 分支门控 | "如何混合三条" | per-position MLP 输出，对三条分支贡献加权 |
| 硬件对齐 | "对 kernel 友好的稀疏" | 选择的稀疏模式使实际 GPU kernel 能达到理论加速 |
| DSA | "NSA 的后继" | DeepSeek Sparse Attention，NSA 之后 DeepSeek 谱系中的架构 |

## 延伸阅读

- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) —— 论文
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) —— NSA 目标应用的架构族
- [Moonshot AI — MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189) —— 同期工作，基于 MoE 思想对块做注意力
- [Beltagy et al. — Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150) —— 滑动窗口的起源
- [Xiao et al. — StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453) —— NSA 改进的推理时稀疏基线
- [Dao et al. — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) —— 全注意力基线，NSA kernel 在 64k 时超越它
