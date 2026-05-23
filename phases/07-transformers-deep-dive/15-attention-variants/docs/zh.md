# 注意力变体 — 滑动窗口、稀疏、微分

> 完全注意力是一个圆。每个token都能看到每个token，而内存为此付出代价。四种变体弯曲了这个圆形，收回了一半的成本。

**类型：** 构建
**语言：** Python
**前提条件：** 阶段7 · 02（自注意力），阶段7 · 03（多头注意力），阶段7 · 12（KV缓存 / Flash注意力）
**时间：** 约60分钟

## 问题所在

完全注意力在序列长度上的内存开销为 `O(N²)`，计算开销为 `O(N²)`。对于一个128K上下文的Llama 3 70B模型，这意味着每层160亿个注意力条目，乘以80层。Flash注意力（第12课）隐藏了 `O(N²)` 激活内存，但并未改变算术成本——每个token仍然要关注所有其他token。

三类变体改变了注意力矩阵本身的拓扑结构：

1. **滑动窗口注意力（SWA）。** 每个token只关注一个固定窗口内的邻居，而不是整个前缀。内存和计算开销降至 `O(N · W)`，其中 `W` 是窗口大小。Gemma 2/3、Mistral 7B的前几层、Phi-3-Long。
2. **稀疏/块状注意力。** 只有选定的对 `(i, j)` 会被评分；其余的被强制赋予零权重。Longformer、BigBird、OpenAI稀疏transformer。
3. **微分注意力。** 使用独立的Q/K投影计算两个注意力图，然后从一个中减去另一个。消除了将权重泄露到前几个token的“注意力汇聚”现象。微软的DIFF Transformer（2024年）。

这些变体可以共存。一个2026年的前沿模型通常会混合使用它们：大多数层是SWA-1024，每五层有一个全局完全注意力层，以及少量用于清理检索的微分注意力头。Gemma 3的5:1 SWA与全局注意力比率是当前教科书默认的配置。

## 核心概念

### 滑动窗口注意力（SWA）

位于位置 `i` 的每个查询只关注 `[i - W, i]`（因果SWA）或 `[i - W/2, i + W/2]`（双向）中的位置。窗口外的token在评分矩阵中获得 `-inf`。

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

对于 `N = 8192` 和 `W = 1024`，评分矩阵预期有1024 × 8192个非零行——减少了8倍。

**KV缓存随SWA缩小。** 每层只需要保留K和V的最后 `W` 个token。对于一个类似Gemma-3的配置（1024窗口，128K上下文），KV缓存减少了128倍。

**质量代价。** 纯SWA的transformer在长程检索上表现不佳。解决方法：将SWA层与全注意力层交错排列。Gemma 3使用5:1的SWA与全局注意力比例。Mistral 7B使用了一个因果SWA堆栈，信息通过重叠窗口“向前流动”——每层将有效感受野扩展 `W`，经过 `L` 层后，模型可以关注到 `L × W` 个token之前。

### 稀疏/块状注意力

预先选择一个 `N × N` 稀疏模式。三种经典模式：

- **局部+步进（OpenAI稀疏transformer）。** 关注最后的 `W` 个token，加上在此之前的每 `stride` 个token。以 `O(N · sqrt(N))` 的计算成本同时捕获局部和长程依赖。
- **Longformer / BigBird。** 局部窗口 + 一小部分全局token（例如 `[CLS]`），它们关注所有token并被所有token关注 + 随机稀疏链接。在匹配的质量下，经验上可处理2倍的上下文。
- **原生稀疏注意力（DeepSeek, 2025）。** 学习哪些 `(Q, K)` 块重要；在内核层面跳过零块。兼容FlashAttention。

稀疏注意力是一个内核工程的故事。数学很简单（掩码评分矩阵）；优势来自永不将零条目加载到SRAM。FlashAttention-3和2026年的FlexAttention API使得自定义稀疏模式在PyTorch中成为一等公民。

### 微分注意力（DIFF Transformer, 2024）

常规注意力存在“注意力汇聚”问题：softmax强制每行之和为1，因此不想关注任何特定内容的token会将权重倾倒在第一个token（或前几个）上。这窃取了本应用于真正内容的能力。

微分注意力通过计算**两个**注意力图并相减来解决这个问题：

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中 `λ` 是一个可学习的标量（通常为0.5–0.8）。A1捕获真正的内容权重；A2捕获汇聚效应。相减操作抵消了汇聚效应，将权重重新分配给相关的token。

报告的结果（微软 2024年）：困惑度降低5–10%，在相同训练长度下有效上下文长度延长1.5–2倍，更敏锐的“大海捞针”检索能力。

### 变体对比

| 变体 | 计算量 | KV缓存 | 相比完全注意力的质量 | 生产使用 |
|---------|---------|----------|-----------------|----------------|
| 完全注意力 | O(N²) | 每层 O(N) | 基准 | 每个模型的默认层 |
| SWA（窗口1024） | O(N·W) | 每层 O(W) | -0.1困惑度，配合全局层效果好 | Gemma 2/3, Phi-3-Long |
| 局部+步进稀疏 | O(N·√N) | 混合 | 与SWA相似 | OpenAI稀疏transformer, Longformer |
| BigBird（局部+全局+随机） | 近似 O(N) | 混合 | 在2倍上下文下匹配完全注意力 | 早期长上下文BERT |
| 原生稀疏（DeepSeek-V3.2） | O(N · 活跃比例) | O(N) | 差异在0.05困惑度以内 | DeepSeek-V3.2, 2025 |
| 微分 | O(2·N²) | O(2N) | -5% 到 -10% 困惑度 | DIFF Transformer, 2026早期模型 |

## 动手构建

参见 `code/main.py`。我们实现一个因果掩码比较器，在一个玩具序列上并排展示完全注意力、SWA、局部+步进和微分注意力。

### 步骤1：完全因果掩码（基准）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

来自第07课的基准。下三角矩阵；对角线以上权重为零。

### 步骤2：滑动窗口因果掩码

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

一个参数—— `window`。当 `window >= n` 时，恢复为完全因果注意力。当 `window = 1` 时，每个token只关注自身。

### 步骤3：局部+步进稀疏掩码

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

密集的局部窗口加上从序列开始每 `stride` 个token。随着层数增加，感受野以对数步长增长。

### 步骤4：微分注意力

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次注意力计算，用一个可学习的混合系数相减。在代码中，我们比较单次注意力与微分注意力的注意力汇聚热图，并观察汇聚效应的消失。

### 步骤5：KV缓存大小

打印在 `N = 131072` 下每种变体每层的缓存大小。SWA和稀疏变体减少10–100倍。微分注意力加倍。请有意识地支付你的内存账单。

## 应用实践

2026年生产模式：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+中的FlexAttention接受一个掩码函数：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

这会编译成一个自定义的Triton内核。对于常见模式，速度接近FlashAttention-3的90%，且掩码函数是一个Python可调用对象。

**何时选择哪种：**

- **纯完全注意力** — 用于上下文长度不超过约16K的每一层，或者当检索质量至关重要时。
- **SWA + 全局混合** — 长上下文（>32K），训练和推理受内存限制。2026年超过32K上下文的默认选择。
- **稀疏块状注意力** — 自定义内核，自定义模式。保留用于专门的工作负载（检索、音频）。
- **微分注意力** — 任何受注意力汇聚污染影响的工作负载（长上下文RAG，大海捞针）。

## 交付部署

参见 `outputs/skill-attention-variant-picker.md`。该技能根据目标上下文长度、检索需求以及训练/推理计算配置，为一个新模型选择注意力拓扑。

## 练习

1. **简单。** 运行 `code/main.py`。验证SWA在 `window=4` 下，每行最后4个token之外的值全为零。验证 `window=n` 能够比特级精确地复现完全因果注意力。
2. **中等。** 在第07课的最终项目之上实现带有 `window=1024` 的因果SWA。在tinyshakespeare数据集上训练1,000步。验证损失相比完全注意力回归了多少？峰值内存下降了多少？
3. **困难。** 在最终项目模型中实现一个Gemma-3风格的5:1层混合（5层SWA，1层全局）。在匹配参数的情况下，比较损失、内存和生成质量与纯SWA和纯全局基线的差异。
4. **困难。** 实现带有每头可学习 `λ` 的微分注意力。在一个合成检索任务（一个目标，2,000个干扰项）上训练。测量在匹配参数下，相比单次注意力基线的检索准确率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 滑动窗口注意力（SWA） | “局部注意力” | 每个查询关注其最后 `W` 个token；KV缓存缩小至 `O(W)`。 |
| 有效感受野 | “模型能看到多远” | 在一个带有窗口 `W` 的 `L` 层SWA堆栈中，最多 `L × W` 个token。 |
| Longformer / BigBird | “局部+全局+随机” | 具有少量始终参与注意力的全局token的稀疏模式；早期的长上下文方法。 |
| 原生稀疏注意力 | “DeepSeek的内核技巧” | 学习块级稀疏性；在内核层面跳过零块，同时保持质量。 |
| 微分注意力 | “两个图，一个相减” | DIFF Transformer：从第一个注意力图中减去一个可学习的 `λ` 倍第二个注意力图，以消除注意力汇聚。 |
| 注意力汇聚 | “权重泄露到token 0” | Softmax归一化强制每行之和为1；无信息量的查询将权重倾倒在位置0上。 |
| FlexAttention | “掩码即Python” | PyTorch 2.5+ API，将任意掩码函数编译成FlashAttention形状的内核。 |
| 层类型混合 | “5:1 SWA对全局” | 在堆栈中交错稀疏和完全注意力层，以在较低内存下保持质量。 |

## 扩展阅读

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — 规范的滑动窗口+全局token论文。
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — 局部+全局+随机。
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI的局部+步进模式。
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1的SWA:全局注意力混合。
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — 窗口=1024的5:1混合，现在是教科书默认配置。
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer论文。
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2的可学习稀疏注意力。
- [PyTorch — FlexAttention博客和文档](https://pytorch.org/blog/flexattention/) — 应用实践部分中掩码即可调用模式的API参考。