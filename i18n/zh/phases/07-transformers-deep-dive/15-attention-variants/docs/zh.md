# 注意力变体 —— 滑动窗口、稀疏、差分

> 全注意力是一个圆。每个 token 都能看到所有 token，而内存为此买单。四种变体弯曲这个圆的形状，收回一半成本。

**类型：** 构建
**语言：** Python
**前置知识：** 第 7 阶段 · 02（自注意力）、第 7 阶段 · 03（多头注意力）、第 7 阶段 · 12（KV 缓存 / Flash Attention）
**时间：** 约 60 分钟

## 问题所在

全注意力的内存和计算量随序列长度呈 `O(N²)` 增长。对于支持 128K 上下文的 Llama 3 70B，每层有 160 亿个注意力条目，共 80 层。Flash Attention（第 12 课）隐藏了 `O(N²)` 的激活内存，但并未改变计算成本——每个 token 仍然 attends 所有其他 token。

三类变体改变了注意力矩阵本身的拓扑结构：

1. **滑动窗口注意力（SWA）。** 每个 token 只关注一个固定大小的邻居窗口，而非完整的前缀。内存和计算量降至 `O(N · W)`，其中 `W` 为窗口大小。Gemma 2/3、Mistral 7B 的前几层、Phi-3-Long 均采用。
2. **稀疏 / 块注意力。** 仅对选定的 `(i, j)` 对进行评分；其余位置权重强制为零。Longformer、BigBird、OpenAI 稀疏 Transformer。
3. **差分注意力。** 用独立的 Q/K 投影分别计算两个注意力图，再相减。消除将权重"泄漏"到前几个 token 的"注意力 sink"现象。微软 DIFF Transformer（2024）。

这些变体可以共存。2026 年的前沿模型通常混合使用它们：大部分层采用 SWA-1024，每第五层是全局全注意力，少数层是负责清理检索结果的差分头。Gemma 3 的 5:1 SWA 与全局比例是当前的教科书默认配置。

## 概念解析

### 滑动窗口注意力（SWA）

位置 `i` 处的每个 query 仅关注 `[i - W, i]` 范围内的位置（因果 SWA），或 `[i - W/2, i + W/2]` 范围（双向 SWA）。窗口外的 token 在分数矩阵中获得 `-inf`。

```
全因果注意力：             滑动窗口 (W=4)：
位置 0-7                   位置 0-7, W=4
    0 1 2 3 4 5 6 7           0 1 2 3 4 5 6 7
0 | x                   0 |  x
1 | x x                 1 |  x x
2 | x x x               2 |  x x x
3 | x x x x             3 |  x x x x
4 | x x x x x           4 |    x x x x
5 | x x x x x x         5 |      x x x x
6 | x x x x x x x       6 |        x x x x
7 | x x x x x x x x     7 |          x x x x
```

当 `N = 8192`、`W = 1024` 时，分数矩阵期望有 1024 × 8192 个非零行——8 倍缩减。

**SWA 使 KV 缓存缩小。** 每层只需保留 K 和 V 的最后 `W` 个 token。对于类 Gemma-3 配置（1024 窗口，128K 上下文），KV 缓存缩减 128 倍。

**质量代价。** 纯 SWA Transformer 在长程检索上存在困难。解决方案：用全注意力层交叉排列 SWA 层。Gemma 3 采用 5:1 的 SWA:全局比例。Mistral 7B 使用了因果 SWA 堆叠，使信息通过重叠窗口"向前流动"——每层将有效感受野扩展 `W`，经过 `L` 层后模型可以回望 `L × W` 个 token。

### 稀疏 / 块注意力

预先选定一个 `N × N` 的稀疏模式。三种典型形状：

- **局部 + 步长（OpenAI 稀疏 Transformer）。** 关注最后 `W` 个 token，再加上每隔 `stride` 个 token 的序列。以 `O(N · sqrt(N))` 的计算量同时捕获局部和长程依赖。
- **Longformer / BigBird。** 局部窗口 + 少量全局 token（如 `[CLS]`）与所有 token 相互关注 + 随机稀疏链接。实证表明在相同质量下上下文长度翻倍。
- **原生稀疏注意力（DeepSeek，2025）。** 学习哪些 `(Q, K)` 块是关键；在 kernel 级别跳过零块。兼容 FlashAttention。

稀疏注意力是一道内核工程题。数学很简单（对分数矩阵加掩码）；真正的收益来自从不将零条目加载到 SRAM 中。FlashAttention-3 和 2026 年的 FlexAttention API 使自定义稀疏模式成为 PyTorch 的一等公民。

### 差分注意力（DIFF Transformer，2024）

标准注意力存在"注意力 sink"问题：softmax 强制每行和为 1，因此那些不想关注任何特定内容的 token 会把权重 dump 到第一个 token（或前几个 token）。这会挤占本应用于实质内容的能力。

差分注意力通过计算**两个**注意力图并相减来解决这个问题：

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中 `λ` 是可学习标量（通常 0.5–0.8）。A1 捕捉真实内容权重；A2 捕捉 sink 模式。相减后抵消 sink，将权重重新分配给相关 token。

报告结果（微软 2024）：困惑度降低 5–10%，相同训练长度下有效上下文延长 1.5–2 倍，"大海捞针"检索更精准。

### 变体对比

| 变体 | 计算量 | KV 缓存 | 相对全注意力的质量 | 生产用途 |
|------|--------|---------|-------------------|----------|
| 全注意力 | O(N²) | 每层 O(N) | 基线 | 所有模型的默认层 |
| SWA（窗口 1024） | O(N·W) | 每层 O(W) | -0.1 ppl，配合全局层效果良好 | Gemma 2/3、Phi-3-Long |
| 局部 + 步长稀疏 | O(N·√N) | 混合 | 与 SWA 相当 | OpenAI 稀疏 Transformer、Longformer |
| BigBird（局部 + 全局 + 随机） | 约 O(N) | 混合 | 2× 上下文下与全注意力持平 | 早期长上下文 BERT |
| 原生稀疏（DeepSeek-V3.2） | O(N · 活跃比例) | O(N) | 0.05 ppl 以内 | DeepSeek-V3.2、2025 |
| 差分 | O(2·N²) | O(2N) | -5% 至 -10% ppl | DIFF Transformer、2026 早期模型 |

```figure
gqa-kv-sharing
```

## 动手实现

见 `code/main.py`。我们实现一个因果掩码对比器，在玩具序列上并排展示全注意力、SWA、局部 + 步长和差分注意力。

### 步骤 1：全因果掩码（基线）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

来自第 7 课的基线。下三角矩阵；对角线以上权重为零。

### 步骤 2：滑动窗口因果掩码

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

一个参数——`window`。当 `window >= n` 时，恢复全因果注意力。当 `window = 1` 时，每个 token 只关注自身。

### 步骤 3：局部 + 步长稀疏掩码

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

密集局部窗口加上每隔 `stride` 个 token 回溯到序列起始。随着层数增加，感受野以对数步长增长。

### 步骤 4：差分注意力

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次注意力前向传播，用可学习的混合系数相减。代码中对比单注意力与差分注意力的注意力 sink 热图，观察 sink 如何坍缩。

### 步骤 5：KV 缓存大小

在 `N = 131072` 时为每种变体打印每层的缓存大小。SWA 和稀疏变体减少 10–100 倍。差分变体翻倍。有意识地支付你的内存账单。

## 实际使用

2026 年生产模式：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 以 5:1 比例混合 SWA（窗口=1024）和全局层。
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 的 FlexAttention 接受掩码函数：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

这会编译为自定义 Triton 内核。对于常见模式，速度接近 FlashAttention-3 的 90%，且掩码函数是一个 Python 可调用对象。

**何时选用哪种方案：**

- **纯全注意力** —— 每层上下文长达 ~16K，或对检索质量要求极高。
- **SWA + 全局混合** —— 长上下文（>32K），训练和推理内存受限。这是 2026 年在 32K 以上的默认方案。
- **稀疏块注意力** —— 自定义内核、自定义模式。保留用于特定工作负载（检索、音频）。
- **差分注意力** —— 任何注意力 sink 污染损害性能的工作负载（长上下文 RAG、大海捞针）。

## 交付

见 `outputs/skill-attention-variant-picker.md`。该技能根据目标上下文长度、检索需求和训练/推理计算画像，为新模型选择注意力拓扑结构。

## 练习

1. **简单。** 运行 `code/main.py`。验证 `window=4` 时 SWA 每行将最后 4 个 token 之外的位置全部置零。验证 `window=n` 时按位精确复现全因果注意力。
2. **中等。** 在第 7 课结业项目的基础上，用 `window=1024` 实现因果 SWA。在 tinyshakespeare 上训练 1,000 步。与全注意力相比，验证损失回退了多少？峰值内存下降了多少？
3. **困难。** 在结业模型中实现类 Gemma-3 的 5:1 层混合（5 层 SWA + 1 层全局）。在参数量匹配的前提下，将损失、内存和生成质量与纯 SWA 和纯全局基线进行比较。
4. **困难。** 实现带每 head 可学习 `λ` 的差分注意力。在合成检索任务（一根针、2000 个干扰项）上训练。在参数量匹配的前提下，将检索准确率与单注意力基线进行比较。

## 关键术语

| 术语 | 人们怎么说的 | 实际含义 |
|------|-------------|----------|
| 滑动窗口注意力（SWA） | "局部注意力" | 每个 query 只关注其最后 `W` 个 token；KV 缓存缩减至 `O(W)`。 |
| 有效感受野 | "模型能回溯多远" | 在窗口为 `W`、共 `L` 层的 SWA 堆叠中，最多可达 `L × W` 个 token。 |
| Longformer / BigBird | "局部 + 全局 + 随机" | 带少量始终参与注意力的全局 token 的稀疏模式；早期长上下文方案。 |
| 原生稀疏注意力 | "DeepSeek 的内核技巧" | 学习块级稀疏性；在 kernel 级别跳过零块同时保持质量。 |
| 差分注意力 | "两个图，一个相减" | DIFF Transformer：从第一个注意力图中减去可学习 `λ` 乘以第二个图，以消除注意力 sink。 |
| 注意力 sink | "权重泄漏到 token 0" | softmax 归一化强制行和为 1；无信息量的 query 将权重 dump 到位置 0。 |
| FlexAttention | "掩码即 Python" | PyTorch 2.5+ API，将任意掩码函数编译为类 FlashAttention 内核。 |
| 层类型混合 | "5:1 SWA 对全局" | 在堆叠中交错稀疏和全注意力层，以更低的内存保持质量。 |

## 延伸阅读

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) —— 经典的滑动窗口 + 全局 token 论文。
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) —— 局部 + 全局 + 随机。
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) —— OpenAI 的局部+步长模式。
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) —— 1:1 的 SWA:全局混合。
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) —— 5:1 混合配 1024 窗口，现已成为教科书默认配置。
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) —— DIFF Transformer 论文。
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) —— DeepSeek-V3.2 的学习式稀疏注意力。
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) —— "Use It" 章节中掩码即可调用的 API 参考。
