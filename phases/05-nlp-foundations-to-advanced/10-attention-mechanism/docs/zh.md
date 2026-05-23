# 注意力机制 — 突破性进展

> 解码器不再盯着压缩的摘要，而是开始审视整个源序列。从此往后的一切，都是注意力机制加上工程实践。

**类型：** 构建
**语言：** Python
**先修知识：** 阶段 5 · 09（序列到序列模型）
**时间：** 约 45 分钟

## 问题所在

第 09 课以一个有限的失败告终。在一个玩具复制任务上训练的 GRU 编码器-解码器，其准确率在序列长度为 5 时达到 89%，但在长度为 80 时几乎降至随机水平。原因在于模型结构，而非训练错误：编码器提取的所有信息都必须压缩到一个固定大小的隐藏状态中，而解码器永远看不到其他信息。

Bahdanau、Cho 和 Bengio 在 2014 年提出了一个三行代码的修复方案。他们不再仅给解码器最终的编码器状态，而是保留所有编码器状态。在解码器的每一步，计算所有编码器状态的加权平均，其中权重表示"解码器此刻需要多大程度地关注编码器位置 `i`？" 这个加权平均就是上下文向量，并且它在解码器的每一步都会变化。

这就是全部的核心思想。Transformer 模型对其进行了扩展。自注意力机制将其应用于单个序列。多头注意力机制并行运行它。但 2014 年的版本已经打破了瓶颈，一旦理解了它，向 Transformer 的过渡就主要是工程问题，而非概念性的飞跃。

## 核心概念

![Bahdanau 注意力：解码器查询所有编码器状态](../assets/attention.svg)

在解码器的每一步 `t`：

1.  使用前一个解码器隐藏状态 `s_{t-1}` 作为**查询**。
2.  将其与每个编码器隐藏状态 `h_1, ..., h_T` 进行比较评分。每个编码器位置得到一个标量分数。
3.  对分数进行 Softmax 归一化，得到总和为 1 的注意力权重 `α_{t,1}, ..., α_{t,T}`。
4.  计算上下文向量 `c_t = Σ α_{t,i} * h_i`。这是编码器状态的加权平均。
5.  解码器将 `c_t` 与前一个输出的 token 结合，产生下一个 token。

加权平均是关键。当解码器需要将 "Je" 翻译为 "I" 时，它会赋予对应 "Je" 的编码器状态很高的权重，而给其他状态很低的权重。当需要翻译 "not" 时，则赋予 "pas" 很高的权重。上下文向量在每一步都在重塑。

## 形状问题（最容易出错的地方）

这是每一个注意力实现第一次都会出错的地方。请仔细阅读。

| 项目 | 形状 | 备注 |
|-------|-------|-------|
| 编码器隐藏状态 `H` | `(T_enc, d_h)` | 如果是双向 LSTM，则为 `d_h = 2 * d_hidden` |
| 解码器隐藏状态 `s_{t-1}` | `(d_s,)` | 一个向量 |
| 注意力分数 `e_{t,i}` | 标量 | 每个编码器位置一个 |
| 注意力权重 `α_{t,i}` | 标量 | 对所有 `i` 进行 Softmax 后得到 |
| 上下文向量 `c_t` | `(d_h,)` | 与单个编码器状态形状相同 |

**Bahdanau (加性) 评分。** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

-   `s_{t-1}` 形状为 `(d_s,)`，`h_i` 形状为 `(d_h,)`。
-   `W_a` 形状为 `(d_attn, d_s)`。`U_a` 形状为 `(d_attn, d_h)`。
-   它们在 tanh 内部的和形状为 `(d_attn,)`。
-   `v_α` 形状为 `(d_attn,)`。与 `v_α` 的内积运算会压缩为一个标量。**这就是 `v_α` 所做的事情。** 这不是魔法。它只是一个投影，将注意力维度向量转换为标量分数。

**Luong (乘性) 评分。** 三种变体：

-   `dot`：`e_{t,i} = s_t^T * h_i`。要求 `d_s == d_h`。这是硬性约束。如果你的编码器是双向的，请跳过此方法。
-   `general`：使用 `W`（形状为 `(d_s, d_h)`）的 `e_{t,i} = s_t^T * W * h_i`。消除了维度相等的限制。
-   `concat`：本质上就是 Bahdanau 形式。由于前两种更高效，此形式很少使用。

**一个值得命名的 Bahdanau / Luong 陷阱。** Bahdanau 使用 `s_{t-1}`（即生成当前词 *之前* 的解码器状态）。Luong 使用 `s_t`（即 *之后* 的状态）。混淆它们会产生微妙的错误梯度，极难调试。选定一篇论文并坚持其约定。

## 动手实现

### 步骤 1：加性 (Bahdanau) 注意力

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

对照上表检查你的形状。`encoder_states` 形状为 `(T_enc, d_h)`。`projected_enc` 形状为 `(T_enc, d_attn)`。`projected_dec` 形状为 `(d_attn,)` 并具有广播特性。`combined` 形状为 `(T_enc, d_attn)`。`scores` 形状为 `(T_enc,)`。`weights` 形状为 `(T_enc,)`。`context` 形状为 `(d_h,)`。完成。

### 步骤 2：Luong 点积和通用形式

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

每种只有三行。这就是 Luong 的论文引起关注的原因。在大多数任务上准确率相同，但代码量少得多。

### 步骤 3：一个数值计算示例

给定三个编码器状态（大致对应 "cat", "sat", "mat"）和一个与第一个状态对齐程度最高的解码器状态，注意力分布将集中在位置 0。如果解码器状态转移到与最后一个对齐，则注意力会移到位置 2。上下文向量随之跟踪。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

第一行胜出。然后移动解码器状态使其更接近第三个编码器状态，观察权重如何转移。就是这样。注意力机制就是显式的对齐。

### 步骤 4：为何这是通往 Transformer 的桥梁

将上述语言转化为 Q/K/V：

-   **查询 (Query)** = 解码器状态 `s_{t-1}`
-   **键 (Key)** = 编码器状态（我们与之评分比较的对象）
-   **值 (Value)** = 编码器状态（我们加权和求和的对象）

在经典的注意力中，键和值是同一个东西。自注意力机制将它们分离：你可以让一个序列查询自身，并为 K 和 V 使用不同的学习投影。多头注意力机制使用不同的学习投影并行运行。Transformer 模型堆叠整个阶段多次并抛弃了 RNN。

数学是相同的。形状是相同的。从 Bahdanau 注意力到缩放点积注意力的教学跃进主要在于符号表示。

## 实际应用

PyTorch 和 TensorFlow 直接提供了注意力模块。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

这就是一个 Transformer 注意力层。查询（Query）批次包含 5 个位置，键/值（Key/Value）批次包含 10 个位置，每个位置 128 维，8 个头。`output` 是新增的、经过上下文增强的查询。`weights` 是你可以可视化的 5x10 对齐矩阵。

### 经典注意力仍然重要的场景

-   **教学。** 单头、单层、基于 RNN 的版本让每个概念都清晰可见。
-   **设备端序列任务**，当 Transformer 模型无法适配时。
-   **2014-2017 年的任何论文。** 如果不了解 Bahdanau 的约定，你会误读它们。
-   **机器翻译中的细粒度对齐分析。** 即使在 Transformer 模型上，原始注意力权重也是一个可解释性工具，理解它们需要知道它们代表什么。

### 注意力权重作为解释的陷阱

注意力权重看起来很可解释。它们是总和为 1 的权重，可以绘制图表；权重高意味着"关注了这里"。审稿人喜欢它们。

但它们并不像看起来那么可解释。Jain 和 Wallace (2019) 表明，在某些任务上，注意力分布可以被排列或替换为任意替代方案，而不会改变模型的预测。在没有消融实验或反事实检查的情况下，绝不要将注意力权重作为推理的证据。

## 提交作业

保存为 `outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## 练习

1.  **简单。** 实现 `softmax` 掩码，使编码器中的填充 token 获得零注意力权重。在一个包含变长序列的批次上进行测试。
2.  **中等。** 为 Luong `general` 形式添加多头注意力。将 `d_h` 分成 `n_heads` 组，为每个头运行注意力，然后拼接。验证单头情况与你之前的实现一致。
3.  **困难。** 使用 Bahdanau 注意力在第 09 课的玩具复制任务上训练一个 GRU 编码器-解码器。绘制准确率随序列长度变化的曲线。与无注意力的基线进行比较。你应该会看到随着长度增加，差距变大，从而确认注意力机制消除了瓶颈。

## 关键术语

| 术语 | 人们常说 | 其实际含义 |
|------|-----------------|-----------------------|
| 注意力 (Attention) | 关注事物 | 一个值序列的加权平均，权重由查询-键的相似度计算得出。 |
| 查询、键、值 (Query, Key, Value) | QKV | 三种投影：Q 发起询问，K 是用于匹配的对象，V 是返回的对象。 |
| 加性注意力 (Additive attention) | Bahdanau | 前馈评分：`v^T tanh(W q + U k)`。 |
| 乘性注意力 (Multiplicative attention) | Luong 点积 / 通用形式 | 评分是 `q^T k` 或 `q^T W k`。更高效，在大多数任务上准确率相同。 |
| 对齐矩阵 (Alignment matrix) | 那张漂亮的图 | 注意力权重组成的 `(T_dec, T_enc)` 网格。通过查看它来了解模型关注了什么。 |

## 扩展阅读

-   [Bahdanau, Cho, Bengio (2014). 通过联合学习对齐与翻译的神经机器翻译](https://arxiv.org/abs/1409.0473) — 原始论文。
-   [Luong, Pham, Manning (2015). 有效的基于注意力的神经机器翻译方法](https://arxiv.org/abs/1508.04025) — 三种评分变体及其比较。
-   [Jain and Wallace (2019). 注意力并非解释](https://arxiv.org/abs/1902.10186) — 关于可解释性的警告。
-   [动手学深度学习 — Bahdanau 注意力](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — 带有 PyTorch 的可运行教程。