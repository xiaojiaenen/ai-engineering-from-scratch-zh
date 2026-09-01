```markdown
# 注意力机制——突破性进展

> 解码器不再眯着眼看一个压缩摘要，而是开始审视整个源序列。此后的一切都是注意力机制加上工程实现。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段5 · 第09课（序列到序列模型）
**耗时：** 约45分钟

## 问题所在

第09课以"有节制的失败"收尾。在玩具复制任务上训练的 GRU 编解码器，从长度5时的89%准确率，下降到长度80时的接近随机水平。原因是结构性的，而非训练bug：编码器获取的所有信息必须塞进一个固定大小的隐藏状态，而解码器永远看不到其他任何内容。

Bahdanau、Cho 和 Bengio 在2014年发表了一个三行的修复方案。不再只给解码器最终编码器状态，而是保留所有编码器状态。在每个解码步骤，计算编码器状态的加权平均，其中权重表示"解码器此刻需要多看多少编码器位置 `i`？"这个加权平均就是上下文向量，且它在每个解码步骤都会变化。

这就是全部思想。Transformer 将其扩展。自注意力将其应用于单一序列。多头注意力并行运行它。但2014年的版本已经打破了瓶颈，一旦有了它，转向 Transformer 就是工程问题，而非概念问题。

## 概念

![Bahdanau 注意力：解码器查询所有编码器状态](../assets/attention.svg)

在每个解码步骤 `t`：

1. 使用之前的解码器隐藏状态 `s_{t-1}` 作为**查询**。
2. 将它与所有编码器隐藏状态 `h_1, ..., h_T` 进行打分。每个编码器位置一个标量。
3. 对分数做 softmax 得到注意力权重 `α_{t,1}, ..., α_{t,T}`，它们之和为1。
4. 上下文向量 `c_t = Σ α_{t,i} * h_i`。编码器状态的加权平均。
5. 解码器接收 `c_t` 和上一个输出 token，生成下一个 token。

加权平均是核心所在。当解码器需要将"Je"翻译为"I"时，它会高分加权"Je"对应的编码器状态，其他状态低分加权。当它需要"not"时，高分加权"pas"。上下文向量每一步都在重塑。

## 形状（这是所有人踩坑的地方）

这是每个注意力实现第一次都会出错的地方。请仔细阅读。

| 内容 | 形状 | 备注 |
|-------|-------|-------|
| 编码器隐藏状态 `H` | `(T_enc, d_h)` | 若是 BiLSTM，`d_h = 2 * d_hidden` |
| 解码器隐藏状态 `s_{t-1}` | `(d_s,)` | 单个向量 |
| 注意力分数 `e_{t,i}` | 标量 | 每个编码器位置一个 |
| 注意力权重 `α_{t,i}` | 标量 | 对所有 `i` 做 softmax 后 |
| 上下文向量 `c_t` | `(d_h,)` | 与编码器状态形状相同 |

**Bahdanau（加性）分数。** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}` 形状为 `(d_s,)`，`h_i` 形状为 `(d_h,)`。
- `W_a` 形状为 `(d_attn, d_s)`，`U_a` 形状为 `(d_attn, d_h)`。
- 它们相加后进入 tanh 的形状为 `(d_attn,)`。
- `v_α` 形状为 `(d_attn,)`。与 `v_α` 的內积坍缩为一个标量。**这就是 `v_α` 的作用。** 它不是魔法。它是将注意力维度向量投影为标量分数的映射。

**Luong（乘性）分数。** 三种变体：

- `dot`：`e_{t,i} = s_t^T * h_i`。要求 `d_s == d_h`。硬性约束。如果编码器是双向的请跳过。
- `general`：`e_{t,i} = s_t^T * W * h_i`，`W` 形状为 `(d_s, d_h)`。消除了等维约束。
- `concat`：本质上是 Bahdanau 形式。由于前两种更廉价，很少使用。

**一个值得命名的 Bahdanau / Luong 陷阱。** Bahdanau 使用 `s_{t-1}`（生成当前词*之前*的解码器状态）。Luong 使用 `s_t`（*之后*的状态）。混淆它们会产生微妙的错误梯度，极其难以调试。选择一篇论文并坚持其约定。

```figure
attention-heatmap
```

## 动手实现

### 步骤1：加性（Bahdanau）注意力

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    # 投影解码器和编码器状态
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    # 相加并通过 tanh
    combined = np.tanh(projected_enc + projected_dec)
    # 投影到标量分数
    scores = combined @ v_a
    # 归一化为权重
    weights = softmax(scores)
    # 加权求和得到上下文向量
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

对照上面的表格检查你的形状。`encoder_states` 形状为 `(T_enc, d_h)`。`projected_enc` 形状为 `(T_enc, d_attn)`。`projected_dec` 形状为 `(d_attn,)` 并广播。`combined` 形状为 `(T_enc, d_attn)`。`scores` 形状为 `(T_enc,)`。`weights` 形状为 `(T_enc,)`。`context` 形状为 `(d_h,)`。提交它。

### 步骤2：Luong dot 和 general

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

每种只有三行。这就是 Luong 的论文能发表的原因。在大多数任务上准确率相同，代码量大幅减少。

### 步骤3：一个数值示例

给定三个编码器状态（大致对应"cat"、"sat"、"mat"），以及主要与第一个对齐的解码器状态，注意力分布会集中在位置0。如果解码器状态转移到与最后一个对齐，注意力会移到位置2。上下文向量随之跟踪。

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

第一行获胜。然后将解码器状态移向第三个编码器状态，观察权重如何移动。就是这样。注意力就是显式的对齐。

### 步骤4：为什么这是通往 Transformer 的桥梁

将上述语言映射为 Q/K/V：

- **查询（Query）** = 解码器状态 `s_{t-1}`
- **键（Key）** = 编码器状态（我们与之打分的对象）
- **值（Value）** = 编码器状态（我们加权求和的对象）

在经典注意力中，键和值是同一个东西。自注意力将它们分离：你可以让一个序列查询自身，并对 K 和 V 使用不同的学习投影。多头注意力并行运行它，使用不同的学习投影。Transformer 将整个过程堆叠多次并抛弃 RNN。

数学是一样的。形状是一样的。从 Bahdanau 注意力到缩放点积注意力的教学跳跃主要是符号上的差异。

## 使用它

PyTorch 和 TensorFlow 直接提供注意力机制。

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

这是一个 Transformer 注意力层。查询批量5个位置，键/值批量10个位置，每个128维，8个头。`output` 是新的上下文增强查询。`weights` 是你可以可视化的 5x10 对齐矩阵。

### 经典注意力仍然重要的场景

- 教学。单头、单层、基于 RNN 的版本使每个概念都清晰可见。
- 在 Transformer 不适合的终端设备序列任务中。
- 2014-2017年的任何论文。如果不了解 Bahdanau 的约定，你会误读它。
- 机器翻译中的细粒度对齐分析。原始注意力权重是可解释性工具，即使在 Transformer 模型上也如此，而阅读它们需要知道它们是什么。

### 注意力权重即解释的陷阱

注意力权重看起来可解释。它们是跨位置之和为1的权重；你可以绘制它们；高权重意味着"看了这里"。审稿人很喜欢它们。

它们不像看起来那样可解释。Jain 和 Wallace（2019）表明，注意力分布可以被置换并由任意替代方案替换，而不会改变某些任务上的模型预测。永远不要在没有消融实验或反事实检查的情况下报告注意力权重作为推理证据。

## 交付

保存为 `outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: 调试注意力实现中的形状 bug。
phase: 5
lesson: 10
---

给定一个有缺陷的注意力实现，你要识别出形状不匹配。输出：

1. 哪个矩阵形状错误。命名该张量。
2. 它应该具有的形状，由 (d_s, d_h, d_attn, T_enc, T_dec, batch_size) 推导。
3. 一行修复。转置、reshape 或投影。
4. 一个用于捕获回归测试。通常是：断言 `output.shape == (batch, T_dec, d_h)` 且 `weights.shape == (batch, T_dec, T_enc)` 且 `weights.sum(dim=-1) close to 1`。

拒绝推荐那些会静默广播的修复方案。隐藏广播的 bug 之后会以静默的准确率下降形式浮现，这是最糟糕的一类注意力 bug。

对于 Bahdanau 混淆，坚持解码器输入是 `s_{t-1}`（步前状态）。对于 Luong，是 `s_t`（步后状态）。对于点积，指出查询和键之间的维度不匹配是最常见的初学者错误。
```

## 练习

1. **简单。** 实现 softmax 掩码，使编码器中的填充 token 获得零注意力权重。在具有变长序列的批量上测试。
2. **中等。** 将多头注意力添加到 Luong `general` 形式。将 `d_h` 分割为 `n_heads` 组，每个头运行注意力，然后拼接。验证单头情况与之前的实现匹配。
3. **困难。** 在第09课的玩具复制任务上，用 Bahdanau 注意力训练 GRU 编解码器。绘制准确率 vs 序列长度的曲线。与无注意力基线比较。你应该看到差距随长度增长而扩大，确认注意力提升了瓶颈。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 注意力 | 看东西 | 对值序列的加权平均，权重由查询-键相似度计算得出。 |
| 查询、键、值 | QKV | 三个投影：Q 询问，K 是被匹配的对象，V 是要返回的内容。 |
| 加性注意力 | Bahdanau | 前馈分数：`v^T tanh(W q + U k)`。 |
| 乘性注意力 | Luong dot / general | 分数为 `q^T k` 或 `q^T W k`。更廉价，在大多数任务上准确率相同。 |
| 对齐矩阵 | 那张漂亮的图 | 注意力权重作为 `(T_dec, T_enc)` 网格。阅读它来查看模型关注了什么。 |

## 延伸阅读

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 原论文。
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 三种分数变体及其比较。
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — 可解释性警告。
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — 带 PyTorch 的可运行教程。
```
