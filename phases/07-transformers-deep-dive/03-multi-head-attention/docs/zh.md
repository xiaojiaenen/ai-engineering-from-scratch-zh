# 多头注意力

> 一个注意力头学习一种关系。八个头学习八种。头是免费的，多用几个。

**类型:** 构建
**语言:** Python
**先决条件:** 阶段 7 · 02（从零开始实现自注意力）
**时间:** 约 75 分钟

## 问题所在

单个自注意力头计算一个注意力矩阵。该矩阵捕获一种关系——通常是使训练信号上的损失最小的那种关系。如果你的数据中主谓一致、共指、远距离语篇和句法分块全都纠缠在一起，单个头会将它们涂抹成单一的 softmax 分布，并丢失一半的信号。

2017年Vaswani论文的解决方案：并行运行多个注意力函数，每个函数都有自己的Q、K、V投影，并拼接输出。每个头在维度为 `d_model / n_heads` 的较小子空间中运作。总参数量保持不变。表达能力得到提升。

多头注意力是2026年每个transformer默认搭载的功能。唯一的争论在于*多少*个头，以及键和值是否共享投影（分组查询注意力、多查询注意力、多头潜在注意力）。

## 概念图解

![多头注意力：分割、注意力、拼接](../assets/multi-head-attention.svg)

**分割。** 取形状为 `(N, d_model)` 的 `X`。分别投影到形状为 `(N, d_model)` 的 Q、K、V。重塑为 `(N, n_heads, d_head)`，其中 `d_head = d_model / n_heads`。转置为 `(n_heads, N, d_head)`。

**并行执行注意力。** 在每个头内部运行缩放点积注意力。每个头产生 `(N, d_head)`。头在嵌入的不同子空间上运作，在注意力计算期间彼此不交流。

**拼接并投影。** 将头重新堆叠回 `(N, d_model)`，然后乘以一个形状为 `(d_model, d_model)` 的可学习输出矩阵 `W_o`。`W_o` 是头之间发生混合的地方。

**为何有效。** 每个头可以专注于特定功能，无需与其他头竞争表示预算。2019-2024年的探测研究揭示了不同的头角色：位置头、关注前一个token的头、复制头、命名实体头、归纳头（这是上下文学习的基础）。

**2026年的变体谱系：**

| 变体 | Q 头数 | K/V 头数 | 使用者 |
|---------|---------|-----------|---------|
| 多头 (MHA) | N | N | GPT-2, BERT, T5 |
| 多查询 (MQA) | N | 1 | PaLM, Falcon |
| 分组查询 (GQA) | N | G (例如 N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| 多头潜在 (MLA) | N | 压缩到低秩 | DeepSeek-V2, V3 |

GQA是现代默认选择，因为它在保持几乎完整质量的同时，将KV缓存内存减少了 `N/G` 倍。MLA更进一步，将K/V压缩到潜在空间，然后在计算时投影回来——消耗FLOPs，但节省更多内存。

## 构建实现

### 步骤1：从我们已有的单头注意力中分割头

取第02课中的 `SelfAttention`，用一个分割/拼接对包裹它。参见 `code/main.py` 的numpy实现；逻辑是：

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一次重塑和一次转置。没有循环。这正是PyTorch在 `nn.MultiheadAttention` 下所做的事情。

### 步骤2：对每个头运行缩放点积注意力

每个头获得自己的Q、K、V切片。注意力计算变为批处理矩阵乘法：

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

在真实硬件上 `Qh @ Kh.transpose(...)` 是一个 `bmm`。GPU看到一个形状为 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)` 的单一批处理矩阵乘法。增加头是免费的。

### 步骤3：分组查询注意力变体

只有键和值投影发生变化。Q获得 `n_heads` 组；K和V获得 `n_kv_heads < n_heads` 组，并进行重复以匹配：

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

在推理时，这节省了内存，因为KV缓存中只存在 `n_kv_heads` 份拷贝，而不是 `n_heads` 份。Llama 3 70B使用64个查询头和8个KV头——缓存缩小了8倍。

### 步骤4：探测每个头学到了什么

对一个包含4个头的短句子运行MHA。对于每个头，打印 `(N, N)` 注意力矩阵。你会看到即使使用随机初始化，不同的头也会提取出不同的结构——这部分是信号，部分是子空间中的旋转对称性。

## 实际使用

在PyTorch中，一行版本：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+中的GQA：

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**多少个头？** 来自2026年生产模型的经验法则：

| 模型规模 | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| 小型 (~125M) | 768 | 12 | 64 |
| 基础 (~350M) | 1024 | 16 | 64 |
| 大型 (~1B) | 2048 | 16 | 128 |
| 前沿 (~70B) | 8192 | 64 | 128 |

`d_head` 几乎总是落在64或128。它是一个头能"看到"多少内容的单位。低于32，头就开始与缩放因子 `sqrt(d_head)` 作斗争；超过256，你就失去了"许多小型专家"的好处。

## 交付部署

参见 `outputs/skill-mha-configurator.md`。该技能根据参数预算、序列长度和部署目标，为一个新的transformer推荐头数量、kv头数量和投影策略。

## 练习

1. **简单。** 从 `code/main.py` 取MHA，将 `n_heads` 从1改为16，同时固定 `d_model=64`。绘制一个单层微型模型在合成复制任务上的损失。更多头是有帮助、达到平台期还是有害？
2. **中等。** 实现MQA（一个KV头在所有查询头之间共享）。测量参数量相比完整MHA减少了多少。计算在N=2048时，推理阶段KV缓存大小缩小了多少。
3. **困难。** 实现一个微型版本的多头潜在注意力：将K,V压缩到秩为 `r` 的潜在表示，将潜在表示存储在KV缓存中，在注意力计算时解压缩。在什么 `r` 下，缓存内存能降到完整MHA的1/8以下，同时质量保持在验证ppl的1比特以内？

## 关键术语

| 术语 | 人们通常怎么说 | 它的实际含义 |
|------|-----------------|-----------------------|
| 头 | "一个注意力电路" | 一个维度为 `d_head = d_model / n_heads` 的Q/K/V投影，拥有自己的注意力矩阵。 |
| d_head | "头维度" | 每个头的隐藏宽度；在生产中几乎总是64或128。 |
| 分割 / 合并 | "重塑技巧" | 围绕注意力的 `(N, d_model) ↔ (n_heads, N, d_head)` 重塑+转置操作。 |
| W_o | "输出投影" | 拼接头后应用的 `(d_model, d_model)` 矩阵；头在此混合。 |
| MQA | "一个KV头" | 多查询注意力：单一共享的K/V投影。最小的KV缓存，但有些质量损失。 |
| GQA | "自Llama 2以来的默认" | 分组查询注意力，使用 `n_kv_heads < n_heads`；重复以匹配Q。 |
| MLA | "DeepSeek的技巧" | 多头潜在注意力：K,V压缩到低秩潜在表示，在计算注意力时解压缩。 |
| 归纳头 | "上下文学习背后的电路" | 一对检测先前出现并复制其后内容的头。 |

## 延伸阅读

- [Vaswani 等 (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 原始的多头规范。
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA论文。
- [Ainslie 等 (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — 如何在训练后将MHA转换为GQA。
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA及其为何在缓存内存方面优于MHA/GQA。
- [Olsson 等 (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — 从机制上探究头实际做了什么。