# 多头注意力（Multi-Head Attention）

> 一个注意力头一次只学习一种关系。八个头就学习八种。头是免费的，多拿几个。

**类型：** Build
**语言：** Python
**前置知识：** Phase 7 · 02（从零实现自注意力）
**预计用时：** ~75 分钟

## 问题所在

一个自注意力头只计算一个注意力矩阵。该矩阵只能捕捉一类关系——通常是 whichever 在训练信号下使损失最小化的那一类。如果你的数据同时纠缠着主谓一致、共指消解、长距离篇章结构和句法分块，单个头会将它们 smeared 进一个 softmax 分布里，丢了一半的信号。

2017 年 Vaswani 论文提出的修复方案：并行运行多个注意力函数，每个头各有独立的 Q、K、V 投影，然后把输出拼起来。每个头在维度为 `d_model / n_heads` 的较小子空间里运作。总参数量保持不变，表达能力却提升了。

多头注意力是 2026 年每一个 Transformer 的标配。唯一争论的是*多少个*头，以及键和值是否共享投影（分组查询注意力 GQA、多查询注意力 MQA、多头潜注意力 MLA）。

## 概念

![多头注意力拆分、注意力计算、拼接](../assets/multi-head-attention.svg)

**拆分。** 取 `X`，形状为 `(N, d_model)`。投影到 Q、K、V，每个形状都是 `(N, d_model)`。reshape 为 `(N, n_heads, d_head)`，其中 `d_head = d_model / n_heads`。转置为 `(n_heads, N, d_head)`。

**并行做注意力。** 在每个头内跑缩放点积注意力。每个头产出 `(N, d_head)`。这些头在嵌入的不同子空间上运作，在注意力计算本身期间互不通信。

**拼接并投影。** 把头重新叠成 `(N, d_model)`，再乘上一个可学习的输出矩阵 `W_o`，形状为 `(d_model, d_model)`。`W_o` 是让头之间互相混合的地方。

**为什么有效。** 每个头可以专业化，而不与其他头争夺表示预算。2019–2024 年的探针研究表明存在不同的头角色分工：位置头、 attends to the previous token 的头、copy 头、命名实体头、诱导头（induction heads，是上下文学习的基础）。

**2026 年的变体谱系：**

| 变体 | Q 头数 | K/V 头数 | 使用方 |
|---------|---------|-----------|---------|
| 多头（MHA） | N | N | GPT-2、BERT、T5 |
| 多查询（MQA） | N | 1 | PaLM、Falcon |
| 分组查询（GQA） | N | G（如 N/8） | Llama 2 70B、Llama 3+、Qwen 2+、Mistral |
| 多头潜注意力（MLA） | N | 压缩为低秩 | DeepSeek-V2、V3 |

GQA 成为现代默认选择，因为它把 KV cache 内存削减了 `N/G` 倍，同时几乎保留完整质量。MLA 更进一步，把 K/V 压缩进潜空间，再在计算时投影回来——多花 FLOPs，省更多内存。

```figure
multihead-split
```

## 动手实现

### 步骤 1：从已有的单头注意力上拆分出头

从第 02 课的 `SelfAttention` 出发，用 split/concat 一对把它包起来。`code/main.py` 里有 numpy 实现；逻辑如下：

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一次 reshape，一次 transpose。没有循环。这正是 PyTorch 在 `nn.MultiheadAttention` 下做的事。

### 步骤 2：按头跑缩放点积注意力

每个头拿到自己的 Q、K、V 切片。注意力变成一个 batched matmul：

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

在真实硬件上 `Qh @ Kh.transpose(...)` 是一次 `bmm`。GPU 看到的是形状为 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)` 的单个 batched matmul。增加头数是免费的。

### 步骤 3：分组查询注意力（GQA）变体

只有键和值的投影发生变化。Q 分到 `n_heads` 组；K 和 V 分到 `n_kv_heads < n_heads` 组，然后 repeat 以匹配：

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

推理时这能省内存，因为只有 `n_kv_heads` 份副本留在 KV cache 里，而不是 `n_heads` 份。Llama 3 70B 用了 64 个 query head、8 个 KV head——cache 缩小 8 倍。

### 步骤 4：探测每个头学到了什么

在短句上跑 MHA，设 4 个头。对每个头打印 `(N, N)` 注意力矩阵。你会看到不同的头即使在随机初始化下也各自挑出不同的结构——这部分是信号，部分是子空间里的旋转对称性。

## 使用方式

PyTorch 里的一行版本：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+ 里的 GQA：

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention 会在 CUDA 上自动 dispatch 到 Flash Attention。
# 对于 GQA，传入形状为 (B, n_heads, N, d_head) 的 Q 和形状为
# (B, n_kv_heads, N, d_head) 的 K、V。PyTorch 会自动处理 repeat。
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**头数怎么选？** 2026 年产线模型的业界经验法则：

| 模型规模 | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| 小型（~125M） | 768 | 12 | 64 |
| 基础（~350M） | 1024 | 16 | 64 |
| 大型（~1B） | 2048 | 16 | 128 |
| 前沿（~70B） | 8192 | 64 | 128 |

`d_head` 几乎总是 64 或 128。它是衡量一个头能"看"多少的基本单位。低于 32，头会与缩放因子 `sqrt(d_head)` 打架；高于 256，就会失去"很多小专家"的收益。

## 交付

参见 `outputs/skill-mha-configurator.md`。该 skill 会针对给定参数预算、序列长度和部署目标的新型 Transformer，推荐头数、KV 头数和投影策略。

## 练习

1. **简单。** 从 `code/main.py` 中的 MHA 出发，把 `n_heads` 从 1 改为 16，固定 `d_model=64`。在一个合成 copy 任务上训练一个极简单层模型并绘制 loss 曲线。头越多越好、收敛后持平、还是反而变差？
2. **中等。** 实现 MQA（所有 query 头共享一个 KV 头）。测量参数量相比完整 MHA 减少了多少。计算 N=2048 时推理阶段 KV cache 缩小了多少。
3. **困难。** 实现一个简版的多头潜注意力（MLA）：把 K、V 压缩到秩为 `r` 的潜变量，把潜变量存入 KV cache，在注意力时再解压。在 `r` 取何值时缓存内存会降到完整 MHA 的 1/8 以下，同时质量仍在验证 ppl 的 1 bit 之内？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Head | "一个独立的注意力电路" | 一组维度为 `d_head = d_model / n_heads` 的 Q/K/V 投影，以及它自己的注意力矩阵。 |
| d_head | "头维度" | 每个头的隐藏宽度；产线中几乎总是 64 或 128。 |
| Split / combine | "reshape 技巧" | 注意力前后围绕 `(N, d_model) ↔ (n_heads, N, d_head)` 的 reshape+transpose。 |
| W_o | "输出投影" | 拼接完头之后应用的 `(d_model, d_model)` 矩阵；头在这里互相混合。 |
| MQA | "一个 KV 头" | Multi-Query Attention：单个共享的 K/V 投影。KV cache 最小，有一定质量损失。 |
| GQA | "Llama 2 以来的默认" | 分组查询注意力，`n_kv_heads < n_heads`；repeat 以匹配 Q。 |
| MLA | "DeepSeek 的招数" | 多头潜注意力：K/V 压缩为低秩潜变量，注意力时再解压。 |
| Induction head | "上下文学习背后的电路" | 一对头，检测之前的出现位置，并 copy 其后的内容。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) —— 原始多头规范。
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) —— MQA 论文。
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) —— 如何在训练后把 MHA 转成 GQA。
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) —— MLA 及为何它在缓存内存上优于 MHA/GQA。
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) —— 从机理角度观察头实际在做什么。
