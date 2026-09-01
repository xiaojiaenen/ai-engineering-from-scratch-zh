# 位置编码 — 正弦波、RoPE、ALiBi

> 注意力机制对排列不变。"The cat sat on the mat" 和 "mat the on sat cat the" 在没有位置信号的情况下会产生相同的输出。三种算法通过不同的方式来解决这个问题——每种算法对"位置"的含义有不同的看法。

**类型：** 构建
**语言：** Python
**先决条件：** 第 7 阶段 · 02（自注意力），第 7 阶段 · 03（多头注意力）
**时间：** 约 45 分钟

## 问题所在

缩放点积注意力对顺序不敏感。注意力矩阵 `softmax(Q K^T / √d) V` 是基于成对相似度计算的。打乱 `X` 的行，输出的行也会以相同方式打乱。注意力内部没有任何东西关心位置。

这在词袋模型中不是 bug。但对于语言、代码、音频、视频——任何顺序承载意义的领域——这是致命的。

修复方法是以某种方式将位置注入嵌入中。三个阶段的答案：

1. **绝对正弦波**（Vaswani 2017）。将 `sin/cos` 的位置值加到嵌入上。简单、无需学习、超出训练长度后外推能力差。
2. **RoPE — 旋转位置编码**（Su 2021）。根据位置按比例旋转 Q 和 K 向量。直接在点积中编码*相对*位置。2026 年主流方案。
3. **ALiBi — 线性偏置注意力**（Press 2022）。完全跳过嵌入技巧；根据距离为注意力分数添加逐头线性惩罚。优秀的外推能力。

截至 2026 年，几乎所有前沿开源模型都使用 RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少数长上下文模型使用 ALiBi 或其现代变体。绝对正弦波已成为历史。

## 概念解析

![正弦波绝对编码 vs RoPE 旋转 vs ALiBi 距离偏置](../assets/positional-encoding.svg)

### 绝对正弦波

预计算一个形状为 `(max_len, d_model)` 的固定矩阵 `PE`：

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

然后在注意力之前执行 `X' = X + PE[:N]`。每个维度是不同频率的正弦波。模型学习从相位模式中读取位置。超出 `max_len` 时失效：当模型只见过 0–2047 位置时，没有告诉它位置 2048 会发生什么。

### RoPE

旋转 Q 和 K 向量（而非嵌入）。对于每对维度 `(2i, 2i+1)`：

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base 默认值为 10000
```

对位置为 `pos_k` 的 keys 应用相同的旋转。点积 `q'_m · k'_n` 成为仅关于 `(m - n)` 的函数。也就是说：**注意力分数仅取决于相对距离**，尽管旋转是基于绝对位置的。巧妙的技巧。

扩展 RoPE：可以将 `base` 缩放（NTK-aware、YaRN、LongRoPE）以在无需重新训练的情况下外推到更长的上下文。Llama 3 通过这种方式将上下文从 8K 扩展到 128K。

### ALiBi

跳过嵌入技巧。直接偏置注意力分数：

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

其中 `m_h` 是特定于头的斜率（例如 `1 / 2^(8·h/H)`）。较近的 token 获得增强；较远的 token 受到惩罚。无训练时成本。论文显示其长度外推能力优于正弦波，并在原始训练长度上与 RoPE 持平。

### 2026 年如何选择

| 变体 | 外推能力 | 训练成本 | 使用者 |
|------|----------|----------|--------|
| 绝对正弦波 | 差 | 免费 | 原始 transformer、早期 BERT |
| 学习的绝对位置 | 无 | 极小 | GPT-2、GPT-3 |
| RoPE | 良好（配合缩放） | 免费 | Llama 2/3/4、Qwen 2/3、Mistral、DeepSeek-V3、Kimi |
| RoPE + YaRN | 优秀 | 微调阶段 | Qwen2-1M、Llama 3.1 128K |
| ALiBi | 优秀 | 免费 | BLOOM、MPT、Baichuan |

RoPE 获胜是因为它可以无缝融入注意力而无需更改架构，编码相对位置，且其 `base` 超参数为长上下文微调提供了清晰的调节旋钮。

```figure
rope-explorer
```

## 构建它

### 步骤 1：正弦波编码

见 `code/main.py`。一个 4 行的计算：

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

在第一个注意力层之前将其添加到嵌入矩阵中。

### 步骤 2：RoPE 应用于 Q、K

RoPE 就地操作 Q 和 K。对于每对维度：

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

关键点：对位置为 `m` 的 Q 和位置为 `n` 的 K 应用相同的函数。它们的点积会在每个坐标对上获得 `cos((m-n)·θ_i)` 因子。注意力免费学会了相对位置。

### 步骤 3：ALiBi 斜率和偏置

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # 在 softmax 之前加到注意力分数上
```

将 `bias[h]` 添加到头 `h` 的 `(seq_len, seq_len)` 注意力分数矩阵中，然后 softmax。

### 步骤 4：验证 RoPE 的相对距离属性

选择两个随机向量 `a, b`。分别用 `(pos_a, pos_b)` 和 `(pos_a + k, pos_b + k)` 旋转。两者的点积必须在浮点误差范围内匹配。该属性是 RoPE 的核心——它对绝对偏移量不变，只有相对间隔才重要。

## 使用它

PyTorch 2.5+ 在 `torch.nn.functional` 中提供了 RoPE 工具。大多数生产代码使用 `flash_attn` 或 `xformers`，其中 RoPE 在注意力内核内应用。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026 年的长上下文技巧：**

- **NTK-aware 插值。** 当从 4K 扩展到 16K+ 时，将 `base` 重缩放为 `base * (scale_factor)^(d/(d-2))`。
- **YaRN。** 更智能的插值，在长上下文中保留注意力熵。Llama 3.1 128K 使用它。
- **LongRoPE。** Microsoft 2024 年的方法，使用进化搜索来选择每维度的缩放因子。Phi-3-Long 使用它。
- **位置插值 + 微调。** 只需将位置按扩展因子缩小，然后微调 1–50 亿 token。出人意料地有效。

## 交付它

见 `outputs/skill-positional-encoding-picker.md`。该技能根据目标上下文长度、外推需求和训练预算为新模型选择编码策略。

## 练习

1. **简单。** 绘制正弦波 `PE` 矩阵的热图，`max_len=512, d=128`。确认"随着维度索引增大条纹变宽"的模式。
2. **中等。** 实现 NTK-aware RoPE 缩放。在长度为 256 的序列上训练小型语言模型，然后在有无缩放的条件下测试长度 1024。测量困惑度。
3. **困难。** 在同一个注意力模块中实现 ALiBi 和 RoPE。在复制任务的 512 长度序列上训练 4 层 transformer。在测试时外推到 2048。比较性能下降情况。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 位置编码 | "告诉注意力顺序" | 添加到嵌入或注意力中的任何编码位置的信息。 |
| 正弦波 | "原来的那个" | 几何频率的 `sin/cos` 添加到嵌入中；无法外推。 |
| RoPE | "旋转嵌入" | 按依赖于位置的角度旋转 Q、K；点积编码相对距离。 |
| ALiBi | "线性偏置技巧" | 向注意力分数添加 `-m·\|i-j\|`；无需嵌入，外推能力强。 |
| base | "RoPE 的旋钮" | RoPE 中的频率缩放器；增大以在推理时扩展上下文。 |
| NTK-aware | "一种 RoPE 缩放技巧" | 重缩放 `base`，使高频维度在上下文扩展时不被压缩。 |
| YaRN | "高级的那个" | 每维度的插值+外推，保留注意力熵。 |
| 外推 | "在训练长度之外有效" | 位置方案能否在训练中未见过的超过 `max_len` 的位置提供正确输出？ |

## 延伸阅读

- [Vaswani 等人 (2017)。Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 原始正弦波。
- [Su 等人 (2021)。RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 论文。
- [Press, Smith, Lewis (2021)。Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi。
- [Peng 等人 (2023)。YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 最先进的 RoPE 缩放。
- [Chen 等人 (2023)。Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta 的 Llama 2 长上下文论文。
- [Ding 等人 (2024)。LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — Microsoft 的方法，被 Phi-3-Long 使用，并在"使用它"部分引用。
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 生产级实现的每种 RoPE 缩放方案（默认、线性、动态、YaRN、LongRoPE、Llama-3）。
