# 位置编码 — 正弦编码、RoPE 与 ALiBi

> 注意力机制是排列不变的。"The cat sat on the mat" 和 "mat the on sat cat the" 在没有位置信号时产生相同的输出。三种算法解决了这个问题——每种都对"位置"的含义有不同的假设。

**类型:** 构建  
**语言:** Python  
**先修知识:** 第 7 阶段 · 02 (自注意力), 第 7 阶段 · 03 (多头注意力)  
**时间:** ~45 分钟

## 问题所在

缩放点积注意力对顺序不敏感。注意力矩阵 `softmax(Q K^T / √d) V` 是基于成对相似性计算的。打乱 `X` 的行，输出的行也会以相同方式被打乱。注意力机制内部不关心位置。

对于词袋模型来说这不是问题。但对于语言、代码、音频、视频——任何顺序承载含义的数据——这是致命的。

解决方法是将位置信息以某种方式注入到嵌入中。三个时代的解决方案：

1.  **绝对正弦编码** (Vaswani 2017)。将位置 `sin/cos` 加到嵌入上。简单、无需学习，但对超出训练长度的外推效果差。
2.  **RoPE — 旋转位置编码** (Su 2021)。将 Q 和 K 向量按与位置成比例的角度旋转。直接在点积中编码*相对*位置。2026 年的主流方法。
3.  **ALiBi — 线性偏置注意力** (Press 2022)。完全跳过嵌入；根据距离向注意力分数添加每头的线性惩罚。具有优异的长度外推能力。

截至 2026 年，几乎所有前沿开源模型都使用 RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少数长上下文模型使用 ALiBi 或其现代变体。绝对正弦编码已成为历史。

## 核心概念

![正弦绝对编码 vs RoPE 旋转 vs ALiBi 距离偏置](../assets/positional-encoding.svg)

### 绝对正弦编码

预计算一个形状为 `(max_len, d_model)` 的固定矩阵 `PE`：

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

然后在注意力之前将其加到 `X' = X + PE[:N]`。每个维度都是不同频率的正弦波。模型学习从相位模式中读取位置。超出 `max_len` 会失败：模型从未被告知当只见过 0-2047 位置时，位置 2048 会发生什么。

### RoPE

旋转 Q 和 K 向量（不是嵌入）。对于一对维度 `(2i, 2i+1)`：

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

对位置为 `pos_k` 的键应用相同的旋转。点积 `q'_m · k'_n` 变成仅关于 `(m - n)` 的函数。也就是说：**注意力分数仅取决于相对距离**，尽管旋转是基于绝对位置的。巧妙的技巧。

扩展 RoPE：可以缩放 `base`（NTK 感知、YaRN、LongRoPE），无需重新训练即可外推到更长上下文。Llama 3 就是通过这种方式将上下文从 8K 扩展到 128K。

### ALiBi

跳过嵌入技巧。直接偏置注意力分数：

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

其中 `m_h` 是每头特定的斜率（例如 `1 / 2^(8·h/H)`）。较近的 token 获得增强；较远的 token 受到惩罚。无训练时间开销。论文表明其长度外推优于正弦编码，并与 RoPE 在其原始训练长度上相匹配。

### 2026 年如何选择

| 变体 | 外推能力 | 训练成本 | 使用者 |
|---------|---------------|---------------|---------|
| 绝对正弦编码 | 差 | 无 | 原始 Transformer、早期 BERT |
| 学习式绝对编码 | 无 | 微小 | GPT-2、GPT-3 |
| RoPE | 通过缩放表现良好 | 无 | Llama 2/3/4、Qwen 2/3、Mistral、DeepSeek-V3、Kimi |
| RoPE + YaRN | 优秀 | 微调阶段 | Qwen2-1M、Llama 3.1 128K |
| ALiBi | 优秀 | 无 | BLOOM、MPT、Baichuan |

RoPE 胜出，因为它无需改变架构即可融入注意力机制，编码相对位置，并且其 `base` 超参数为长上下文微调提供了简洁的调节旋钮。

## 构建实现

### 第 1 步：正弦编码

参见 `code/main.py`。4 行计算：

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

在第一个注意力层之前将其加到嵌入矩阵上。

### 第 2 步：将 RoPE 应用于 Q、K

RoPE 就地操作 Q 和 K。对于每一对维度：

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

关键点：对位置 `m` 的 Q 和位置 `n` 的 K 应用相同的函数。它们的点积在每个坐标对上都获得一个 `cos((m-n)·θ_i)` 因子。注意力机制免费学习了相对位置。

### 第 3 步：ALiBi 斜率和偏置

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

将 `bias[h]` 加到头 `h` 的 `(seq_len, seq_len)` 注意力分数矩阵上，然后进行 softmax。

### 第 4 步：验证 RoPE 的相对距离特性

选取两个随机向量 `a, b`。先按 `(pos_a, pos_b)` 旋转，再按 `(pos_a + k, pos_b + k)` 旋转。两个点积必须在浮点误差范围内匹配。这个特性是 RoPE 的核心——它对绝对偏移不变，只有相对间隔才重要。

## 实际应用

PyTorch 2.5+ 在 `torch.nn.functional` 中提供了 RoPE 工具。大多数生产代码使用 `flash_attn` 或 `xformers`，其中 RoPE 在注意力核内部应用。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026 年的长上下文技巧：**

- **NTK 感知插值。** 将 `base` 重新缩放为 `base * (scale_factor)^(d/(d-2))`，当从 4K 扩展到 16K+ 时。
- **YaRN。** 更智能的插值，在长上下文上保持注意力熵。Llama 3.1 128K 使用此方法。
- **LongRoPE。** 微软 2024 年的方法，使用进化搜索为每个维度选择缩放因子。Phi-3-Long 使用此方法。
- **位置插值 + 微调。** 只需按扩展因子缩小位置并微调 1–5B 个 token。效果出人意料地好。

## 部署建议

参见 `outputs/skill-positional-encoding-picker.md`。该技能根据目标上下文长度、外推需求和训练预算为新模型选择编码策略。

## 练习

1.  **简单。** 将 `max_len=512, d=128` 的正弦 `PE` 矩阵绘制成热力图。确认"条纹随维度索引增加而变宽"的模式。
2.  **中等。** 实现 NTK 感知 RoPE 缩放。在长度为 256 的序列上训练一个微型语言模型，然后在长度为 1024 的序列上测试，分别使用和不使用缩放。测量困惑度。
3.  **困难。** 在同一个注意力模块中实现 ALiBi 和 RoPE。在长度为 512 的序列上训练一个 4 层 Transformer 进行复制任务。在测试时外推到 2048。比较性能下降。

## 关键术语

| 术语 | 人们通常的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 位置编码 | "告诉注意力机制顺序" | 任何添加到嵌入或注意力中以编码位置的信号。 |
| 正弦编码 | "最初的那个" | 以几何频率 `sin/cos` 添加到嵌入；无法外推。 |
| RoPE | "旋转嵌入" | 通过位置相关角度旋转 Q、K；点积编码相对距离。 |
| ALiBi | "线性偏置技巧" | 向注意力分数添加 `-m·|i-j|`；无需嵌入，外推优秀。 |
| base | "RoPE 的旋钮" | RoPE 中的频率缩放器；增加它可在推理时扩展上下文。 |
| NTK 感知 | "一种 RoPE 缩放技巧" | 重新缩放 `base`，使得上下文扩展时高频维度不被压缩。 |
| YaRN | "那个高级的" | 保持注意力熵的每维度插值+外推。 |
| 外推 | "在训练长度之外有效" | 位置方案能否在训练中见过的 `max_len` 之外提供正确的输出？ |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 原始正弦编码。
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 论文。
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi。
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 最先进的 RoPE 缩放技术。
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta 的 Llama 2 长上下文论文。
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — 微软的方法，用于 Phi-3-Long，并在“实际应用”部分引用。
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 每种 RoPE 缩放方案的生产级实现（默认、线性、动态、YaRN、LongRoPE、Llama-3）。