# 完整的 Transformer — 编码器 + 解码器

> 注意力机制是主角。其余一切——残差连接、归一化、前馈网络、交叉注意力——都是为了让你能深深堆叠它的脚手架。

**类型：** 构建
**语言：** Python
**前置知识：** 第7阶段 · 02（自注意力）、第7阶段 · 03（多头注意力）、第7阶段 · 04（位置编码）
**预计时间：** 约75分钟

## 问题所在

单个注意力层只是特征提取器，不是完整模型。每层一次矩阵乘法所提供的容量不足以处理语言任务。你需要深度——而深度在没有合适"管道"的情况下会崩溃。

2017年Vaswani的论文总结了六项设计决策，将一个注意力层变成了可堆叠的模块。此后所有Transformer——仅编码器（BERT）、仅解码器（GPT）、编码器-解码器（T5）——都继承了相同的骨架。到2026年，这些模块已被改进（RMSNorm、SwiGLU、预归一化、RoPE），但骨架完全相同。

本课时讲解的就是这个骨架。接下来的课时将对它进行专业化——第06课讲编码器，第07课讲解码器，第08课讲编码器-解码器。

## 核心概念

![编码器与解码器模块内部结构](../assets/full-transformer.svg)

### 六大组件

1. **嵌入 + 位置信号。** 词元 → 向量。位置信息通过 RoPE（现代方案）或正弦编码（经典方案）注入。
2. **自注意力。** 每个位置都关注所有其他位置。在解码器中使用掩码。
3. **前馈网络（FFN）。** 逐位置两层 MLP：`W_2 · activation(W_1 · x)`。扩展比例默认为 4×。
4. **残差连接。** `x + sublayer(x)`。没有它，梯度在经过约6层后会消失。
5. **层归一化。** `LayerNorm` 或 `RMSNorm`（现代方案）。稳定残差流。
6. **交叉注意力（仅解码器）。** 查询来自解码器，键和值来自编码器输出。

观察一个向量流经一个模块：注意力跨位置混合信息，残差将其向前传递，FFN 对其进行变换，归一化保持流的稳定。

```figure
transformer-block
```

### 编码器模块（用于 BERT、T5 编码器）

```
x → LN → MHA(自注意力) → + → LN → FFN → + → 输出
                     ^              ^
                     |              |
                     └── 残差 ──────┘
```

编码器是双向的。无需掩码。所有位置都能看到所有位置。

### 解码器模块（用于 GPT、T5 解码器）

```
x → LN → MHA(掩码自注意力) → + → LN → MHA(交叉到编码器) → + → LN → FFN → + → 输出
```

解码器每个模块包含三个子层。中间那个——交叉注意力——是信息从编码器流向解码器的唯一途径。在纯解码器架构（如 GPT）中，省略交叉注意力，只保留掩码自注意力 + FFN。

### 预归一化 vs 后归一化

原始论文中的两种形式：`x + sublayer(LN(x))` 与 `LN(x + sublayer(x))`。后归一化在2019年前后逐渐失宠——如果不经过精心设计的预热，深层训练会更困难。预归一化（`LN` *在* 子层之前）是2026年的默认选择：Llama、Qwen、GPT-3+、Mistral 均使用它。

### 2026 年现代化模块

Vaswani 2017 发布的是 LayerNorm + ReLU。现代堆栈两者都替换了。实际生产模块的样子如下：

| 组件 | 2017年 | 2026年 |
|------|--------|--------|
| 归一化 | LayerNorm | RMSNorm |
| FFN 激活函数 | ReLU | SwiGLU |
| FFN 扩展比例 | 4× | 2.6×（SwiGLU 使用三个矩阵，总参数量匹配） |
| 位置编码 | 正弦绝对位置 | RoPE |
| 注意力 | 完整 MHA | GQA（或 MLA） |
| 偏置项 | 有 | 无 |

RMSNorm 去掉了 LayerNorm 的均值中心化（少一次减法），节省了计算量，且实证稳定性至少相当。SwiGLU（`Swish(W1 x) ⊙ W3 x`）在 Llama、PaLM 和 Qwen 论文中一致比 ReLU/GELU 的 FFN 高出约 0.5 个 ppl（perplexity）。

### 参数量

对于一个模块，设 `d_model = d`，FFN 扩展比例为 `r`：

- MHA：`4 · d²`（Q、K、V、O 投影）
- FFN（SwiGLU）：`3 · d · (r · d)` ≈ `3rd²`
- 归一化：可忽略

在 `d = 4096, r = 2.6, layers = 32`（大约 Llama 3 8B 的配置）下，总计：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B 参数每层 × 32 ≈ 7B`（加上嵌入和输出头）。与公开发布的参数量吻合。

## 动手实现

### 步骤1：基础构建块

使用第03课中的小型 `Matrix` 类（复制到此文件以保持独立性）：

- `layer_norm(x, eps=1e-5)` — 减均值，除以标准差。
- `rms_norm(x, eps=1e-6)` — 除以均方根。无均值减法。
- `gelu(x)` 和 `silu(x) * W3 x`（SwiGLU）。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` 和 `decoder_block(x, enc_out, params)`。

完整接线参见 `code/main.py`。

### 步骤2：搭建一个2层编码器和一个2层解码器

将它们堆叠。把编码器输出传入每个解码器的交叉注意力层。在输出投影前添加最终的 LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 步骤3：在玩具示例上运行前向传播

输入一个6词元的源序列和一个5词元的目标序列。验证输出形状为 `(5, vocab)`。无需训练——本课时聚焦架构，而非损失函数。

### 步骤4：替换为 RMSNorm + SwiGLU

将 LayerNorm 和 ReLU-FFN 替换为 RMSNorm 和 SwiGLU。确认形状仍然匹配。这就是通过一次函数替换完成的2026年现代化。

## 实际应用

PyTorch/TF 参考实现：`nn.TransformerEncoderLayer`、`nn.TransformerDecoderLayer`。但大多数2026年的生产代码都会自行实现模块，因为：

- Flash Attention 在注意力内部调用，而非通过 `nn.MultiheadAttention`。
- GQA / MLA 不在标准库参考实现中。
- RoPE、RMSNorm、SwiGLU 并非 PyTorch 默认选项。

HuggingFace `transformers` 提供了干净的参考模块供你阅读：`modeling_llama.py` 是2026年解码器优先模块的标准范本。它大约500行，值得一读。

**何时选择编码器 vs 解码器 vs 编码器-解码器：**

| 需求 | 选择 | 示例 |
|------|------|------|
| 分类、嵌入、文本 QA | 仅编码器 | BERT、DeBERTa、ModernBERT |
| 文本生成、聊天、代码、推理 | 仅解码器 | GPT、Llama、Claude、Qwen |
| 结构化输入 → 结构化输出（翻译、摘要） | 编码器-解码器 | T5、BART、Whisper |

仅解码器架构在语言任务中胜出，因为它扩展最干净，且能同时处理理解和生成。当输入具有清晰的"源序列"身份时（翻译、语音识别、结构化任务），编码器-解码器仍然是最佳选择。

## 交付成果

参见 `outputs/skill-transformer-block-reviewer.md`。技能评测会根据2026年默认配置审查新的 Transformer 模块实现，并标记缺失的部分（预归一化、RoPE、RMSNorm、GQA、FFN 扩展比例）。

## 练习

1. **简单。** 计算你的 `encoder_block` 在 `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` 时的参数量。通过实现该模块并使用 `sum(p.numel() for p in block.parameters())` 进行验证。
2. **中等。** 从后归一化切换到预归一化。初始化两者，并在随机输入上经过12层堆叠后测量激活范数。后归一化的激活应爆炸；预归一化的应保持在有界范围内。
3. **困难。** 在玩具复制任务（复制 `x` 并反转）上实现一个4层编码器-解码器。训练100步，报告损失。替换为 RMSNorm + SwiGLU + RoPE——损失是否下降？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| 模块（Block） | "一个 Transformer 层" | 归一化 + 注意力 + 归一化 + FFN 的堆叠，包裹在残差连接中。 |
| 残差（Residual） | "跳跃连接" | `x + f(x)` 的输出；使梯度能够流经深层堆叠。 |
| 预归一化（Pre-norm） | "先归一化，不在之后" | 现代方案：`x + sublayer(LN(x))`。无需预热技巧即可训练更深网络。 |
| RMSNorm | "去掉均值的 LayerNorm" | 除以均方根；少一次运算，实证稳定性相同。 |
| SwiGLU | "大家切换到的 FFN" | `Swish(W1 x) ⊙ W3 x → W2`。在语言模型 ppl 上优于 ReLU/GELU。 |
| 交叉注意力（Cross-attention） | "解码器如何看到编码器" | MHA，其中 Q 来自解码器，K/V 来自编码器输出。 |
| FFN 扩展（FFN expansion） | "中间 MLP 有多宽" | 隐藏层大小与 d_model 的比值，通常为 4（LayerNorm）或 2.6（SwiGLU）。 |
| 无偏置（Bias-free） | "去掉 +b 项" | 现代堆栈在线性层中省略偏置；ppl 略有改善，模型更小。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始模块规范。
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — 为何预归一化在深层中优于后归一化。
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU 论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 2026年标准的解码器优先模块。
