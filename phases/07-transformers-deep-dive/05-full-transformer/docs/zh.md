# 完整的 Transformer — 编码器 + 解码器

> 注意力是核心。其他所有组件——残差连接、归一化、前馈网络、交叉注意力——都是使其能够深层堆叠的脚手架。

**类型：** 构建
**语言：** Python
**前置课程：** 第 7 阶段 · 02 (自注意力)， 第 7 阶段 · 03 (多头注意力)， 第 7 阶段 · 04 (位置编码)
**时间：** ~75 分钟

## 问题所在

单个注意力层是一个特征提取器，而不是模型。每层仅一次矩阵乘法对于语言任务来说容量不足。你需要深度——而没有正确的管道，深度就无法实现。

2017 年 Vaswani 的论文打包了六个设计决策，将一个注意力层变成了可堆叠的模块。此后每个 Transformer——仅编码器（BERT）、仅解码器（GPT）、编码器-解码器（T5）——都继承了相同的骨架。在 2026 年，这些模块已被改进（RMSNorm、SwiGLU、pre-norm、RoPE），但骨架是相同的。

本课讲解的就是这个骨架。后续课程将对其进行特化——06 用于编码器，07 用于解码器，08 用于编码器-解码器。

## 核心概念

![编码器和解码器模块内部结构图，已连接](../assets/full-transformer.svg)

### 六大组成部分

1.  **嵌入 + 位置信号。** Token → 向量。位置信息通过 RoPE（现代）或正弦函数（经典）注入。
2.  **自注意力。** 每个位置都关注其他所有位置。在解码器中被掩码。
3.  **前馈网络 (FFN)。** 逐位置的两层 MLP：`W_2 · activation(W_1 · x)`。默认扩展比率 4×。
4.  **残差连接。** `x + sublayer(x)`。没有它，梯度在大约 6 层后就会消失。
5.  **层归一化。** `LayerNorm` 或 `RMSNorm`（现代）。稳定残差流。
6.  **交叉注意力（仅解码器）。** 查询来自解码器，键和值来自编码器输出。

### 编码器模块（BERT, T5 编码器使用）

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

编码器是双向的。没有掩码。所有位置都能看到所有位置。

### 解码器模块（GPT, T5 解码器使用）

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

解码器每个模块有三个子层。中间那个——交叉注意力——是信息从编码器流向解码器的唯一位置。在纯解码器架构（GPT）中，交叉注意力被省略，你只有掩码自注意力 + FFN。

### Pre-norm 与 Post-norm

原始论文：`x + sublayer(LN(x))` 对比 `LN(x + sublayer(x))`。Post-norm 在 2019 年左右失宠——没有仔细预热的情况下，它更难进行深度训练。Pre-norm（`LN` *在* 子层 *之前*）是 2026 年的默认选择：Llama、Qwen、GPT-3+、Mistral 都使用它。

### 2026 年的现代化模块

Vaswani 2017 版使用 LayerNorm + ReLU。现代技术栈替换了两者。生产环境中的模块实际如下：

| 组件 | 2017 | 2026 |
|-----------|------|------|
| 归一化 | LayerNorm | RMSNorm |
| FFN 激活函数 | ReLU | SwiGLU |
| FFN 扩展 | 4× | 2.6× (SwiGLU 使用三个矩阵，总参数量匹配) |
| 位置编码 | 正弦绝对位置 | RoPE |
| 注意力 | 全 MHA | GQA (或 MLA) |
| 偏置项 | 有 | 无 |

RMSNorm 去掉了 LayerNorm 的均值中心化（少一次减法），节省计算且经验上至少同样稳定。SwiGLU (`Swish(W1 x) ⊙ W3 x`) 在 Llama、PaLM 和 Qwen 论文中，一致性地比 ReLU/GELU FFN 高出约 0.5 个 ppl 点数。

### 参数计数

对于一个具有 `d_model = d` 和 FFN 扩展 `r` 的模块：

- MHA：`4 · d²` (Q, K, V, O 投影)
- FFN (SwiGLU)：`3 · d · (r · d)` ≈ `3rd²`
- 归一化层：可忽略

当 `d = 4096, r = 2.6, layers = 32`（大致相当于 Llama 3 8B）时，总计：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B`（加上嵌入层和输出头）。与已发布数量一致。

## 构建它

### 步骤 1：基础构件

使用第 3 课中微小的 `Matrix` 类（为独立性复制到本文件）：

- `layer_norm(x, eps=1e-5)` — 减去均值，除以标准差。
- `rms_norm(x, eps=1e-6)` — 除以 RMS。不减均值。
- `gelu(x)` 和 `silu(x) * W3 x` (SwiGLU)。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` 和 `decoder_block(x, enc_out, params)`。

完整接线见 `code/main.py`。

### 步骤 2：连接一个 2 层编码器和一个 2 层解码器

将它们堆叠起来。将编码器输出传入解码器的每个交叉注意力层。在输出投影前添加一个最终的层归一化。

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

### 步骤 3：在一个简单示例上运行前向传播

输入一个 6 token 的源序列和一个 5 token 的目标序列。验证输出形状为 `(5, vocab)`。不进行训练——本课是关于架构，而非损失。

### 步骤 4：替换为 RMSNorm + SwiGLU

用 RMSNorm 和 SwiGLU 替换 LayerNorm 和 ReLU-FFN。确认形状仍然匹配。这是通过一个函数替换实现的 2026 年现代化。

## 使用它

PyTorch/TF 参考实现：`nn.TransformerEncoderLayer`， `nn.TransformerDecoderLayer`。但大多数 2026 年的生产代码会自己编写模块，因为：

- Flash Attention 在注意力内部调用，而不是通过 `nn.MultiheadAttention`。
- GQA / MLA 不在标准库参考中。
- RoPE、RMSNorm、SwiGLU 不是 PyTorch 的默认值。

HF `transformers` 有你应该阅读的清晰参考模块：`modeling_llama.py` 是规范的 2026 年仅解码器模块。它大约 500 行，值得通读一次。

**编码器 vs 解码器 vs 编码器-解码器 — 何时选择：**

| 需求 | 选择 | 示例 |
|------|------|------|
| 分类、嵌入、文本问答 | 仅编码器 | BERT, DeBERTa, ModernBERT |
| 文本生成、聊天、代码、推理 | 仅解码器 | GPT, Llama, Claude, Qwen |
| 结构化输入 → 结构化输出（翻译、摘要） | 编码器-解码器 | T5, BART, Whisper |

仅解码器架构主导了语言任务，因为它扩展性最干净，能同时处理理解和生成。当输入具有明确的“源序列”身份时（翻译、语音识别、结构化任务），编码器-解码器仍是最佳选择。

## 部署它

见 `outputs/skill-transformer-block-reviewer.md`。该技能根据 2026 年默认值检查新的 Transformer 模块实现，并标记缺失部分（pre-norm, RoPE, RMSNorm, GQA, FFN 扩展比率）。

## 练习题

1.  **简单。** 计算你的 encoder_block 在 `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` 处的参数。通过实现该模块并使用 `sum(p.numel() for p in block.parameters())` 来验证。
2.  **中等。** 从 post-norm 切换到 pre-norm。初始化两者，并在随机输入上堆叠 12 层后测量激活范数。Post-norm 的激活应该会爆炸；pre-norm 的应保持有界。
3.  **困难。** 在一个简单的复制任务上（复制 `x` 并反转）实现一个 4 层的编码器-解码器。训练 100 步。报告损失。替换为 RMSNorm + SwiGLU + RoPE — 损失是否下降？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 模块 (Block) | “一个 Transformer 层” | 由残差连接包裹的 归一化 + 注意力 + 归一化 + FFN 堆栈。 |
| 残差 (Residual) | “跳跃连接” | `x + f(x)` 输出；使梯度能够在深层堆栈中流动。 |
| Pre-norm | “归一化在前，不在后” | 现代：`x + sublayer(LN(x))`。无需预热技巧即可训练更深。 |
| RMSNorm | “没有均值的 LayerNorm” | 除以 RMS；少一次操作，经验上同样稳定。 |
| SwiGLU | “大家都换用的 FFN” | `Swish(W1 x) ⊙ W3 x → W2`。在语言模型 ppl 上优于 ReLU/GELU。 |
| 交叉注意力 (Cross-attention) | “解码器如何看到编码器” | 多头注意力，其中 Q 来自解码器，K/V 来自编码器输出。 |
| FFN 扩展 (FFN expansion) | “中间 MLP 有多宽” | 隐藏层大小与 d_model 的比率，通常为 4（LayerNorm）或 2.6（SwiGLU）。 |
| 无偏置 (Bias-free) | “去掉 +b 项” | 现代技术栈省略线性层中的偏置；略微提升 ppl，模型更小。 |

## 扩展阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始模块规范。
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — 为什么 pre-norm 在深层结构中优于 post-norm。
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU 论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 规范的 2026 年仅解码器模块。