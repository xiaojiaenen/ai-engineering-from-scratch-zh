# T5、BART — 编码器-解码器模型

> 编码器负责理解，解码器负责生成。把它们组合在一起，就得到了专为输入 → 输出任务设计的模型：翻译、摘要、改写、转录。

**类型：** 学习
**语言：** Python
**前置知识：** 第 7 阶段 · 05（完整 Transformer）、第 7 阶段 · 06（BERT）、第 7 阶段 · 07（GPT）
**时间：** 约 45 分钟

## 问题所在

仅解码器的 GPT 和仅编码器的 BERT 各自为不同目标对 2017 年的架构做了简化。但许多任务天然就是输入-输出形式：

- 翻译：英文 → 法文。
- 摘要：5,000 token 的文章 → 200 token 的摘要。
- 语音识别：音频 token → 文本 token。
- 结构化提取：散文 → JSON。

对于这类任务，编码器-解码器是最干净的选择。编码器产生输入的稠密表示。解码器逐步生成输出，每一步都通过交叉注意力机制关注该表示。训练时在输出侧做移位一位预测。损失函数与 GPT 相同，只是以编码器输出来条件化。

两篇论文定义了现代做法：

1. **T5**（Raffel 等，2019）。「Text-to-Text Transfer Transformer」—— 将所有 NLP 任务重新表述为文本输入、文本输出。单一架构、单一词表、单一损失函数。在掩码片段预测上预训练（输入中的片段被破坏，解码器在输出中还原它们）。
2. **BART**（Lewis 等，2019）。「Bidirectional and Auto-Regressive Transformer」—— 去噪自编码器：以多种方式破坏输入（乱序、掩码、删除、旋转），让解码器重建原文。

到 2026 年，编码器-解码器格式仍在以下场景中使用：

- Whisper（语音 → 文本）。
- Google 的翻译系统。
- 具有独立上下文和编辑结构的某些代码补全/修复模型。
- Flan-T5 及其变体，用于结构化推理任务。

仅解码器吸引了更多关注，但编码器-解码器从未消失。

## 核心概念

![Encoder-decoder with cross-attention](../assets/encoder-decoder.svg)

### 前向循环

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

关键点：编码器对每个输入只运行一次。解码器自回归地运行，但在每一步都对*相同的*编码器输出进行交叉注意力计算。缓存编码器输出对长输入来说是免费的速度提升。

### T5 预训练 — 片段破坏

随机选取输入中的片段（平均长度 3 个 token，占总 token 数的 15%）。将每个片段替换为唯一的哨兵标记：`<extra_id_0>`、`<extra_id_1>` 等。解码器仅输出被破坏的片段及其哨兵前缀：

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

这比预测整个序列的代价更低。在 T5 论文的消融实验中，其效果与 MLM（BERT）和 prefix-LM（UniLM）相当。

### BART 预训练 — 多噪声去噪

BART 尝试了五种噪声函数：

1. Token 掩码。
2. Token 删除。
3. 文本填充（掩码一个片段，解码器插入正确长度的内容）。
4. 句子置换。
5. 文档旋转。

组合「文本填充 + 句子置换」在下游任务中取得了最佳结果。解码器始终重建原始序列。BART 的输出是完整序列，而非仅被破坏的片段 —— 因此预训练计算量高于 T5。

### 推理

与 GPT 相同的自回归生成方式。贪心采样、束搜索（beam search）、top-p 采样均可使用。束搜索（宽度 4–5）在翻译和摘要中是标准做法，因为输出分布比对话场景更窄。

### 2026 年何时选择各变体

| 任务 | 编码器-解码器？ | 原因 |
|------|------------------|-----|
| 翻译 | 通常选是 | 输入序列明确；输出分布固定；束搜索效果好 |
| 语音转文本 | 选是（Whisper） | 输入模态与输出不同；编码器用于提取音频特征 |
| 对话 / 推理 | 不选，仅解码器 | 没有持久的"输入"——对话本身就是序列 |
| 代码补全 | 通常不选 | 仅解码器配合长上下文更优；代码模型如 Qwen 2.5 Coder 均为仅解码器 |
| 摘要 | 两者皆可 | BART、PEGASUS 优于早期的仅解码器基线；现代仅解码器大模型已能匹敌 |
| 结构化提取 | 两者皆可 | T5 很简洁，因为"文本 → 文本"可以吸收任何输出格式 |

~2022 年以来的趋势：仅解码器接管了编码器-解码器曾经主导的任务，原因是：(a) 指令微调的仅解码器大模型通过提示泛化到任何任务，(b) 一种架构比两种更容易扩展，(c) RLHF 假设使用的是解码器。编码器-解码器则在输入模态不同（语音、图像）或束搜索质量至关重要的场景中保留。

```figure
encoder-decoder
```

## 动手实现

参见 `code/main.py`。我们针对小规模语料实现 T5 风格的片段破坏 —— 这是本课最有用的单一部分，因为自它出现以来，每个编码器-解码器预训练方案中都有类似的影子。

### 步骤 1：片段破坏

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """选取约占 mask_rate token 数的片段。返回 (corrupted_input, target)。"""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

目标格式遵循 T5 约定：`<sent0> span0 <sent1> span1 ...`。破坏后的输入在被破坏的片段位置处穿插哨兵标记，其余 token 保持不变。

### 步骤 2：验证往返

给定破坏后的输入和目标，重建原始句子。如果你的破坏过程可逆，前向传播就是有定义的。这是一种自检 —— 真实训练不会做这一步，但测试成本低廉，能捕获片段管理中的 off-by-one 错误。

### 步骤 3：BART 噪声函数

五个函数：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。组合其中两个并展示结果。

## 使用它

HuggingFace 参考实现：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5 的技巧：任务名放入输入文本。同一个模型能处理数十种任务，因为每种任务都是文本输入、文本输出。到 2026 年，这一模式已被指令微调的仅解码器模型所泛化，但 T5 是最先将其系统化的。

## 交付

参见 `outputs/skill-seq2seq-picker.md`。该技能会根据输入-输出结构、延迟和质量目标，在编码器-解码器和仅解码器之间做出选择。

## 练习

1. **简单。** 运行 `code/main.py`，对一句 30 token 的句子施加片段破坏，验证将非哨兵源 token 与解码出的目标片段拼接后能否还原原始句子。
2. **中等。** 实现 BART 的 `text_infill` 噪声：用单个 `<mask>` token 替换随机片段，解码器必须推断出正确的片段长度和内容。展示一个示例。
3. **困难。** 在小型英文 → pig-Latin 语料（200 对）上微调 `flan-t5-small`。在保留的 50 对集合上测量 BLEU。与在同一数据上使用相同计算量微调 `Llama-3.2-1B` 的结果进行对比。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 编码器-解码器 | "Seq2seq transformer" | 两个堆栈：双向编码器用于输入，带交叉注意力的因果解码器用于输出。 |
| 交叉注意力 | "源与目标的对话之处" | 解码器的 Q × 编码器的 K/V。编码器信息进入解码器的唯一途径。 |
| 片段破坏 | "T5 的预训练技巧" | 用哨兵 token 替换随机片段；解码器输出这些片段。 |
| 去噪目标 | "BART 的游戏" | 对输入施加噪声函数，训练解码器重建干净序列。 |
| 哨兵 token | "占位符 `<extra_id_N>`" | 特殊 token，在源中标记被破坏的片段，在目标中重新标记它们。 |
| Flan | "指令微调的 T5" | 在超过 1,800 个任务上微调的 T5；让编码器-解码器在指令遵循上具备竞争力。 |
| 束搜索 | "解码策略" | 每一步保留 top-k 个部分序列；翻译/摘要的标准做法。 |
| 教师强制 | "训练时的输入" | 训练时，向解码器输入真实的上一个输出 token，而非采样得到的 token。 |

## 延伸阅读

- [Raffel 等（2019）。Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5。
- [Lewis 等（2019）。BART: Denoising Sequence-to-Sequence Pre-Training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART。
- [Chung 等（2022）。Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5。
- [Radford 等（2022）。Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper，2026 年编码器-解码器的典范。
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — 参考实现。
