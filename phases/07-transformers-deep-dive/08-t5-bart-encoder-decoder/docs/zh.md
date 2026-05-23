# T5、BART — 编码器-解码器模型

> 编码器负责理解，解码器负责生成。将二者重新组合，你就得到了一个专为输入 → 输出任务构建的模型：翻译、摘要、改写、转录。

**类型：** 学习  
**语言：** Python  
**前置课程：** Phase 7 · 05（完整Transformer），Phase 7 · 06（BERT），Phase 7 · 07（GPT）  
**时长：** ~45分钟

## 问题所在

仅解码器的GPT和仅编码器的BERT，各自为不同的目标精简了2017年的原始架构。但许多任务天然就是输入到输出的：

- 翻译：英文 → 法文。
- 摘要：5000个token的文章 → 200个token的摘要。
- 语音识别：音频token → 文本token。
- 结构化提取：散文 → JSON。

对于这些任务，编码器-解码器是最契合的架构。编码器为源文本生成一个密集的表示。解码器生成输出，并在每一步都对该表示进行交叉注意力。训练时，输出侧进行平移（shift-by-one）。损失函数与GPT相同，只是条件化在编码器输出之上。

两篇论文定义了现代范式：

1.  **T5**（Raffel等人，2019）。“文本到文本迁移Transformer”。将所有NLP任务重构为文本输入、文本输出。单一架构、单一词表、单一损失函数。通过掩码跨度预测进行预训练（在输入中损坏跨度，在输出中解码它们）。
2.  **BART**（Lewis等人，2019）。“双向与自回归Transformer”。去噪自编码器：通过多种方式损坏输入（打乱、掩码、删除、旋转），要求解码器重构原始文本。

在2026年，编码器-解码器格式在输入结构很重要的场景中依然存在：

- Whisper（语音 → 文本）。
- 谷歌的翻译技术栈。
- 一些具有不同上下文和编辑结构的代码补全/修复模型。
- Flan-T5及其变体，用于结构化推理任务。

仅解码器模型赢得了聚光灯，但编码器-解码器从未消失。

## 核心概念

![带交叉注意力的编码器-解码器](../assets/encoder-decoder.svg)

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

关键点是，编码器对每个输入只运行一次。解码器是自回归运行的，但在每一步都对*相同的*编码器输出进行交叉注意力。缓存编码器输出是长输入场景下免费的加速技巧。

### T5预训练 — 跨度损坏

随机选取输入中的跨度（平均长度3个token，共15%）。用一个唯一的哨兵标记替换每个跨度：`<extra_id_0>`, `<extra_id_1>`，等等。解码器仅输出损坏的跨度，并带有哨兵前缀：

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

这比预测整个序列的信号更便宜。在T5论文的消融实验中，其竞争力与MLM（BERT）和前缀语言模型（UniLM）相当。

### BART预训练 — 多噪声去噪

BART尝试了五种噪声函数：

1.  Token掩码。
2.  Token删除。
3.  文本填充（掩码一个跨度，解码器插入正确长度的内容）。
4.  句子置换。
5.  文档旋转。

结合文本填充和句子置换产生了最佳的下游任务性能。解码器总是重构原始序列。BART的输出是完整序列，而不仅仅是损坏的跨度 — 因此预训练计算量比T5更高。

### 推理

与GPT相同的自回归生成方式。贪心搜索、束搜索、top-p采样均适用。束搜索（宽度4-5）是翻译和摘要的标准方法，因为输出的分布比聊天场景更窄。

### 2026年如何选择各变体

| 任务 | 用编码器-解码器？ | 原因 |
|------|------------------|------|
| 翻译 | 是，通常 | 有清晰的源序列；输出分布固定；束搜索有效 |
| 语音转文本 | 是（Whisper） | 输入模态与输出不同；编码器处理音频特征 |
| 聊天/推理 | 否，用仅解码器 | 没有固定的“输入” — 对话本身就是序列 |
| 代码补全 | 通常不用 | 具有长上下文的仅解码器模型表现更优；如Qwen 2.5 Coder这样的代码模型是仅解码器的 |
| 摘要 | 两者皆可 | BART、PEGASUS超越了早期的仅解码器基线；现代的仅解码器LLM可以与之匹敌 |
| 结构化提取 | 两者皆可 | T5很简洁，因为“文本 → 文本”的形式吸收了任何输出格式 |

自~2022年以来的趋势是：仅解码器模型接管了编码器-解码器曾经擅长的任务，原因在于（a）指令微调的仅解码器LLM通过提示可以泛化到任何任务，（b）单一架构比双架构更易扩展，（c）RLHF假设使用解码器。编码器-解码器在输入模态不同（如语音、图像）或束搜索质量至关重要的场景中仍然占据一席之地。

## 实现它

参见`code/main.py`。我们为一个小规模语料库实现了T5风格的跨度损坏 — 这是本课最有用的部分，因为它在之后的每一个编码器-解码器预训练方案中都会出现。

### 步骤1：跨度损坏

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

目标格式遵循T5惯例：`<sent0> span0 <sent1> span1 ...`。损坏的输入将未改变的token与位于跨度位置的哨兵token交错排列。

### 步骤2：验证往返重构

给定损坏的输入和目标，重构原始句子。如果你的损坏是可逆的，那么前向传播就是明确定义的。这只是一个完整性检查 — 实际的训练不会这么做，但这个测试成本低，能发现你在跨度管理中的偏移错误。

### 步骤3：BART加噪

五个函数：`token_mask`, `token_delete`, `text_infill`, `sentence_permute`, `document_rotate`。组合其中两个并展示结果。

## 使用它

HuggingFace参考：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5的诀窍是：任务名称被放入输入文本中。同一个模型可以处理数十种任务，因为每种任务都是文本输入、文本输出。在2026年，这种模式已被指令微调的仅解码器模型所泛化，但T5首先将其规范化。

## 部署它

参见`outputs/skill-seq2seq-picker.md`。该技巧根据输入-输出结构、延迟和质量目标，为一个新任务在编码器-解码器和仅解码器模型之间进行选择。

## 练习

1.  **简单。** 运行`code/main.py`，对一个30个token的句子应用跨度损坏，验证将非哨兵源token与解码后的目标跨度连接起来能否重现原文。
2.  **中等。** 实现BART的`text_infill`噪声：用一个`<mask>` token替换随机跨度，解码器必须推断出正确的跨度长度和内容。展示一个例子。
3.  **困难。** 在一个微型的英语 → 猪拉丁语语料库（200对句子）上微调`flan-t5-small`。在一个保留的50对句子集上测量BLEU值。与在相同数据、相同计算资源下微调`Llama-3.2-1B`的结果进行比较。

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|------------|--------------|
| 编码器-解码器 | “序列到序列的Transformer” | 双堆栈：用于输入的双向编码器，和用于输出的带交叉注意力的因果解码器。 |
| 交叉注意力 | “源与目标对话的地方” | 解码器的Q × 编码器的K/V。编码器信息进入解码器的唯一位置。 |
| 跨度损坏 | “T5的预训练技巧” | 用哨兵token替换随机跨度；解码器输出这些跨度。 |
| 去噪目标 | “BART的游戏” | 对输入应用噪声函数，训练解码器重构干净的序列。 |
| 哨兵token | “`<extra_id_N>`占位符” | 用于在源序列中标记损坏跨度，并在目标序列中重新标记的特殊token。 |
| Flan | “指令微调的T5” | 在超过1,800个任务上微调的T5；使编码器-解码器在遵循指令方面具有竞争力。 |
| 束搜索 | “解码策略” | 在每一步保留top-k的部分序列；翻译/摘要的标准方法。 |
| 教师强制 | “训练时的输入” | 训练期间，将真实的前一个输出token喂给解码器，而不是采样得到的token。 |

## 延伸阅读

- [Raffel等人 (2019). 探索基于统一文本到文本Transformer的迁移学习极限](https://arxiv.org/abs/1910.10683) — T5。
- [Lewis等人 (2019). BART：用于自然语言生成、翻译和理解的去噪序列到序列预训练](https://arxiv.org/abs/1910.13461) — BART。
- [Chung等人 (2022). 扩展指令微调的语言模型](https://arxiv.org/abs/2210.11416) — Flan-T5。
- [Radford等人 (2022). 通过大规模弱监督实现稳健的语音识别](https://arxiv.org/abs/2212.04356) — Whisper，2026年典型的编码器-解码器模型。
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — 参考实现。