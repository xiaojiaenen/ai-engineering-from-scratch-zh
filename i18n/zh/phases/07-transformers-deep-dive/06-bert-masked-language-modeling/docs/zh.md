# BERT — 掩码语言建模

> GPT 预测下一个词。BERT 预测一个缺失的词。一字之差 — 却开启了整个嵌入技术的半代人。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 7 · 05（完整 Transformer），阶段 5 · 02（文本表示）
**时间：** 约 45 分钟

## 问题背景

2018 年，每个 NLP 任务 — 情感分析、命名实体识别、问答、蕴涵 — 都在自己的标注数据上从头训练自己的模型。不存在一个可以微调的预训练"理解英语"检查点。ELMo（2018）展示了可以用双向 LSTM 预训练上下文嵌入；它有所帮助，但泛化性不足。

BERT（Devlin 等人，2018）提出了一个问题：如果我们将一个 transformer 编码器在网络上所有的句子之上进行训练，并迫使它从两侧上下文预测缺失的词，会怎样？然后在你的下游任务上微调一个分类头。参数效率之高令人震惊。

结果：在 18 个月内，BERT 及其变体（RoBERTa、ALBERT、ELECTRA）主导了所有已有的 NLP 排行榜。到 2020 年，地球上的每个搜索引擎、内容审核管道和语义搜索系统内部都有一个 BERT。

2026 年，纯编码器模型仍然是分类、检索和结构化提取的正确工具 — 它们每个 token 的推理速度比解码器快 5–10 倍，其嵌入是每一个现代检索栈的骨干。ModernBERT（2024 年 12 月）将架构推进到 8K 上下文，使用了 Flash Attention + RoPE + GeGLU。

## 概念

![掩码语言建模：选择 token、掩码它们、预测原始 token](../assets/bert-mlm.svg)

### 训练信号

取一句话：`the quick brown fox jumps over the lazy dog`。

随机掩码 15% 的 token：

```
输入： the [MASK] brown fox jumps [MASK] the lazy dog
目标： the  quick brown fox jumps  over  the lazy dog
```

训练模型预测掩码位置上的原始 token。由于编码器是双向的，预测位置 1 的 `[MASK]` 可以使用位置 2+ 的 `brown fox jumps`。这是 GPT 做不到的事情。

### BERT 的掩码规则

在被选中用于预测的 15% token 中：

- 80% 替换为 `[MASK]`。
- 10% 替换为随机 token。
- 10% 保持不变。

为什么不能总是 `[MASK]`？因为 `[MASK]` 在推理时从不出现。如果训练模型在 100% 的掩码位置都期望看到 `[MASK]`，就会在预训练和微调之间产生分布偏移。10% 随机 + 10% 保持不变，让模型保持诚实。

### 下一句预测（NSP）— 以及它为何被丢弃

原始 BERT 还训练了 NSP：给定两个句子 A 和 B，预测 B 是否跟在 A 后面。RoBERTa（2019）通过消融实验表明 NSP 有害无益。现代编码器跳过了它。

### 2026 年的变化：ModernBERT

2024 年的 ModernBERT 论文用 2026 年的原语重建了模块：

| 组件 | 原始 BERT（2018） | ModernBERT（2024） |
|------|-------------------|---------------------|
| 位置编码 | 学习式绝对位置 | RoPE |
| 激活函数 | GELU | GeGLU |
| 归一化 | LayerNorm | 预归一化 RMSNorm |
| 注意力 | 全稠密 | 交替局部（128）+ 全局 |
| 上下文长度 | 512 | 8192 |
| 分词器 | WordPiece | BPE |

与 2018 年的堆栈不同，它原生支持 Flash Attention。在序列长度 8K 时，推理比 DeBERTa-v3 快 2–3 倍，且 GLUE 分数更高。

### 2026 年仍选择编码器的用例

| 任务 | 为什么编码器胜过解码器 |
|------|------------------------|
| 检索 / 语义搜索嵌入 | 双向上下文 = 每个 token 更好的嵌入质量 |
| 分类（情感、意图、毒性） | 一次前向传播；无生成开销 |
| NER / token 标注 | 逐位置输出，原生双向 |
| 零样本蕴涵（NLI） | 编码器上方的分类头 |
| RAG 重排序器 | 交叉编码器打分，比 LLM 重排序器快 10 倍 |

```figure
transformer-residual
```

## 动手构建

### 步骤 1：掩码逻辑

参见 `code/main.py`。函数 `create_mlm_batch` 接受一个 token ID 列表、词表大小和掩码概率。返回输入 ID（应用了掩码）和标签（仅在掩码位置有值，其余为 -100 — PyTorch 的 ignore index 约定）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: 保持原始
    return input_ids, labels
```

### 步骤 2：在小型语料上运行 MLM 预测

在 20 个词的词表和 200 个句子上训练一个 2 层编码器 + MLM 头。不进行梯度更新 — 我们只做前向传播的合理性检查。完整训练需要 PyTorch。

### 步骤 3：比较掩码类型

展示三路规则如何让模型在没有 `[MASK]` 的情况下仍然可用。在未掩码句子和掩码句子上进行预测。两者都应产生合理的 token 分布，因为模型在训练中看到了两种模式。

### 步骤 4：微调分类头

用玩具情感数据集上的分类头替换 MLM 头。只有分类头在训练；编码器保持冻结。这是每个 BERT 应用遵循的模式。

## 使用它

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**嵌入模型是微调后的 BERT。** `sentence-transformers` 模型如 `all-MiniLM-L6-v2` 是用对比损失训练的 BERT。编码器是一样的。损失函数变了。

**交叉编码器重排序器也是微调后的 BERT。** 对 `[CLS] query [SEP] doc [SEP]` 进行配对分类。query 和 doc 之间的双向注意力正是赋予交叉编码器相对于双编码器质量优势的原因。

**2026 年何时不选 BERT。** 任何生成任务。编码器没有合理的方式来自回归地生成 token。此外：任何参数量低于 1B 的任务，小解码器可以用更高的灵活性匹配质量（Phi-3-Mini、Qwen2-1.5B）。

## 交付它

参见 `outputs/skill-bert-finetuner.md`。该 skill 为一个新分类或提取任务定义了 BERT 微调的范围（骨干选择、头规格、数据、评估、停止条件）。

## 练习

1. **简单。** 运行 `code/main.py` 并打印 10,000 个 token 上的掩码分布。确认约 15% 被选中，其中约 80% 变为 `[MASK]`。
2. **中等。** 实现整词掩码：如果一个词被分词为子词，则一起掩码所有子词或都不掩码。测量这是否提高了 500 句语料上的 MLM 准确率。
3. **困难。** 在一个公开数据集的 10,000 个句子上训练一个小型（2 层，d=64）BERT。对 SST-2 情感数据微调 `[CLS]` token。与参数量匹配的纯解码器基线进行比较 — 哪个获胜？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| MLM | "掩码语言建模" | 训练信号：随机将 15% 的 token 替换为 `[MASK]`，预测原始 token。 |
| 双向 | "两边都看" | 编码器注意力没有因果掩码 — 每个位置都能看到其他所有位置。 |
| `[CLS]` | "池化 token" | 特殊 token，前置到每个序列开头；其最终嵌入用作句子级表示。 |
| `[SEP]` | "分隔符" | 分隔配对序列（如 query/doc、句子 A/B）。 |
| NSP | "下一句预测" | BERT 的第二个预训练任务；在 RoBERTa 中被证明无用，2019 年后被丢弃。 |
| 微调 | "适配任务" | 保持编码器基本冻结；在其上方训练一个小头以用于下游任务。 |
| 交叉编码器 | "重排序器" | 一个 BERT，同时接收 query 和 doc 作为输入，输出相关性分数。 |
| ModernBERT | "2024 年刷新" | 用 RoPE、RMSNorm、GeGLU、交替局部/全局注意力重建的编码器，支持 8K 上下文。 |

## 延伸阅读

- [Devlin 等人（2018）。BERT：用于语言理解的深度双向 Transformer 预训练](https://arxiv.org/abs/1810.04805) — 原始论文。
- [Liu 等人（2019）。RoBERTa：一种鲁棒优化的 BERT 预训练方法](https://arxiv.org/abs/1907.11692) — 如何正确训练 BERT；废除了 NSP。
- [Clark 等人（2020）。ELECTRA：将文本编码器预训练为判别器而非生成器](https://arxiv.org/abs/2003.10555) — 替换 token 检测在相同计算量下优于 MLM。
- [Warner 等人（2024）。更智能、更好、更快、更长：一种现代双向编码器](https://arxiv.org/abs/2412.13663) — ModernBERT 论文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — 编码器参考实现。
