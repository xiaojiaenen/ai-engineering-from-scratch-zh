# 子词分词 — BPE、WordPiece、Unigram、SentencePiece

> 词分词器在未见词汇上会卡住。字符分词器会爆炸性增长序列长度。子词分词器则折中处理。每个现代大语言模型都搭载了其中之一。

**类型：** 学习
**语言：** Python
**前提知识：** 第五阶段 · 01（文本处理），第五阶段 · 04（GloVe / FastText / 子词）
**时间：** 约60分钟

## 问题所在

你的词表有50,000个词。用户输入了“untokenizable”。你的分词器返回了`[UNK]`。模型现在对这个词毫无信号。更糟糕的是：你的语料库中第90百分位的文档有40个罕见词，这意味着每份文档会丢弃40比特的信息。

子词分词解决了这个问题。常见词保持为单个token。罕见词分解为有意义的部分：`untokenizable` → `un`, `token`, `izable`。训练数据覆盖一切，因为任何字符串最终都是字节序列。

2026年，每个前沿大语言模型都搭载了三种算法（BPE、Unigram、WordPiece）中的一种，并由三个库（tiktoken、SentencePiece、HF Tokenizers）之一包装。你无法发布一个语言模型而不选择其中之一。

## 核心概念

![BPE vs Unigram vs WordPiece, 逐字符对比](../assets/subword-tokenization.svg)

**BPE（字节对编码）。** 从字符级词表开始。统计每对相邻字符。将最频繁的词对合并为一个新token。重复直到达到目标词表大小。这是主导算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**字节级BPE。** 算法相同，但工作在原始字节（256个基础token）而不是Unicode字符上。保证零`[UNK]` token——任何字节序列都可编码。GPT-2使用50,257个token（256个字节 + 50,000次合并 + 1个特殊token）。

**Unigram。** 从一个巨大的词表开始。为每个token分配一个unigram概率。迭代地剪除那些移除后最不会增加语料库对数似然度的token。在推理时是概率性的：可以对分词进行采样（这对于通过子词正则化进行数据增强很有用）。被T5、mBART、ALBERT、XLNet、Gemma使用。

**WordPiece。** 合并那些能最大化训练语料库似然度（而非原始频率）的词对。被BERT、DistilBERT、ELECTRA使用。

**SentencePiece vs tiktoken。** SentencePiece是一个*训练*词表（BPE或Unigram）的库，它直接处理原始Unicode文本，并将空格编码为`▁`。tiktoken是OpenAI的快速*编码器*，针对预构建的词表工作；它不进行训练。

经验法则：

- **训练新词表：** SentencePiece（多语言，无需预分词）或HF Tokenizers。
- **针对GPT词表进行快速推理：** tiktoken（cl100k_base, o200k_base）。
- **两者兼顾：** HF Tokenizers——一个库，涵盖训练和服务。

## 动手构建

### 步骤 1：从零实现BPE

见`code/main.py`。循环如下：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

算法编码了三个事实。`</w>` 标记词尾，这样 "low"（后缀）和 "lower"（前缀）得以保持区别。频率加权使得高频词对较早胜出。合并列表是有序的——推理时按训练顺序应用合并。

### 步骤 2：使用学到的合并规则进行编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素实现的时间复杂度是 O(n·|merges|)。生产环境实现（tiktoken、HF Tokenizers）使用合并等级查找和优先队列，运行时间接近线性。

### 步骤 3：实践中的SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：无需预分词，空格被编码为`▁`，`character_coverage` 控制是保留罕见字符还是将其映射为`<unk>` 的积极程度。

### 步骤 4：用于OpenAI兼容词表的tiktoken

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

仅用于编码。速度很快（Rust后端）。与GPT-4/5的分词完全匹配，适用于字节计数、成本估算、上下文窗口预算。

## 2026年仍然存在的陷阱

- **分词器漂移。** 在词表A上训练，却部署到词表B。Token ID不同；模型输出垃圾。在CI中检查`tokenizer.json`哈希值。
- **空格歧义。** BPE中 "hello" 和 " hello" 产生不同的token。始终明确指定`add_special_tokens`和`add_prefix_space`。
- **多语言训练不足。** 以英语为主的语料库产生的词表，会将非拉丁文字拆分成多5-10倍的token。在GPT-3.5上，相同的提示在日语/阿拉伯语中成本要高5-10倍。o200k_base 部分解决了这个问题。
- **表情符号拆分。** 一个表情符号可能占用5个token。在预算上下文时，要检查点表情符号的处理方式。

## 实际应用

2026年的技术栈：

| 场景 | 选择 |
|------|------|
| 从零训练单语模型 | HF Tokenizers (BPE) |
| 训练多语言模型 | SentencePiece (Unigram, `character_coverage=0.9995`) |
| 提供OpenAI兼容的API服务 | tiktoken (`o200k_base` 用于GPT-4+) |
| 领域特定词表（代码、数学、蛋白质） | 在领域语料库上训练自定义BPE，与基础词表合并 |
| 边缘推理，小型模型 | Unigram（较小的词表效果更好） |

词表大小是一个伸缩性决策，而非常数。粗略启发法：参数少于10亿用32k，1-100亿参数用50-100k，多语言/前沿模型用200k以上。

## 部署它

保存为`outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## 练习

1. **简单。** 在`code/main.py`的小型语料库上训练一个500次合并的BPE。编码三个留出词。有多少恰好产生了1个token，有多少产生了>1个token？
2. **中等。** 比较`cl100k_base`、`o200k_base`和一个你用vocab=32k训练的SentencePiece BPE在100个英语维基百科句子上的token计数。报告每种方法的压缩比。
3. **困难。** 用BPE、Unigram和WordPiece训练相同的语料库。在一个小的情感分类器上使用每种分词方法时，测量下游准确性。该选择是否使F1分数移动超过1个点？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| BPE | 字节对编码 | 贪心合并最频繁的字符对，直到达到目标词表大小。 |
| 字节级BPE | 永远没有未知token | 基于原始256字节的BPE；GPT-2 / Llama 使用此方法。 |
| Unigram | 概率分词器 | 使用对数似然度从大型候选集中剪枝；被T5、Gemma使用。 |
| SentencePiece | 处理空格的那个 | 在原始文本上训练BPE/Unigram的库；空格编码为`▁`。 |
| tiktoken | 快的那个 | OpenAI的、基于Rust的BPE编码器，用于预构建词表。不训练。 |
| 合并列表 | 神奇数字 | `(a, b) → ab`合并的有序列表；推理时按序应用。 |
| 字符覆盖 | 多罕见算太罕见？ | 分词器必须覆盖训练语料库中字符的比例；典型值约0.9995。 |

## 延伸阅读

- [Sennrich, Haddow, Birch (2015). 使用子词单元进行罕见词的神经机器翻译](https://arxiv.org/abs/1508.07909) — BPE论文。
- [Kudo (2018). 使用Unigram语言模型的子词正则化](https://arxiv.org/abs/1804.10959) — Unigram论文。
- [Kudo, Richardson (2018). SentencePiece: 一个简单且语言无关的子词分词器](https://arxiv.org/abs/1808.06226) — 该库。
- [Hugging Face — 分词器概述](https://huggingface.co/docs/transformers/tokenizer_summary) — 简明参考。
- [OpenAI tiktoken 代码库](https://github.com/openai/tiktoken) — 教程 + 编码列表。