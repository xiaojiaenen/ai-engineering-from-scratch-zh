# 子词分词 — BPE、WordPiece、Unigram、SentencePiece

> 词级分词器遇到生僻词会卡壳。字符级分词器会让序列长度爆炸。子词分词器折中两者。所有现代 LLM 都基于其中之一。

**类型：** Learn
**语言：** Python
**前置知识：** Phase 5 · 01（文本处理）、Phase 5 · 04（GloVe / FastText / 子词）
**预计时间：** 约 60 分钟

## 问题所在

你的词表有 5 万个词。用户输入了 `"untokenizable"`。你的分词器返回 `[UNK]`。模型对这个词没有任何信号。更糟的是：你语料库中第 90 百分位的文档包含 40 个生僻词，这意味着每个文档会丢失 40 比特的信息。

子词分词解决了这个问题。常见词保持为单个 token。罕见词分解为有意义的片段：`untokenizable` → `un`、`token`、`izable`。训练数据能覆盖一切，因为任何字符串最终都是一系列字节。

2026 年所有前沿 LLM 都基于三种算法之一（BPE、Unigram、WordPiece），并打包在三种库之一中（tiktoken、SentencePiece、HF Tokenizers）。不选一种就无法交付语言模型。

## 概念

![BPE vs Unigram vs WordPiece，逐字符对比](../assets/subword-tokenization.svg)

**BPE（Byte-Pair Encoding，字节对编码）。** 从字符级词表开始。统计所有相邻词对。将最高频的词对合并为新 token。重复直到达到目标词表大小。主流算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**字节级 BPE。** 算法相同，但操作对象是原始字节（256 个基础 token）而非 Unicode 字符。保证零 `[UNK]` token——任何字节序列都能编码。GPT-2 使用 50,257 个 token（256 个字节 + 50,000 次合并 + 1 个特殊 token）。

**Unigram。** 从超大词表开始。为每个 token 分配一个 Unigram 概率。迭代剪枝那些移除后对语料库对数似然增加最小的 token。推理时具有概率性：可以采样分词结果（用于通过子词正则化进行数据增强）。T5、mBART、ALBERT、XLNet、Gemma 使用。

**WordPiece。** 合并使训练语料库似然最大化的词对，而非原始频率。BERT、DistilBERT、ELECTRA 使用。

**SentencePiece vs tiktoken。** SentencePiece 是直接在原始 Unicode 文本上*训练*词表的库（BPE 或 Unigram），将空白编码为 `▁`。tiktoken 是 OpenAI 针对预构建词表的快速*编码器*；它不训练。

经验法则：

- **训练新词表：** SentencePiece（多语言，无需预分词）或 HF Tokenizers。
- **针对 GPT 词表快速推理：** tiktoken（cl100k_base、o200k_base）。
- **两者兼顾：** HF Tokenizers——一个库搞定训练和服务。

```figure
bpe-merge
```

## 动手实现

### 步骤 1：从零实现 BPE

见 `code/main.py`。核心循环：

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

算法编码了三个要点。`</w>` 标记单词结尾，使 `"low"`（后缀）和 `"lower"`（前缀）保持区分。频率加权使高频词对优先胜出。合并列表是有顺序的——推理时按训练顺序应用合并。

### 步骤 2：用学到的合并规则编码

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

朴素实现是 O(n·|merges|)。生产级实现（tiktoken、HF Tokenizers）使用合并秩查找和优先级队列，运行时间接近线性。

### 步骤 3：SentencePiece 实战

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # 或 "unigram"
    character_coverage=0.9995, # CJK 语言可降低（例如英文 0.9995，日文 0.995）
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：无需预分词，空格被编码为 `▁`，`character_coverage` 控制稀有字符是被保留还是映射为 `<unk>` 的激进程度。

### 步骤 4：使用 tiktoken 处理 OpenAI 兼容词表

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

仅编码。快速（Rust 后端）。与 GPT-4/5 的分词精确匹配，用于字节计数、成本估算、上下文窗口预算。

## 2026 年仍会踩坑的点

- **分词器漂移。** 用词表 A 训练，却针对词表 B 部署。Token ID 不同；模型输出变成乱码。在 CI 中检查 `tokenizer.json` 的哈希值。
- **空白歧义。** BPE 中 `"hello"` 和 `" hello"` 会产生不同的 token。务必显式指定 `add_special_tokens` 和 `add_prefix_space`。
- **多语言欠训练。** 以英文为主的语料库产生的词表会把非拉丁字母脚本拆成 5-10 倍的 token。在 GPT-3.5 上，同一提示词用日语/阿拉伯语的成本是英语的 5-10 倍。o200k_base 部分解决了这个问题。
- **表情符号拆分。** 单个 emoji 可能占用 5 个 token。预算化上下文时需检查 emoji 处理方式。

## 使用指南

2026 年的技术栈：

| 场景 | 选择 |
|------|------|
| 从头训练单语模型 | HF Tokenizers（BPE） |
| 训练多语言模型 | SentencePiece（Unigram，`character_coverage=0.9995`） |
| 服务 OpenAI 兼容 API | tiktoken（GPT-4+ 用 `o200k_base`） |
| 领域特定词表（代码、数学、蛋白质） | 在领域语料上训练自定义 BPE，再与基础词表合并 |
| 边缘推理、小模型 | Unigram（较小词表效果更好） |

词表大小是扩展决策，不是固定常数。粗略经验法则：<1B 参数用 32k，1-10B 参数用 50-100k，多语言/前沿模型用 200k+。

## 交付物

保存为 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: 根据给定语料和部署目标，选择分词算法、词表大小、库。
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

给定语料（规模、语言、领域）和部署目标（从头训练 / 微调 / API 兼容推理），输出：

1. 算法。BPE、Unigram 或 WordPiece。一句话理由。
2. 库。SentencePiece、HF Tokenizers 或 tiktoken。理由。
3. 词表大小。四舍五入到最近的 1k。理由与模型大小和语言覆盖率相关。
4. 覆盖率设置。`character_coverage`、`byte_fallback`、特殊 token 列表。
5. 验证方案。保留集中的平均词 token 数、OOV 率、压缩比、往返解码相等性。

拒绝在含有稀有脚本内容的语料上训练 character_coverage <0.995 的分词器。拒绝在 CI 中没有冻结的 `tokenizer.json` 哈希检查的词表上交付。将词表低于 16k 的单语分词器标记为可能规格不足。
```

## 练习题

1. **简单。** 在 `code/main.py` 的小型语料上训练 500 次合并的 BPE。编码三个保留词。多少个恰好产生 1 个 token，多少个产生 >1 个 token？
2. **中等。** 在 100 个英文维基百科句子上比较 `cl100k_base`、`o200k_base` 和你用 vocab=32k 训练的 SentencePiece BPE 的 token 数量。报告各自的压缩比。
3. **困难。** 用 BPE、Unigram 和 WordPiece 分别训练同一语料。在每个分词器上用于小型情感分类器时测量下游准确率。选择是否让 F1 分数提升超过 1 个点？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| BPE | Byte-Pair Encoding | 贪婪合并最高频字符对，直到达到目标词表大小。 |
| Byte-level BPE | 永远不会有未知 token | 对原始 256 字节进行 BPE；GPT-2 / Llama 使用此方式。 |
| Unigram | 概率分词器 | 使用对数似然从大量候选集中剪枝；T5、Gemma 使用。 |
| SentencePiece | 那个处理空白的 | 直接在原始文本上训练 BPE/Unigram 的库；空格编码为 `▁`。 |
| tiktoken | 那个快速的 | OpenAI 基于 Rust 的 BPE 编码器，用于预构建词表。不训练。 |
| Merge list | 魔法数字 | 有序的 `(a, b) → ab` 合并列表；推理时按顺序应用。 |
| Character coverage | 多稀有算稀有？ | 分词器必须覆盖的训练语料中字符比例；通常约 0.9995。 |

## 延伸阅读

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE 论文。
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — Unigram 论文。
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — 库论文。
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — 简明参考。
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — 食谱 + 编码列表。
