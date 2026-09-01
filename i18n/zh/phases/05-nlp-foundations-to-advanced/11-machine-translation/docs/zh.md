# 机器翻译

> 翻译是一项为 NLP 研究提供了三十年资金支持的任务，如今仍在继续为此买单。

**类型：** 构建
**语言：** Python
**前置知识：** 第 5 阶段 · 10（注意力机制），第 5 阶段 · 04（GloVe、FastText、子词）
**时间：** 约 75 分钟

## 问题定义

模型读取一种语言的句子，并生成另一种语言的句子。长度各异，语序各异。某些源语言词汇对应多个目标词汇，反之亦然。习语无法一一对应。"I miss you" 的法语是 "tu me manques"——字面意思是"你对我来说是缺乏的"。那种情况下，词级对齐无从幸存。

机器翻译是迫使 NLP 发明编码器-解码器、注意力机制、Transformer，并最终催生整个 LLM 范式的任务。每一步进步都源于翻译质量可衡量，且人机差距顽固难消。

本课跳过历史课，教授 2026 年的工作流水线：预训练多语言编码器-解码器（NLLB-200 或 mBART）、子词分词、束搜索、BLEU 与 chrF 评估，以及少数仍会漏网到生产环境的故障模式。

## 概念

![MT pipeline: tokenize → encode → decode with attention → detokenize](../assets/mt-pipeline.svg)

现代机器翻译是一种在平行文本上训练的 Transformer 编码器-解码器。编码器按源语言的分词方式读取输入。解码器每次生成一个子词，通过交叉注意力（第 10 课）使用编码器的输出。解码使用束搜索以避免贪心解码的陷阱。输出经去分词、去真性处理后，与参考译文进行评分。

三个关键决策决定实际世界的 MT 质量。

- **分词器。** 在混合语言语料上训练的 SentencePiece BPE。跨语言共享词汇表是实现 NLLB 零样本配对的关键。
- **模型规模。** NLLB-200 distilled 600M 可装在笔记本电脑上运行。NLLB-200 3.3B 是已发布的生产默认值。54.5B 是研究天花板。
- **解码策略。** 通用内容使用束宽 4-5。长度惩罚以避免输出过短。约束解码用于需要术语一致性的场景。

```figure
seq2seq-alignment
```

## 动手构建

### 步骤 1：调用预训练 MT 模型

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

这里有三点关键。`src_lang` 告诉分词器使用哪种脚本和分段方式。`forced_bos_token_id` 告诉解码器生成哪种语言。两者都是 NLLB 特有的技巧；mBART 和 M2M-100 使用各自的约定，它们不可互换。

### 步骤 2：BLEU 与 chrF

BLEU 衡量输出与参考译文之间的 n-gram 重叠。对四种参考 n-gram 尺寸（1-4），取精确率的几何平均，并对过短输出施加长度惩罚。分数范围在 [0, 100]。常用，但难以解读：30 BLEU 算"可用"；40 算"好"；50 算"卓越"；低于 1 BLEU 的差异是噪音。

chrF 衡量字符级 F 分数。对于形态丰富的语言，BLEU 容易低估匹配，chrF 对此更敏感。常与 BLEU 一起报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

始终使用 `sacrebleu`。它标准化了分词，使不同论文中的分数可比。自己实现 BLEU 计算是导致误导性基准测试的主要原因。

### 2026 年三层评估体系

现代 MT 评估使用三种互补的度量家族。至少使用其中两种。

- **启发式**（BLEU、chrF）。快速、基于参考、可解释，但对 paraphrase 不敏感。用于历史对比和回归检测。
- **学习式**（COMET、BLEURT、BERTScore）。在人工判断上训练的神经网络模型；比较翻译与源文和参考译文的语义相似度。自 2023 年以来，COMET 与 MT 研究的相关性最高，是 2026 年注重质量的生产默认选择。
- **LLM 即裁判**（无参考）。提示大型模型从流利度、充分性、语气、文化适当性等方面对翻译进行评分。当评判标准设计良好时，GPT-4 作为裁判与人类一致性约 80%。用于不存在参考译文的开放内容场景。

实用的 2026 技术栈：`sacrebleu` 用于 BLEU 和 chrF，`unbabel-comet` 用于 COMET，以及一个带提示的 LLM 用于最终面向人类的结果信号。在信任生产数据之前，先用 50-100 个手工标注样本校准每种指标。

无参考指标（COMET-QE、BLEURT-QE、LLM 即裁判）使你能够在没有参考译文的情况下评估翻译，这对于不存在参考译文的长尾语言对至关重要。

### 步骤 3：生产环境中常见故障

上述工作流水线有 80% 的情况能流畅翻译，但剩余 20% 会静默失败。已知故障模式：

- **幻觉。** 模型编造了源文中不存在的内容。在陌生领域词汇中常见。症状：输出流畅，但声称了源文未陈述的事实。缓解方法：对领域术语使用约束解码，对受监管内容让人工审核，监控输出远长于输入的情况。
- **目标语言偏移。** 模型翻译到了错误的语言。NLLB 对罕见语言对在这一点上 surprisingly 容易出错。缓解方法：验证 `forced_bos_token_id`，并始终在输出时配合语言 ID 模型检查。
- **术语漂移。** "Sign up" 在文档 1 中译为 "s'inscrire"，在文档 2 中译为 "créer un compte"。对于 UI 文本和用户可见字符串，一致性比原始质量更重要。缓解方法：词汇表约束解码或后编辑词典。
- **正式度不匹配。** 法语的 "tu" 与 "vous"、日语的敬语层级。模型会选择训练数据中更常见的形式。对于面向客户的内容，这通常是错误的。缓解方法：如果模型支持，使用形式化 token 的前缀提示，或在纯正式语料上微调一个小模型。
- **短输入时的长度膨胀。** 非常短的输入句子常产生过长的翻译，因为长度惩罚在源词少于约 5 个时急剧失效。缓解方法：根据源长度设置硬性最大长度上限。

### 步骤 4：针对特定领域的微调

预训练模型是通用型选手。法律、医疗或游戏对话翻译从领域平行数据的微调中可获得实质性提升。配方并不复杂：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

几千条高质量的平行示例胜过几十万条嘈杂的网页抓取数据。训练数据质量是生产中最大的杠杆。

## 使用它

2026 年 MT 生产技术栈：

| 使用场景 | 推荐起点 |
|---------|---------------------------|
| 任意到任意，200 种语言 | `facebook/nllb-200-distilled-600M`（笔记本）或 `nllb-200-3.3B`（生产） |
| 以英语为中心，高质量，50 种语言 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短文本，低成本推理，英法/德/西 | Helsinki-NLP / Marian 模型 |
| 低延迟浏览器端 | ONNX 量化 Marian（约 50 MB） |
| 最高质量，愿意付费 | GPT-4 / Claude / Gemini 配合翻译提示 |

截至 2026 年，LLM 在若干语言对上已超越专用 MT 模型，尤其是在习语内容和长上下文方面。代价是每 token 成本和延迟。当上下文长度、风格一致性或通过提示实现的领域适配比吞吐量更重要时，选择 LLM。

## 交付

保存为 `outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## 练习

1. **简单。** 使用 `nllb-200-distilled-600M` 将一个 5 句英文段落翻译成法文再回译成英文。衡量往返翻译与原文的接近程度。你应该能看到语义保留但词汇选择产生漂移。
2. **中等。** 使用 `fasttext lid.176` 或 `langdetect` 实现翻译输出的语言 ID 检查。将其集成到 MT 调用中，使目标语言偏移的生成在返回前被捕获。
3. **困难。** 在你选择的 5,000 对领域语料上微调 `nllb-200-distilled-600M`。在微调前后在保留集上测量 BLEU。报告哪些类型的句子得到改进，哪些出现退步。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-------------------------|------------------------|
| BLEU | 翻译分数 | 带长度惩罚的 n-gram 精确率。[0, 100]。 |
| chrF | 字符 F 分数 | 字符级 F 分数。对形态丰富语言更敏感。 |
| NMT | 神经机器翻译 | 在平行文本上训练的 Transformer 编码器-解码器。2017 年以后的默认方案。 |
| NLLB | No Language Left Behind | Meta 的 200 语言 MT 模型系列。 |
| 约束解码 | 控制输出 | 强制特定 token 或 n-gram 出现在输出中 / 不出现在输出中。 |
| 幻觉 | 编造内容 | 无法被源文支持的模型输出。 |

## 延伸阅读

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — NLLB 论文。
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — 为什么 `sacrebleu` 是唯一正确的 BLEU 报告方式。
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — chrF 论文。
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — 实用的微调指南。
