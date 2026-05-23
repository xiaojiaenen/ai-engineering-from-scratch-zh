# 机器翻译

> 翻译是过去三十年持续资助自然语言处理研究、且至今仍在产生价值的任务。

**类型：** 构建
**语言：** Python
**先修条件：** 阶段5·10（注意力机制），阶段5·04（GloVe, FastText, 子词）
**时间：** 约75分钟

## 问题描述

模型读入一种语言的句子，输出另一种语言的句子。句子长度可变，词序可变。某些源语言单词会映射到多个目标语言单词，反之亦然。习语无法实现一对一映射。法语的"I miss you"是"tu me manques"——字面意思是"你对我而言是缺失的"。没有任何单词级别的对齐能处理这种情况。

机器翻译任务迫使自然语言处理领域发明了编码器-解码器、注意力机制、Transformer模型，最终催生了整个大语言模型范式。每一步进展都源于翻译质量可量化评估，且人类与机器之间的差距始终难以弥合。

本课程跳过历史回顾，直接讲授2026年的工作流程：预训练多语言编码器-解码器（NLLB-200或mBART）、子词分词、束搜索、BLEU与chrF评估，以及那些潜入生产环境的少数未捕获故障模式。

## 核心概念

![MT 流水线：分词 → 编码 → 带注意力的解码 → 反分词](../assets/mt-pipeline.svg)

现代机器翻译是在平行文本上训练的Transformer编码器-解码器。编码器使用源语言分词规则读取源文本。解码器通过交叉注意力（第10课）利用编码器输出，逐个生成子词。解码过程采用束搜索以避免贪心解码陷阱。输出经过反分词、还原大小写处理，并与参考译文进行评分。

三个操作性选择决定了实际机器翻译的质量。

- **分词器。** 在混合语言语料上训练的SentencePiece BPE。跨语言共享词汇表是实现NLLB零样本语言对的关键。
- **模型大小。** NLLB-200 蒸馏版（600M参数）可在笔记本运行。NLLB-200 3.3B是发布的生产环境默认版本。54.5B是研究上限。
- **解码策略。** 通用内容使用束宽4-5。长度惩罚以避免过短输出。需要术语一致性时使用约束解码。

## 构建实践

### 步骤1：预训练机器翻译调用

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

此处三点至关重要。`src_lang`告知分词器应应用何种文字系统与切分方式。`forced_bos_token_id`告知解码器应生成何种语言。两者均为NLLB特有技巧；mBART与M2M-100使用各自约定且不可互换。

### 步骤2：BLEU与chrF

BLEU衡量输出与参考译文的n-gram重合度。使用四种n-gram尺寸（1-4），取精度的几何平均值，并对过短输出施加长度惩罚。分值范围[0, 100]。常用但解读困难：30 BLEU为"可用"，40为"良好"，50为"卓越"；低于1 BLEU的差异属于噪声。

chrF衡量字符级F分数。对于形态丰富的语言（BLEU会漏计匹配）更敏感。常与BLEU同时报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

务必使用`sacrebleu`。它会标准化分词处理，确保不同论文间的分数可比。自行计算BLEU是产生误导性基准的根源。

### 三级评估体系（2026）

现代机器翻译评估采用三类互补指标。实际部署至少使用两种。

- **启发式指标**（BLEU, chrF）。快速、基于参考、可解释、对释义不敏感。用于历史比较与回归检测。
- **学习型指标**（COMET, BLEURT, BERTScore）。基于人类判断训练的神经模型；比较译文与源文、参考文的语义相似度。COMET自2023年以来与机器翻译研究关联最高，2026年成为质量敏感场景的生产环境默认指标。
- **大语言模型作为裁判**（无参考）。提示大模型评估译文的流畅度、充分性、语气、文化适应性。当评估标准设计良好时，GPT-4作为裁判与人类判断的一致性约达80%。用于无参考译文的开放式内容评估。

2026实用技术栈：`sacrebleu`用于BLEU和chrF，`unbabel-comet`用于COMET，经提示的大语言模型提供面向人类的最终评估信号。在信赖生产环境数据前，需用50-100条人工标注样本校准每项指标。

无参考指标（COMET-QE, BLEURT-QE, 大语言模型裁判）允许在没有参考译文的情况下评估翻译，这对缺乏参考译文的长尾语言对尤为重要。

### 步骤3：生产环境中的失效模式

上述工作流程80%的时间能流畅翻译，但剩下20%会静默失败。已知失效模式：

- **幻觉**。模型编造源文未提及的内容。常见于陌生领域词汇。症状：输出流畅但陈述源文未声明的事实。缓解措施：领域术语约束解码、监管内容人工审核、监控输出长度远超输入的情况。
- **目标语言偏离**。模型翻译成错误语言。NLLB在稀有语言对上意外易发此问题。缓解措施：验证`forced_bos_token_id`并始终使用语言ID模型校验输出。
- **术语漂移**。"Sign up"在文档1译为"s'inscrire"，文档2译为"créer un compte"。对UI文本和用户面向字符串，一致性比原始质量更重要。缓解措施：术语表约束解码或译后编辑词典。
- **正式度失配**。法语"tu" vs "vous"，日语敬语层级。模型选择训练数据中更常见的形式。对面向客户的内容通常不适用。缓解措施：若模型支持，使用带正式度标记的提示前缀，或用纯正式语料微调小模型。
- **短输入长度爆炸**。极短输入句常产出过长译文，因为当源token少于约5个时长度惩罚会急剧下降。缓解措施：设置与源长成比例的硬性最大长度限制。

### 步骤4：领域微调

预训练模型是通才。法律、医疗或游戏对话翻译通过领域平行数据微调可获得显著提升。方法并不复杂：

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

数千个高质量平行样本胜过数十万噪声网络抓取数据。训练数据质量是生产环境中最大的单一杠杆点。

## 应用实践

2026年机器翻译生产技术栈：

| 使用场景 | 推荐起点 |
|---------|---------------------------|
| 任意语言对，200种语言 | `facebook/nllb-200-distilled-600M`（笔记本）或 `nllb-200-3.3B`（生产环境） |
| 以英语为中心，高质量，50种语言 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短文本、低成本推理、英法/德/西 | Helsinki-NLP / Marian 模型 |
| 延迟敏感的浏览器端 | ONNX量化Marian模型（约50MB） |
| 追求最高质量，愿承担成本 | 使用翻译提示的GPT-4 / Claude / Gemini |

截至2026年，大语言模型在多个语言对上的表现已超越专用机器翻译模型，尤其在习语处理和长上下文方面。权衡在于每token成本和延迟。当上下文长度、风格一致性或通过提示进行领域适配比吞吐量更重要时，选择大语言模型。

## 部署实践

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

1. **简单。** 使用`nllb-200-distilled-600M`将一段5句英文翻译为法文再译回英文。衡量往返翻译与原文的接近程度。应能看到语义保留但用词漂移。
2. **中等。** 使用`fasttext lid.176`或`langdetect`对翻译输出实施语言ID检查。集成到机器翻译调用中，以便在返回前捕获目标语言偏离的生成。
3. **困难。** 在自选的5000对领域语料上微调`nllb-200-distilled-600M`。在微调前后于留出集上测量BLEU。报告哪些句子类型得到改善，哪些出现退化。

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| BLEU | 翻译评分 | 带长度惩罚的n-gram精确率。[0, 100]范围。 |
| chrF | 字符F分数 | 字符级F分数。对形态丰富语言更敏感。 |
| NMT | 神经机器翻译 | 在平行文本上训练的Transformer编码器-解码器。2017年后的默认范式。 |
| NLLB | 不遗落任何语言 | Meta的200种语言机器翻译模型家族。 |
| 约束解码 | 受控输出 | 强制特定token或n-gram在输出中出现/不出现。 |
| 幻觉 | 编造内容 | 模型输出未被源文支持的内容。 |

## 延伸阅读

- [Costa-jussà 等 (2022). 不遗落任何语言：以人为本的机器翻译扩展研究](https://arxiv.org/abs/2207.04672) — NLLB论文。
- [Post (2018). 呼吁BLEU分数报告的清晰化](https://aclanthology.org/W18-6319/) — 为何`sacrebleu`是报告BLEU的唯一正确方式。
- [Popović (2015). chrF：用于自动机器翻译评估的字符n-gram F分数](https://aclanthology.org/W15-3049/) — chrF论文。
- [Hugging Face机器翻译指南](https://huggingface.co/docs/transformers/tasks/translation) — 实用的微调教程。