# 长上下文评估 — NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 宣称拥有 1000 万 token 的上下文。但在 100 万 token 时，8 针 MRCR 准确率降至 26.3%。宣称的容量 ≠ 实际可用容量。长上下文评估能揭示你即将部署模型的真实容量。

**类型：** 学习
**语言：** Python
**先决条件：** 第 5 阶段 · 13（问答），第 5 阶段 · 23（分块策略）
**时间：** 约 60 分钟

## 问题描述

你有一份 200 页的合同。模型宣称支持 100 万 token 上下文。你将合同粘贴进去并提问：“终止条款是什么？” 模型给出了答案——但答案来自封面页，因为终止条款位于 12 万 token 深处，超出了模型实际关注的范围。

这就是 2026 年的上下文容量差距。规格书上写着 100 万或 1000 万。而现实是，其中仅 60-70% 是可用的，且“可用”程度取决于具体任务。

- **检索（单针）：** 在前沿模型上，直到宣称的最大值附近都近乎完美。
- **多跳/聚合：** 在大多数模型上，超过约 12.8 万 token 后性能急剧下降。
- **基于分散事实的推理：** 这是最先失败的任务。

长上下文评估衡量这些维度。本课程将介绍相关基准测试、它们各自的实际衡量内容，以及如何为你的领域构建定制化的“大海捞针”测试。

## 核心概念

![NIAH 基准，RULER 多任务，LongBench 综合评估](../assets/long-context-eval.svg)

**大海捞针（NIAH, 2023）。** 将一个事实（如“魔法词是菠萝”）放置在长文本的某个可控深度处，要求模型检索它。扫描深度×长度。这是最初的长上下文基准。前沿模型现已在此项上趋于饱和；这是一个必要但不充分的基准。

**RULER（英伟达，2024）。** 涵盖 4 大类共 13 种任务类型：检索（单键/多键/多值）、多跳追踪（变量追踪）、聚合（常见词频率）、问答。可配置上下文长度（4k 至 128k+）。能揭示那些在 NIAH 上饱和但多跳失败的模型。在 2024 年的测试中，声称支持 32k+ 上下文的 17 个模型里，仅一半能在 32k 处保持质量。

**LongBench v2（2024）。** 503 道选择题，上下文长度 8k-200 万词，涵盖六类任务：单文档问答、多文档问答、长上下文学习、长对话、代码仓库、长结构化数据。用于评估真实世界长上下文行为的实用基准。

**MRCR（多轮共指消解）。** 大规模多轮共指消解。有 8 针、24 针、100 针等变体。能暴露模型在注意力机制退化前能同时处理多少事实。

**NoLiMa。** “非词汇针”。针和查询之间没有任何字面上的重叠；检索需要一步语义推理。比 NIAH 更难。

**HELMET。** 拼接多个文档，询问其中任意一个文档中的问题。测试选择性注意力。

**BABILong。** 将 bAbI 推理链嵌入无关的“干草堆”中。测试的是“干草堆中的推理”，而不仅仅是检索。

### 实际应报告什么

- **宣称的上下文窗口。** 规格书上的数字。
- **有效检索长度。** 在某个阈值（例如 90%）下通过 NIAH 测试的长度。
- **有效推理长度。** 在该阈值下通过多跳或聚合测试的长度。
- **性能衰减曲线。** 准确率 vs 上下文长度，按任务类型绘制。

为你的规格书准备两个数字：有效检索长度和有效推理长度。通常，有效推理长度是宣称窗口的 25-50%。

## 动手构建

### 步骤 1：为你领域定制 NIAH

参见 `code/main.py`。基本骨架：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

扫描 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}。绘制热图。这就是针对你目标模型的 NIAH 评估卡。

### 步骤 2：多针变体

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

类似“三个魔法词是什么？”的问题需要检索到所有三个。单针成功并不能预测多针成功。

### 步骤 3：多跳变量追踪（RULER 风格）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

答案需要链接三次赋值。前沿模型在 128k 时常在此处准确率降至 50-70%。

### 步骤 4：在你的技术栈上运行 LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

按类别报告准确率。聚合分数会掩盖任务层面的巨大差异。

## 常见陷阱

- **仅评估 NIAH。** 在 100 万 token 处通过 NIAH 并不能说明任何关于多跳的能力。务必运行 RULER 或定制的多跳测试。
- **均匀深度采样。** 许多实现只测试深度=0.5。请测试深度=0, 0.25, 0.5, 0.75, 1.0——“中间位置丢失”效应是真实存在的。
- **与填充内容存在词汇重叠。** 如果“针”与填充内容有共享关键词，检索将变得平凡。使用 NoLiMa 风格的非重叠“针”。
- **忽略延迟。** 100 万 token 的提示需要 30-120 秒进行预填充。需同时测量首次生成 token 时间和准确率。
- **厂商自报数据。** OpenAI、谷歌、Anthropic 都发布自己的分数。务必针对你的用例独立重新测试。

## 实际使用

2026 年的技术栈建议：

| 场景 | 推荐基准 |
|-----------|-----------|
| 快速健康检查 | 定制 NIAH，3 个深度 × 3 个长度 |
| 为生产选择模型 | RULER（13 个任务），在目标长度下测试 |
| 真实世界问答质量 | LongBench v2 单文档问答子集 |
| 多跳推理 | BABILong 或定制变量追踪 |
| 对话/会话 | MRCR 8 针，在目标长度下测试 |
| 模型升级回归测试 | 固定的内部 NIAH + RULER 测试套件，在每个新模型上运行 |

生产环境的经验法则：在未用 NIAH + 1 个推理任务于你的目标长度测试过之前，永远不要信任宣称的上下文窗口。

## 交付

保存为 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## 练习

1. **简单。** 构建一个 NIAH，3 个深度（0.25, 0.5, 0.75）× 3 个长度（1k, 4k, 16k）。在任意模型上运行。将通过率绘制为 3×3 热图。
2. **中等。** 添加一个 3 针变体。测量每个长度下对所有 3 针的检索情况。与同长度单针通过率进行比较。
3. **困难。** 构造一个变量追踪任务（X1 → X2 → X3，3 跳），嵌入 64k 填充内容中。测量 3 个前沿模型的准确率。报告每个模型的有效推理长度。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| NIAH | 大海捞针 | 在填充内容中植入一个事实，要求模型检索它。 |
| RULER | 加强版 NIAH | 涵盖检索/多跳/聚合/问答等 4 大类共 13 种任务类型。 |
| 有效上下文 | 真实容量 | 准确率仍保持在阈值之上的长度。 |
| 中间位置丢失 | 深度偏置 | 模型对长输入中间部分的内容关注度不足。 |
| 多针 | 多个事实同时处理 | 植入多个点；测试注意力调度能力，而非单纯的检索。 |
| MRCR | 多轮共指消解 | 8、24 或 100 针共指；暴露注意力饱和。 |
| NoLiMa | 非词汇针 | 针与查询无字面重叠；需要推理。 |

## 延伸阅读

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 原始 NIAH 仓库。
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — 多任务基准。
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 真实世界长上下文评估。
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — 更难的针。
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — 干草堆中的推理。
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — 关于深度偏置的论文。