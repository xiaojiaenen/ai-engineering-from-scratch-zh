# 长上下文评估 — NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 宣称支持 1000 万 token 上下文。在 100 万 token 时，8-needle MRCR 降至 26.3%。宣传 ≠ 可用。长上下文评估告诉你所交付模型的实际容量。

**类型：** 学习
**语言：** Python
**前置知识：** 第 5 阶段 · 13（问答）、第 5 阶段 · 23（分块策略）
**时间：** 约 60 分钟

## 问题所在

你有一份 200 页的合同。模型声称支持 100 万 token 上下文。你把合同粘贴进去并提问："终止条款是什么？"模型回答了——但它回答的是封面页的内容，因为终止条款位于距开头 12 万 token 处，超出了模型实际能注意到的范围。

这就是 2026 年的上下文容量差距。规格表声称 100 万或 1000 万。现实是其中 60-70% 才是可用的，而"可用"取决于具体任务。

- **检索（ haystack 中的单根针）：** 前沿模型在宣传的最大长度内接近完美。
- **多跳 / 聚合：** 大多数模型在约 128k token 后性能急剧下降。
- **对分散事实的推理：** 这是最先失败的任务。

长上下文评估测量这些维度。本教程命名了各基准测试、每个基准实际测量的内容，以及如何为你的领域构建定制的 needle 测试。

## 概念

![NIAH 基线、RULER 多任务、LongBench 综合评估](../assets/long-context-eval.svg)

**Haystack 中找针（Needle-in-a-Haystack，NIAH，2023）。** 在长上下文的受控深度位置放置一个事实（"魔法词是 pineapple"），然后询问模型将其检索出来。 Sweep depth × length。这是原始的长上下文基准。前沿模型现已在此基准上饱和；它是必要但不充分的基线。

**RULER（Nvidia，2024）。** 跨 4 个类别的 13 种任务类型：检索（单键 / 多键 / 多值）、多跳追踪（变量追踪）、聚合（常见词频）、QA。上下文长度可配置（4k 到 128k+）。揭示了那些在 NIAH 上饱和但在多跳任务上失败的模型。在 2024 年发布中，17 个声称支持 32k+ 上下文的模型中，只有一半在 32k 长度下保持了质量。

**LongBench v2（2024）。** 503 道选择题，8k-2M 词的上下文，六个任务类别：单文档 QA、多文档 QA、长上下文学习、长对话、代码仓库、长结构化数据。真实世界长上下文行为的生产基准。

**MRCR（多轮指代消解）。** 大规模多轮指代。8-needle、24-needle、100-needle 变体。暴露模型在注意力退化前能处理多少事实。

**NoLiMa。** "非词汇 needle。" needle 和查询之间没有字面重叠；检索需要一步语义推理。比 NIAH 更难。

**HELMET。** 拼接多个文档，从其中任一篇提出问题。测试选择性注意力。

**BABILong。** 在无关的 haystack 中嵌入 bAbI 推理链。测试的是 haystack 中的推理能力，而不仅仅是检索。

### 实际应报告的内容

- **宣传的上下文窗口。** 规格表上的数字。
- **有效检索长度。** 某阈值（如 90%）下的 NIAH 通过率。
- **有效推理长度。** 该阈值下的多跳或聚合通过率。
- **退化曲线。** 每类任务的准确率 vs 上下文长度曲线。

你的规格表需要两个数字：检索有效长度和推理有效长度。通常推理有效长度是宣传窗口的 25-50%。

```figure
gx-niah-decay
```

## 构建它

### 步骤 1：为你领域定制 NIAH

参见 `code/main.py`。骨架代码：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio 必须在 [0, 1] 范围内，实际得到 {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens 必须为正数，实际得到 {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text 未能生成任何 token")

    # 重复 filler 直到足够填充 haystack 主体。
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

遍历 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}。绘制热力图。这就是你目标模型的 NIAH 卡片。

### 步骤 2：多 needle 变体

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

问题如"三个魔法词是什么？"需要同时检索全部三个。单 needle 成功不能预测多 needle 成功。

### 步骤 3：多跳变量追踪（RULER 风格）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "X3 是多少？"
```

答案需要链式推导三个赋值。前沿模型在 128k 长度时准确率常降至 50-70%。

### 步骤 4：在你的栈上运行 LongBench v2

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

按类别报告准确率。聚合分数会掩盖重大的任务级差异。

## 陷阱

- **仅做 NIAH 评估。** 在 100 万 token 上通过 NIAH 无法说明多跳能力。始终运行 RULER 或自定义多跳测试。
- **均匀深度采样。** 许多实现仅测试 depth=0.5。应测试 depth=0, 0.25, 0.5, 0.75, 1.0——"迷失在中间"效应是真实存在的。
- **与 filler 的词法重叠。** 如果 needle 与 filler 共享关键词，检索会变得过于简单。使用 NoLiMa 风格的不重叠 needle。
- **忽略延迟。** 100 万 token 的 prompt 需要 30-120 秒进行预填充。应同时测量首次 token 时间（TTFT）和准确率。
- **厂商自报数据。** OpenAI、Google、Anthropic 都发布了自己的分数。始终在你的用例上独立重跑。

## 使用它

2026 年的推荐方案：

| 场景 | 基准测试 |
|-----------|-----------|
| 快速 sanity check | 3 个深度 × 3 个长度的定制 NIAH |
| 生产用模型选型 | 目标长度下的 RULER（13 项任务） |
| 真实世界 QA 质量 | LongBench v2 单文档 QA 子集 |
| 多跳推理 | BABILong 或自定义变量追踪 |
| 对话 / 多轮 | 目标长度下的 MRCR 8-needle |
| 模型升级回归测试 | 固定的内部 NIAH + RULER 测试套件，每次新模型均运行 |

生产环境的经验法则：在没有在你预期长度上完成 NIAH + 至少 1 个推理任务之前，不要信任任何上下文窗口。

## 交付物

保存为 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: 为给定模型和使用场景设计长上下文评估套件。
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

给定目标模型、目标上下文长度和使用场景，输出：

1. 测试。NIAH 深度 × 长度网格；RULER 多跳；自定义领域任务。
2. 采样。每个长度下测试深度 0, 0.25, 0.5, 0.75, 1.0。
3. 指标。检索通过率；推理通过率；首次 token 时间；单次查询成本。
4. 截断点。有效检索长度（90% 通过）和有效推理长度（70% 通过）。两者均需报告。
5. 回归测试。固定测试套件，每次模型升级时重新运行，记录差异。

拒绝仅凭模型卡就信任上下文窗口。拒绝针对多跳工作负载仅做 NIAH 评估。拒绝将厂商自报的长上下文分数作为独立证据。
```

## 练习

1. **简单。** 构建一个 NIAH，包含 3 个深度（0.25, 0.5, 0.75）× 3 个长度（1k, 4k, 16k）。在任何模型上运行。将通过率绘制为 3×3 热力图。
2. **中等。** 添加一个 3-needle 变体。测量每个长度下对全部 3 个 needle 的检索成功率。与同长度下单 needle 的通过率进行比较。
3. **困难。** 构建一个变量追踪任务（X1 → X2 → X3，包含 3 步跳），嵌入 64k 的 filler 中。在 3 个前沿模型上测量准确率。报告每个模型的有效推理长度。

## 关键术语

| 术语 | 人们常说的含义 | 实际含义 |
|------|-----------------|-----------------------|
| NIAH | haystack 中找针 | 在 filler 中植入一个事实，让模型将其检索出来。 |
| RULER | 加强版 NIAH | 13 种任务类型，涵盖检索 / 多跳 / 聚合 / QA。 |
| 有效上下文 | 真实容量 | 准确率仍能保持在阈值之上的长度。 |
| 迷失在中间 | 深度偏差 | 模型对长输入中间部分的内容关注不足。 |
| 多 needle | 同时处理多个事实 | 多个植入点；测试的是注意力多任务处理能力，而非单纯检索。 |
| MRCR | 多轮指代消解 | 8、24 或 100-needle 指代；暴露注意力饱和程度。 |
| NoLiMa | 非词汇 needle | needle 和查询无字面 token 重叠；需要推理能力。 |

## 延伸阅读

- [Kamradt (2023). Needle in a Haystack 分析](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 原始 NIAH 仓库。
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — 多任务基准。
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 真实世界长上下文评估。
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — 更难度的 needle。
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — haystack 中的推理能力。
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — 深度偏差论文。
