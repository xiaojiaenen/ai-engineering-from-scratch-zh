# LLM 评估 — RAGAS、DeepEval、G-Eval

> 精确匹配和 F1 分数无法捕捉语义等价性。人工评审无法扩展。LLM-as-judge 是生产环境的答案 — 但需足够的校准才能信任这些数字。

**类型：** Build
**语言：** Python
**前置要求：** Phase 5 · 13（问答），Phase 5 · 14（信息检索）
**时间：** 约 75 分钟

## 问题所在

你的 RAG 系统回答："June 29th, 2007."
黄金参考答案是："June 29, 2007."
精确匹配得分为 0。F1 分数约为 75%。人工评分会是 100%。

现在将测试用例乘以 10,000。再乘以每一次对检索器、分块、提示词或模型的变更。你需要一个能理解语义、低成本规模化运行、不会在回归时撒谎、并能揭示正确失败模式的评估器。

2026 年有三个框架主导这个问题。

- **RAGAS。** Retrieval-Augmented Generation ASsessment（检索增强生成评估）。四种 RAG 指标（忠实度、答案相关性、上下文精确度、上下文召回率），带有 NLI + LLM-judge 后端。具有研究支持，轻量级。
- **DeepEval。** Pytest for LLMs（LLM 的 Pytest）。G-Eval、任务完成、幻觉、偏见指标。CI/CD 原生。
- **G-Eval。** 一种方法（也是 DeepEval 的一个指标）：带思维链的 LLM-as-judge、自定义标准、0-1 分数。

三者都依赖 LLM-as-judge。本课程建立对此方法及围绕它的信任层面的直觉。

## 概念

![四种评估维度，LLM-as-judge 架构](../assets/llm-evaluation.svg)

**LLM-as-judge。** 用给定评分标准的 LLM 替代静态指标。给定 `(query, context, answer)`，提示评审 LLM："按忠实度打分 0-1。"返回分数。

为何有效：LLM 以极低的成本近似人类判断。GPT-4o-mini 每次评分成本约 $0.003，支持 1000 样本的回归评估运行费用低于 $5。

为何会无声失败：

1. **评审者偏见。** 评审者偏好较长的答案、来自自己模型家族的答案、与提示风格匹配的答案。
2. **JSON 解析失败。** 糟糕的 JSON → NaN 分数 → 被静默排除在聚合之外。RAGAS 用户深知此痛。用 try/except + 明确的失败模式设置网关。
3. **随模型版本漂移。** 升级评审者会改变所有指标。冻结评审者模型 + 版本。

**RAG 四指标。**

| 指标 | 问题 | 后端 |
|--------|----------|---------|
| 忠实度 (Faithfulness) | 答案中的每个主张是否来自检索到的上下文？ | 基于 NLI 的蕴含 |
| 答案相关性 (Answer relevance) | 答案是否回答了问题？ | 从答案生成假设问题；与实际问题比较 |
| 上下文精确度 (Context precision) | 在检索到的块中，有多少比例是相关的？ | LLM-judge |
| 上下文召回率 (Context recall) | 检索是否返回了所有必要内容？ | 与黄金答案对比的 LLM-judge |

**G-Eval。** 定义自定义标准："答案是否引用了正确的来源？"框架自动展开为思维链评估步骤，然后打分 0-1。适用于 RAGAS 未覆盖的领域特定质量维度。

**校准。** 在与人工标签的相关性验证之前，不要信任原始评审分数。运行 100 个手工标注示例。绘制评审者 vs 人工散点图。计算 Spearman rho。如果 rho < 0.7，说明你的评审标准需要改进。

```figure
n5-judge-gauge
```

## 构建

### 步骤 1：使用 NLI 的忠实度（RAGAS 风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` 是任意可调用函数：提示字符串 -> 生成的字符串。
# 示例：llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""将此答案拆分为简单的事实主张（每行一个）：
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

将答案分解为原子主张。针对检索到的上下文对每个主张进行 NLI 检查。忠实度 = 被支持的比例。

### 步骤 2：答案相关性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder：任何实现 .encode(texts, normalize_embeddings=True) -> ndarray 的模型
# 例如，encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"写出 {n} 个此答案可能回答的问题：\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果答案暗示了与被问问题不同的问题，相关性就会下降。

### 步骤 3：G-Eval 自定义指标

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

评估步骤就是评分标准。显式步骤比隐式的"评分 0-1"提示更稳定。

### 步骤 4：CI 网关

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

保存为 pytest 文件。在每次 PR 时运行。在出现回归时阻止合并。

### 步骤 5：从零开始的玩具评估

参见 `code/main.py`。仅使用标准库对忠实度（答案主张与上下文的交集）和相关性（答案令牌与问题令牌的交集）的近似实现。不是生产级。展示了基本形态。

## 陷阱

- **无校准。** 与人工标签相关系数为 0.3 的评审者是噪声。发货前要求运行校准。
- **自评估。** 使用同一个 LLM 生成和评审会使分数虚高 10-20%。评审者使用不同的模型家族。
- **成对评审中的位置偏见。** 评审者偏好首先呈现的选项。始终随机化顺序并双向运行。
- **原始聚合掩盖失败。** 平均分 0.85 经常掩盖 5% 的灾难性失败。始终检查最低分位数。
- **黄金数据集腐化。** 未版本化的评估集随时间漂移会破坏纵向比较。用每次变更标记数据集。
- **LLM 成本。** 在大规模下，评审调用主导成本。使用满足校准阈值的最低成本模型。GPT-4o-mini、Claude Haiku、Mistral-small。

## 使用

2026 技术栈：

| 用例 | 框架 |
|---------|-----------|
| RAG 质量监控 | RAGAS（4 项指标） |
| CI/CD 回归网关 | DeepEval + pytest |
| 自定义领域标准 | DeepEval 中的 G-Eval |
| 在线实时流量监控 | 无参考模式的 RAGAS |
| 人工抽查 | 带标注 UI 的 LangSmith 或 Phoenix |
| 红队测试 / 安全评估 | Promptfoo + DeepEval |

典型栈：RAGAS 用于监控，DeepEval 用于 CI，G-Eval 用于新颖维度。运行全部三个；它们会产生有 useful 的分歧。

## 交付

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## 练习

1. **简单。** 对 10 个已知存在幻觉的 RAG 示例使用 RAGAS。验证忠实度指标是否捕获了每一个。
2. **中等。** 手工标注 50 个 QA 答案的正确性 0-1。用 G-Eval 评分。测量评审者与人工之间的 Spearman rho。
3. **困难。** 使用 DeepEval 构建 pytest CI 网关。故意使检索器回归。验证网关是否失败。通过对最低 10% 的阈值检查添加最低分位数告警。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| LLM-as-judge | 用 LLM 评分 | 给定评分标准，提示评审模型对输出打分 0-1。 |
| RAGAS | RAG 指标库 | 具有 4 个无参考 RAG 指标的开源评估框架。 |
| 忠实度 (Faithfulness) | 答案是否基于依据？ | 被检索上下文蕴含的答案主张比例。 |
| 上下文精确度 (Context precision) | 检索到的块是否相关？ | 前 K 个块中实际重要的比例。 |
| 上下文召回率 (Context recall) | 检索是否找到了所有内容？ | 被检索块支持的黄金答案主张比例。 |
| G-Eval | 自定义 LLM 评审 | 评分标准 + 思维链评估步骤 + 0-1 分数。 |
| 校准 | 信任但验证 | 评审分数与人工分数之间的 Spearman 相关系数。 |

## 延伸阅读

- [Es 等人（2023）。RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 论文。
- [Liu 等人（2023）。G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 论文。
- [DeepEval 文档](https://deepeval.com/docs/metrics-introduction) — 开源生产栈。
- [Zheng 等人（2023）。Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — 偏见、校准、局限性。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — 集成 RAGAS、DeepEval、Phoenix 的统一框架。
