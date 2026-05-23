# LLM评估 — RAGAS、DeepEval、G-Eval

> 精确匹配和F1分数无法捕捉语义等效性。人工评审无法规模化。LLM作为评判器是生产级的解决方案——但需要足够的校准以信任其输出数字。

**类型：** 构建
**语言：** Python
**前提条件：** 阶段5 · 13（问答系统），阶段5 · 14（信息检索）
**时间：** ~75分钟

## 问题所在

你的RAG系统回答："June 29th, 2007."
标准参考答案是："June 29, 2007."
精确匹配得分为0。F1分数约为75%。而人类评分会是100%。

现在将这个情况乘以10,000个测试用例。再乘以检索器、分块策略、提示词或模型的每一次变更。你需要一个能够理解语义、能以低成本大规模运行、不会在回归测试中撒谎、并能正确揭示失败模式的评估器。

2026年有三个框架解决了这个问题。

- **RAGAS.** 检索增强生成评估。四个RAG指标（忠实度、答案相关性、上下文精确度、上下文召回率），具有NLI + LLM评判器后端。有研究支持，轻量级。
- **DeepEval.** 面向LLM的Pytest。包含G-Eval、任务完成度、幻觉、偏见等指标。原生支持CI/CD。
- **G-Eval.** 一种方法（以及DeepEval中的一个指标）：使用思维链、自定义标准、0-1评分的LLM作为评判器。

这三个框架都依赖于LLM作为评判器。本课程旨在建立对这种方法及其周围信任层的直觉。

## 核心概念

![四个评估维度，LLM作为评判器架构](../assets/llm-evaluation.svg)

**LLM作为评判器。** 用一个根据评分标准（Rubric）对输出进行评分的LLM来替代静态指标。给定 `(query, context, answer)`，提示一个评判LLM："在忠实度上打0-1分。"返回分数。

为何有效：LLM能以极低的成本近似人类的判断。使用GPT-4o-mini（每个评分用例约$0.003）进行1000个样本的回归评估，成本低于5美元。

为何会静默失败：

1.  **评判偏见。** 评判器偏爱更长的答案、来自其自身模型家族的答案、与提示词风格匹配的答案。
2.  **JSON解析失败。** 错误的JSON → NaN分数 → 被静默排除在聚合结果之外。RAGAS用户深知此痛点。应使用 try/except 进行门控并设置明确的失败模式。
3.  **跨模型版本漂移。** 升级评判模型会改变所有指标。应冻结评判模型及其版本。

**RAG四大指标。**

| 指标 | 问题 | 后端 |
|------|------|------|
| 忠实度 | 答案中的每个声明是否都来自检索到的上下文？ | 基于NLI的蕴含关系 |
| 答案相关性 | 答案是否切中问题？ | 从答案生成假设性问题；与实际问题比较 |
| 上下文精确度 | 在检索到的分块中，相关的占比是多少？ | LLM评判器 |
| 上下文召回率 | 检索是否返回了所有必需的信息？ | 使用标准答案通过LLM评判器评判 |

**G-Eval.** 定义一个自定义标准："答案是否引用了正确的来源？"该框架会自动将其扩展为思维链评估步骤，然后给出0-1分。适用于RAGAS未覆盖的特定领域质量维度。

**校准。** 在未获得其与人类标签的相关性之前，永远不要相信原始的评判分数。运行100个手标签例子。绘制评判分数 vs 人类分数的图表。计算斯皮尔曼等级相关系数（Spearman rho）。如果rho < 0.7，说明你的评分标准需要改进。

## 构建它

### 步骤1：使用NLI计算忠实度（RAGAS风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
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

将答案分解为原子声明。使用NLI逐一检查每个声明与检索到的上下文。忠实度 = 被支持的声明所占比例。

### 步骤2：答案相关性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果答案暗示的问题与被问及的问题不同，相关性就会下降。

### 步骤3：G-Eval自定义指标

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

评估步骤即为评分标准。明确的步骤比隐晦的“打0-1分”提示更稳定。

### 步骤4：CI门控

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

打包成一个pytest文件。在每个PR上运行。在发生回归时阻止合并。

### 步骤5：从头构建一个简单的评估器

参见 `code/main.py`。仅使用标准库近似实现忠实度（答案声明与上下文的重叠）和相关性（答案token与问题token的重叠）。非生产级。仅用于展示结构。

## 常见陷阱

- **缺乏校准。** 一个与人类标签相关性仅为0.3的评判器是噪声。在发布前必须进行校准运行。
- **自我评估。** 使用同一个LLM生成和评判答案，会使分数膨胀10-20%。为评判器使用不同的模型家族。
- **成对评判中的位置偏见。** 评判器偏爱先呈现的选项。务必随机化顺序并运行两个方向的评估。
- **原始聚合结果隐藏失败。** 平均分数0.85常常隐藏了5%的灾难性失败。务必检查低分位数。
- **黄金数据集腐化。** 随时间漂移的、未版本化的评估集会破坏纵向比较。每次变更都应标记数据集。
- **LLM成本。** 大规模时，评判调用是成本大头。使用满足校准阈值的最便宜模型。如GPT-4o-mini、Claude Haiku、Mistral-small。

## 应用它

2026年的技术栈：

| 用例 | 框架 |
|------|------|
| RAG质量监控 | RAGAS（4个指标） |
| CI/CD回归门控 | DeepEval + pytest |
| 自定义领域标准 | DeepEval中的G-Eval |
| 在线实时流量监控 | 采用无参考模式的RAGAS |
| 人工参与的抽查 | 带标注UI的LangSmith或Phoenix |
| 红队测试/安全评估 | Promptfoo + DeepEval |

典型技术栈：RAGAS用于监控，DeepEval用于CI，G-Eval用于新颖维度。三者都运行；它们之间的分歧是有益的。

## 交付它

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

1.  **简单。** 在10个已知存在幻觉的RAG示例上使用RAGAS。验证忠实度指标能捕捉到每一个。
2.  **中等。** 手动为50个QA答案的正确性标注0-1分。使用G-Eval评分。测量评判分数与人类分数之间的斯皮尔曼等级相关系数。
3.  **困难。** 使用DeepEval构建一个pytest CI门控。故意使检索器性能退化。验证门控失败。通过检查最低10%分数的阈值，添加低分位警报。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| LLM作为评判器 | 用LLM打分 | 提示一个评判模型，根据评分标准对输出进行0-1评分。 |
| RAGAS | RAG指标库 | 一个开源的评估框架，提供4个无参考的RAG指标。 |
| 忠实度 | 答案是否基于上下文？ | 答案中由检索到的上下文所蕴含的声明比例。 |
| 上下文精确度 | 检索到的分块相关吗？ | 实际起作用的top-K分块所占比例。 |
| 上下文召回率 | 检索是否找到了所有信息？ | 被检索到的分块所支持的黄金答案声明比例。 |
| G-Eval | 自定义LLM评判器 | 评分标准 + 思维链评估步骤 + 0-1评分。 |
| 校准 | 信任但需验证 | 评判分数与人类分数之间的斯皮尔曼等级相关系数。 |

## 扩展阅读

- [Es et al. (2023). RAGAS: Retrieval Augmented Generation Assessment](https://arxiv.org/abs/2309.15217) — RAGAS论文。
- [Liu et al. (2023). G-Eval: 使用GPT-4进行NLG评估并更好地与人类对齐](https://arxiv.org/abs/2303.16634) — G-Eval论文。
- [DeepEval文档](https://deepeval.com/docs/metrics-introduction) — 开源的生产级技术栈。
- [Zheng et al. (2023). 使用MT-Bench和Chatbot Arena评判LLM作为评判器](https://arxiv.org/abs/2306.05685) — 偏见、校准、局限性。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — 集成了RAGAS、DeepEval、Phoenix的统一框架。