# 评估：基准测试、评估、LM Harness

> 古德哈特定律：当一项度量成为目标，它就不再是一项好的度量。每个前沿实验室都在刷基准测试。MMLU 分数不断上升，但模型仍然无法可靠地数出 "strawberry" 中有几个 R。唯一重要的评估是**你的**评估——针对**你的**任务、使用**你的**数据。

**类型：** 构建
**语言：** Python
**前置知识：** 第 10 阶段，课程 01-05（从零开始构建 LLM）
**时间：** 约 90 分钟

## 学习目标

- 构建一个自定义评估框架，对语言模型运行多项选择和开放式基准测试
- 解释为什么标准基准测试（MMLU、HumanEval）会饱和并无法区分前沿模型
- 使用适当的指标实现任务特定评估：精确匹配、F1、BLEU 和 LLM-as-judge 评分
- 设计针对你特定用例的自定义评估套件，而不是完全依赖公共排行榜

## 问题所在

MMLU 于 2020 年发布，包含跨 57 个学科的 15,908 道题目。三年之内，前沿模型就饱和了它。GPT-4 得分 86.4%，Claude 3 Opus 得分 86.8%，Llama 3 405B 得分 88.6%。排行榜压缩到了一个 3 分的范围内，其中的差异只是统计噪声，而非真正的能力差距。

与此同时，那些同样的模型在 10 岁孩子不假思索就能完成的任务上失败了。Claude 3.5 Sonnet 在 MMLU 上得分 88.7%，最初无法数出 "strawberry" 中的字母数量——这是一个零世界知识、零推理要求的任务，只需要字符级迭代。HumanEval 用 164 个问题测试代码生成。模型在上面得分 90%+，但产出的代码在处理任何初级开发者都能发现的边界情况时会崩溃。

基准测试表现与现实可靠性之间的差距是 LLM 评估的核心问题。基准测试告诉你模型在基准测试上表现如何。它们几乎不能告诉你模型在你的特定任务、你的特定数据、你的特定失败模式下表现如何。如果你正在构建客户支持机器人，MMLU 毫无意义。如果你正在构建代码助手，HumanEval 只覆盖函数级生成——它对跨文件调试、重构或解释代码毫无说明力。

你需要自定义评估。不是因为基准测试无用——它们对粗略模型选择有用——而是因为最终评估必须与你的部署条件完全匹配。

## 概念

### 评估全景

评估分为三类，每类的成本和信号质量各不相同。

**基准测试**是标准化测试套件。MMLU、HumanEval、SWE-bench、MATH、ARC、HellaSwag。你将模型与基准测试进行比较并获得分数。优势：每个人都使用相同的测试，所以你可以比较模型。劣势：模型和训练数据越来越污染这些基准测试。实验室在包含基准测试问题的数据上进行训练。分数上涨。能力可能没有。

**自定义评估**是你为自己特定用例构建的测试套件。你定义输入、期望输出和评分函数。法律文档摘要器在 legal documents 上评估。SQL 生成器在你的数据库模式上评估。创建成本高，但它们是预测生产表现的唯一评估方式。

**人工评估**使用付费标注员根据有用性、正确性、流畅性和安全性等标准评判模型输出。对于自动化评分失败的开放式任务是黄金标准。Chatbot Arena 收集了超过 200 万次人工偏好投票，覆盖 100+ 模型。劣势：成本（每次评判 $0.10-$2.00）和速度（几小时到几天）。

```mermaid
graph TD
    subgraph Eval["评估全景"]
        direction LR
        B["基准测试\n(MMLU, HumanEval)\n便宜、标准化\n可被刷分、过时"]
        C["自定义评估\n你的任务、你的数据\n最高信号质量\n构建成本高"]
        H["人工评估\n(Chatbot Arena)\n黄金标准\n慢、昂贵"]
    end

    B -->|"粗略模型选择"| C
    C -->|"模糊案例"| H

    style B fill:#1a1a2e,stroke:#ffa500,color:#fff
    style C fill:#1a1a2e,stroke:#51cf66,color:#fff
    style H fill:#1a1a2e,stroke:#e94560,color:#fff
```

### 为什么基准测试会失效

三种机制导致基准测试分数停止反映真实能力。

**数据污染。** 训练语料抓取互联网。基准测试问题存在于互联网上。模型在训练期间看到了答案。这不是传统意义上的作弊——实验室不会故意包含基准测试数据。但网络规模抓取使其几乎不可能排除。

**应试教学。** 实验室优化训练混合以追求基准测试表现。如果训练混合中有 5% 是 MMLU 风格的多选题，模型会学习格式和答案分布。MMLU 是四选项选择题。模型学会了答案分布大致在 A/B/C/D 之间均匀分布，这有助于即使模型不知道答案时也能得分。

**饱和。** 当每个前沿模型都在基准测试上得分 85-90% 时，基准测试停止区分。剩余的 10-15% 问题可能是模糊的、标注错误的，或需要生僻领域知识。MMLU 从 87% 提升到 89% 可能意味着模型多背诵了两个生僻问题，而不是变得更聪明。

### 困惑度：快速健康检查

困惑度衡量模型对词元序列的惊讶程度。形式化地说，它是指数化的平均负对数似然：

```
PPL = exp(-1/N * sum(log P(token_i | context)))
```

困惑度为 10 意味着模型平均而言，在每个词元位置的不确定性等同于在 10 个选项中均匀选择。越低越好。GPT-2 在 WikiText-103 上的困惑度约为 30。GPT-3 约为 20。Llama 3 8B 约为 7。

困惑度用于比较同一测试集上的模型很有用，但它有盲点。一个模型可能因为擅长预测常见模式而具有低困惑度，同时对罕见但重要的模式却很糟糕。它也无法说明指令遵循、推理或事实准确性。将其用作 sanity check，而非最终结论。

### LLM-as-Judge

使用强模型评估弱模型的输出。想法很简单：让 GPT-4o 或 Claude Sonnet 按照 1-5 分制对回答的正确性、有用性和安全性进行评分。使用 GPT-4o-mini 每次评判成本约 $0.01，与人工评判的相关性相当不错——在大多数任务上约 80% 的一致性。

评分提示比模型更重要。模糊的提示（"对这个回答评分"）会产生噪声分数。带有评分标准的结构化提示（"如果答案事实正确且引用来源得 5 分，正确但无来源得 4 分，部分正确得 3 分……"）会产生一致、可复现的分数。

失效模式：评判模型存在位置偏见（在成对比较中偏向第一个回答）、长度偏见（偏向更长回答）和自偏好（GPT-4 对 GPT-4 输出的评分高于等效的 Claude 输出）。缓解措施：随机化顺序、按长度归一化、使用与被评估模型不同的评判模型。

### 来自成对比较的 ELO 评级

Chatbot Arena 的方法。向同一个人展示来自不同模型的两个回答。人工（或 LLM 评判）选择更好的那个。从数千次此类比较中，计算每个模型的 ELO 评级——与国际象棋使用的系统相同。

ELO 优势：相对排名比绝对评分更可靠，能优雅处理平局，且收敛所需的比较次数少于独立评分每个输出。截至 2026 年初，Chatbot Arena 排名显示 GPT-4o、Claude 3.5 Sonnet 和 Gemini 1.5 Pro 在前 20 个 ELO 点内。

```mermaid
graph LR
    subgraph ELO["ELO 评级流水线"]
        direction TB
        P["提示"] --> MA["模型 A 输出"]
        P --> MB["模型 B 输出"]
        MA --> J["评判\n(人工或 LLM)"]
        MB --> J
        J --> W["A 胜 / B 胜 / 平局"]
        W --> E["ELO 更新\nK=32"]
    end

    style P fill:#1a1a2e,stroke:#0f3460,color:#fff
    style J fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#51cf66,color:#fff
```

### 评估框架

**lm-evaluation-harness**（EleutherAI）：标准的开源评估框架。支持 200+ 基准测试。一条命令即可将任何 Hugging Face 模型与 MMLU、HellaSwag、ARC 等运行比较。被 Open LLM Leaderboard 使用。

**RAGAS**：专为 RAG 流水线设计的评估框架。测量忠实度（回答是否与检索到的上下文匹配？）、相关性（检索到的上下文是否与问题相关？）和答案正确性。

**promptfoo**：配置驱动的提示工程评估。在 YAML 中定义测试用例，针对多个模型运行，获取通过/失败报告。适用于回归测试提示——确保提示更改不会破坏现有测试用例。

### 构建自定义评估

生产环境中唯一重要的评估。流程如下：

1. **定义任务。** 模型应该做什么？要精确。"回答问题"过于模糊。"给定客户投诉邮件，提取产品名称、问题类别和情感"是一个可以评估的任务。

2. **创建测试用例。** 原型评估至少 50 个，生产环境 200+ 个。每个测试用例是一个 (input, expected_output) 对。包括边界情况：空输入、对抗性输入、模糊输入、其他语言的输入。

3. **定义评分。** 结构化输出用精确匹配。文本相似度用 BLEU/ROUGE。开放式质量用 LLM-as-judge。提取任务用 F1。将多个指标加权组合。

4. **自动化。** 每次评估通过一条命令运行。无需手动步骤。将结果存储在便于随时间比较的格式中。

5. **随时间追踪。** 孤立来看评估分数毫无意义。你需要趋势线。上次提示更改后分数是否提高？切换模型后是否倒退？将你的评估与提示一起版本化。

| 评估类型 | 每次评判成本 | 与人工的一致性 | 最适合 |
|-----------|------------------|----------------------|----------|
| 精确匹配 | ~$0 | 100%（适用时） | 结构化输出、分类 |
| BLEU/ROUGE | ~$0 | ~60% | 翻译、摘要 |
| LLM-as-judge | ~$0.01 | ~80% | 开放式生成 |
| 人工评估 | $0.10-$2.00 | N/A（是地面真值） | 模糊、高风险任务 |

```figure
perplexity-loss
```

## 构建

### 步骤 1：最小评估框架

定义核心抽象。一个评估用例包含输入、期望输出和可选的元数据字典。评分器接收预测和参考，返回 0 到 1 之间的分数。

```python
import json
from collections import Counter

class EvalCase:
    def __init__(self, input_text, expected, metadata=None):
        self.input_text = input_text
        self.expected = expected
        self.metadata = metadata or {}

class EvalSuite:
    def __init__(self, name, cases, scorers):
        self.name = name
        self.cases = cases
        self.scorers = scorers

    def run(self, model_fn):
        results = []
        for case in self.cases:
            prediction = model_fn(case.input_text)
            scores = {}
            for scorer_name, scorer_fn in self.scorers.items():
                scores[scorer_name] = scorer_fn(prediction, case.expected)
            results.append({
                "input": case.input_text,
                "expected": case.expected,
                "prediction": prediction,
                "scores": scores,
            })
        return results
```

### 步骤 2：评分函数

构建精确匹配、token F1 和模拟的 LLM-as-judge 评分器。

```python
def exact_match(prediction, expected):
    return 1.0 if prediction.strip().lower() == expected.strip().lower() else 0.0

def token_f1(prediction, expected):
    pred_tokens = set(prediction.lower().split())
    exp_tokens = set(expected.lower().split())
    if not pred_tokens or not exp_tokens:
        return 0.0
    common = pred_tokens & exp_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(exp_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def llm_judge_simulated(prediction, expected):
    pred_words = set(prediction.lower().split())
    exp_words = set(expected.lower().split())
    if not exp_words:
        return 0.0
    overlap = len(pred_words & exp_words) / len(exp_words)
    length_penalty = min(1.0, len(prediction) / max(len(expected), 1))
    return round(overlap * 0.7 + length_penalty * 0.3, 3)
```

### 步骤 3：ELO 评级系统

实现带有 ELO 更新的成对比较。这正是 Chatbot Arena 用于排名模型的系统。

```python
class ELOTracker:
    def __init__(self, k=32, initial_rating=1500):
        self.ratings = {}
        self.k = k
        self.initial_rating = initial_rating
        self.history = []

    def _ensure_player(self, name):
        if name not in self.ratings:
            self.ratings[name] = self.initial_rating

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def record_match(self, player_a, player_b, outcome):
        self._ensure_player(player_a)
        self._ensure_player(player_b)

        ea = self.expected_score(self.ratings[player_a], self.ratings[player_b])
        eb = 1 - ea

        if outcome == "a":
            sa, sb = 1.0, 0.0
        elif outcome == "b":
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        self.ratings[player_a] += self.k * (sa - ea)
        self.ratings[player_b] += self.k * (sb - eb)

        self.history.append({
            "a": player_a, "b": player_b,
            "outcome": outcome,
            "rating_a": round(self.ratings[player_a], 1),
            "rating_b": round(self.ratings[player_b], 1),
        })

    def leaderboard(self):
        return sorted(self.ratings.items(), key=lambda x: -x[1])
```

### 步骤 4：困惑度计算

使用 token 概率计算困惑度。在实际应用中，你将从模型的 logits 获取这些值。这里我们用概率分布模拟。

```python
import numpy as np

def perplexity(log_probs):
    if not log_probs:
        return float("inf")
    avg_neg_log_prob = -np.mean(log_probs)
    return float(np.exp(avg_neg_log_prob))

def token_log_probs_simulated(text, model_quality=0.8):
    np.random.seed(hash(text) % 2**31)
    tokens = text.split()
    log_probs = []
    for i, token in enumerate(tokens):
        base_prob = model_quality
        if len(token) > 8:
            base_prob *= 0.6
        if i == 0:
            base_prob *= 0.7
        prob = np.clip(base_prob + np.random.normal(0, 0.1), 0.01, 0.99)
        log_probs.append(float(np.log(prob)))
    return log_probs
```

### 步骤 5：汇总结果

计算评估运行的摘要统计量：均值、中位数、阈值通过率以及各指标分解。

```python
def summarize_results(results, threshold=0.8):
    all_scores = {}
    for r in results:
        for metric, score in r["scores"].items():
            all_scores.setdefault(metric, []).append(score)

    summary = {}
    for metric, scores in all_scores.items():
        arr = np.array(scores)
        summary[metric] = {
            "mean": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "std": round(float(np.std(arr)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "pass_rate": round(float(np.mean(arr >= threshold)), 3),
            "n": len(scores),
        }
    return summary

def print_summary(summary, suite_name="Eval"):
    print(f"\n{'=' * 60}")
    print(f"  {suite_name} 摘要")
    print(f"{'=' * 60}")
    for metric, stats in summary.items():
        print(f"\n  {metric}:")
        print(f"    均值:      {stats['mean']:.3f}")
        print(f"    中位数:    {stats['median']:.3f}")
        print(f"    标准差:    {stats['std']:.3f}")
        print(f"    范围:      [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"    通过率:    {stats['pass_rate']:.1%} (阈值 >= 0.8)")
        print(f"    数量:      {stats['n']}")
```

### 步骤 6：运行完整流水线

将所有组件连接起来。定义任务，创建测试用例，模拟两个模型，运评测，从成对比较计算 ELO，并打印排行榜。

```python
def demo_model_good(prompt):
    responses = {
        "What is the capital of France?": "Paris",
        "What is 2 + 2?": "4",
        "Who wrote Hamlet?": "William Shakespeare",
        "What language is PyTorch written in?": "Python and C++",
        "What is the boiling point of water?": "100 degrees Celsius",
    }
    return responses.get(prompt, "I don't know")

def demo_model_bad(prompt):
    responses = {
        "What is the capital of France?": "Paris is the capital city of France",
        "What is 2 + 2?": "The answer is four",
        "Who wrote Hamlet?": "Shakespeare",
        "What language is PyTorch written in?": "Python",
        "What is the boiling point of water?": "212 Fahrenheit",
    }
    return responses.get(prompt, "Unknown")

cases = [
    EvalCase("What is the capital of France?", "Paris"),
    EvalCase("What is 2 + 2?", "4"),
    EvalCase("Who wrote Hamlet?", "William Shakespeare"),
    EvalCase("What language is PyTorch written in?", "Python and C++"),
    EvalCase("What is the boiling point of water?", "100 degrees Celsius"),
]

suite = EvalSuite(
    name="General Knowledge",
    cases=cases,
    scorers={
        "exact_match": exact_match,
        "token_f1": token_f1,
        "llm_judge": llm_judge_simulated,
    },
)

results_good = suite.run(demo_model_good)
results_bad = suite.run(demo_model_bad)

print_summary(summarize_results(results_good), "Model A (concise)")
print_summary(summarize_results(results_bad), "Model B (verbose)")
```

"好"模型给出精确答案。"差"模型给出冗长释义。精确匹配对冗长模型严惩。Token F1 和 LLM-as-judge 更宽容。这说明了指标选择的重要性：同一个模型根据你如何评分，看起来可能很棒或很差。

### 步骤 7：ELO 锦标赛

在多个轮次中运行模型之间的成对比较。

```python
elo = ELOTracker(k=32)

for case in cases:
    pred_a = demo_model_good(case.input_text)
    pred_b = demo_model_bad(case.input_text)

    score_a = token_f1(pred_a, case.expected)
    score_b = token_f1(pred_b, case.expected)

    if score_a > score_b:
        outcome = "a"
    elif score_b > score_a:
        outcome = "b"
    else:
        outcome = "tie"

    elo.record_match("model_a_concise", "model_b_verbose", outcome)

print("\nELO 排行榜:")
for name, rating in elo.leaderboard():
    print(f"  {name}: {rating:.0f}")
```

### 步骤 8：困惑度比较

比较不同质量级别"模型"的困惑度。

```python
test_text = "The quick brown fox jumps over the lazy dog in the garden"

for quality, label in [(0.9, "Strong model"), (0.7, "Medium model"), (0.4, "Weak model")]:
    log_probs = token_log_probs_simulated(test_text, model_quality=quality)
    ppl = perplexity(log_probs)
    print(f"  {label} (quality={quality}): perplexity = {ppl:.2f}")
```

## 使用

### lm-evaluation-harness（EleutherAI）

在任意模型上运行基准测试的标准工具。

```python
# pip install lm-eval
# 命令行:
# lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B --tasks mmlu --batch_size 8

# Python API:
# import lm_eval
# results = lm_eval.simple_evaluate(
#     model="hf",
#     model_args="pretrained=meta-llama/Llama-3.1-8B",
#     tasks=["mmlu", "hellaswag", "arc_easy"],
#     batch_size=8,
# )
# print(results["results"])
```

### promptfoo

配置驱动的提示工程评估。在 YAML 中定义测试，针对多个提供商运行。

```yaml
# promptfoo.yaml
providers:
  - openai:gpt-4o-mini
  - anthropic:claude-3-haiku

prompts:
  - "用一词回答：{{question}}"

tests:
  - vars:
      question: "What is the capital of France?"
    assert:
      - type: contains
        value: "Paris"
  - vars:
      question: "What is 2 + 2?"
    assert:
      - type: equals
        value: "4"
```

### RAGAS 用于 RAG 评估

```python
# pip install ragas
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
#
# result = evaluate(
#     dataset,
#     metrics=[faithfulness, answer_relevancy, context_precision],
# )
# print(result)
```

RAGAS 测量通用评估所遗漏的内容：模型的回答是否基于检索到的上下文，而不仅仅是答案在抽象意义上是否"正确"。

## 交付

本课程产出 `outputs/prompt-eval-designer.md`——一个可复用的提示，为任何任务设计自定义评估套件。给它一个任务描述，它会生成测试用例、评分函数和通过/失败阈值建议。

它还产出 `outputs/skill-llm-evaluation.md`——一个决策框架，根据你的任务类型、预算和延迟要求选择正确的评估策略。

## 练习

1. 添加一个"一致性"评分器，将相同输入通过模型运行 5 次并测量输出匹配的频率。确定性输入上不一致的回答揭示了脆弱的提示或过高的温度设置。

2. 扩展 ELO 跟踪器以支持多种评判函数（精确匹配、F1、LLM-as-judge）并对它们加权。比较当你重加权精确匹配与重加权 F1 时排行榜如何变化。

3. 为特定任务构建评估套件：将电子邮件分类为 5 个类别。创建 100 个包含多样示例的测试用例，包括边界情况（可能属于多个类别的邮件、空邮件、其他语言的邮件）。测量不同"模型"（基于规则、关键词匹配、模拟 LLM）的表现。

4. 实现污染检测：给定一组评估问题和训练语料库，检查评估问题（或接近的释义）在训练数据中出现的百分比。这是研究人员审计基准测试有效性的方法。

5. 构建"模型差异"工具。给定两个模型版本的评估结果，突出显示哪些具体测试用例改进、哪些倒退、哪些保持不变。这是评估等效的代码 diff——对于理解更改是有帮助还是有害至关重要。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| MMLU | "那个基准测试" | Massive Multitask Language Understanding——跨 57 个学科的 15,908 道多选题，到 2025 年已被饱和至 88% 以上 |
| HumanEval | "代码评估" | OpenAI 的 164 个 Python 函数补全问题，仅测试隔离的函数生成 |
| SWE-bench | "真实编码评估" | 来自 12 个 Python 仓库的 2,294 个 GitHub issue，测量端到端 bug 修复包括测试生成 |
| Perplexity | "模型有多困惑" | exp(-avg(log P(token_i given context)))——越低意味着模型为实际 token 分配更高概率 |
| ELO rating | "模型的象棋排名" | 从成对胜负记录计算的相对技能评级，被 Chatbot Arena 用于排名 100+ 模型 |
| LLM-as-judge | "用 AI 给 AI 打分" | 强模型根据评分标准评估弱模型输出，与人工评判约 80% 一致性，每次评判约 $0.01 |
| Data contamination | "模型看过考题" | 训练数据包含基准测试问题，在不提升真实能力的情况下推高分数 |
| Eval suite | "一堆测试" | 版本化的 (input, expected_output, scorer) 三元组集合，测量特定能力 |
| Pass rate | "正确率百分比" | 得分高于阈值的评估用例比例——比均值分数更具可操作性，因为它衡量可靠性 |
| Chatbot Arena | "模型排名网站" | LMSYS 平台，拥有 200 万+ 人工偏好投票，通过 ELO 评级产生最可信的 LLM 排行榜 |

## 延伸阅读

- [Hendrycks et al., 2021 -- "Measuring Massive Multitask Language Understanding"](https://arxiv.org/abs/2009.03300)——MMLU 论文，尽管已饱和仍是引用最多的 LLM 基准测试
- [Chen et al., 2021 -- "Evaluating Large Language Models Trained on Code"](https://arxiv.org/abs/2107.03374)——OpenAI 的 HumanEval 论文，确立了代码生成评估方法论
- [Zheng et al., 2023 -- "Judging LLM-as-a-Judge"](https://arxiv.org/abs/2306.05685)——使用 LLM 评估 LLM 的系统分析，包括位置偏见和长度偏见发现
- [LMSYS Chatbot Arena](https://chat.lmsys.org/)——众包模型比较平台，拥有 200 万+ 投票，是最可信的现实 LLM 排名
