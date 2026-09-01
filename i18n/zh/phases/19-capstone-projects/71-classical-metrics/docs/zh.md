```markdown
# 经典指标

> BLEU、ROUGE-L、F1、精确匹配、准确率。这五个指标仍占据大多数已发表的 LLM 评估数值。从零开始实现每一个指标，这样你就知道这些数值代表什么含义。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 19 Track B 基础、第 70 课
**耗时：** 约 90 分钟

## 学习目标

- 实现基于显式分词规则的 token 级精确匹配、F1 和准确率。
- 从零实现 BLEU-4：修改后的 n-gram 精度、n 从 1 到 4 的几何平均、短句惩罚（brevity penalty）。
- 使用最长公共子序列（LCS）实现 ROUGE-L，并采用 F-beta 组合精度与召回率。
- 根据第 70 课的 `metric_name` 字段进行分发，使运行器保持指标无关性。
- 通过来自worked示例的参考向量固定行为，而非依赖第三方库。

```figure
cd-bleu-overlap
```

## 为什么要重新实现

你会读到报告 BLEU 28.3 的论文，也会看到报告 BLEU 0.283 的论文。你会发现由于一个库截断为小写而另一个不截断，不同库的 ROUGE-L 分数相差十分。停止困惑的最快方法是亲自编写这些指标，然后明确指出分词器在哪里被决定、平滑在哪里被应用。从那以后，比较论文之间的数值就变成了阅读指标配置的问题，而不是争论使用哪个库。

stdlib 加 numpy 就足够了。BLEU 是计数和截断。ROUGE-L 是动态规划。F1 是 token 集合的交集。最困难的部分是选择分词器并对此做出承诺。

## 分词

分词器是 `re.findall(r"\w+", text.lower())`。小写、字母数字序列、去除标点。本节的每个指标都使用完全相同的分词器。运行器无权选择。如果你更换分词器，你就是在运行不同的基准测试。

```python
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
def tokenize(text):
    # 小写并按单词边界切分
    return TOKEN_RE.findall(text.lower())
```

这是一种刻意的简化。生产环境会关心 CJK（中日韩）、缩略语和代码标识符。本节的关键在于分词器是一项契约，而非一个旋钮。

## 精确匹配

```python
def exact_match(pred, targets):
    # 去除首尾空格后与任意目标进行比较
    return float(any(pred.strip() == t.strip() for t in targets))
```

对每个任务返回 1.0 或 0.0。数据集上的聚合结果取均值。这是算术题、多选题和短分类任务的主力指标。

## Token 级 F1

为预测和目标分别建立 token 多重集。精度是多重集交集除以预测的多重集大小。召回率是同一交集除以目标的多重集大小。F1 是调和平均数。实现处理了空预测和空目标的边界情况。

```mermaid
flowchart LR
    A[预测文本] -->|分词| P[预测 tokens]
    B[目标文本] -->|分词| T[目标 tokens]
    P --> X[多重集交集]
    T --> X
    X --> PR[精度 = 交集 / 预测]
    X --> RE[召回率 = 交集 / 目标]
    PR --> F[F1 = 2 * P * R / (P + R)]
    RE --> F
```

对于多目标任务，我们取目标列表中的最佳 F1。这与文献中广泛报道的 SQuAD 风格行为一致。

## BLEU-4

BLEU 是机器翻译的经典指标，它仍然出现在摘要工作中。我们使用的是带有标准短句惩罚的语料级 BLEU-4，并对修改后的 n-gram 计数加上 1 的平滑，以免单个缺失的 4-gram 将分数降为零。

对于每个候选-参考对，我们统计 n 等于 1、2、3、4 的修改后 n-gram 精度。修改后的精度通过将候选 n-gram 计数与任意参考中该 n-gram 的最大计数取最小值来截断，从而防止候选通过重复同一个短语来膨胀。四个精度值的几何平均数被短句惩罚包裹。

```mermaid
flowchart TD
    A[候选 tokens] --> B[统计 n-gram n=1..4]
    R[参考 tokens] --> C[每个 n-gram 的最大计数]
    B --> D[截断后的 n-gram 计数]
    C --> D
    D --> E[修改后精度 p_n]
    A --> F[候选长度 c]
    R --> G[参考长度 r]
    F --> BP[BP = 1 若 c>=r 否则 exp(1 - r/c)]
    G --> BP
    E --> M[p_n 的几何平均]
    M --> S[BLEU = BP * 几何平均]
    BP --> S
```

平滑规则是 Lin 和 Och 所称的方法 1：在取对数之前，将每个 n-gram 精度的分子和分母都加一。这避免了当参考中没有匹配的 4-gram 时出现 `log 0`，且在长候选上保持接近未平滑的值。

## ROUGE-L

ROUGE-L 比较候选与参考 token 序列的最长公共子序列。LCS 在不强制连续的情况下捕捉词序，这也是它成为摘要默认指标的原因。我们使用标准的动态规划表格计算 LCS 长度，然后通过 `lcs / 参考长度` 得出召回率，通过 `lcs / 候选长度` 得出精度，并使用 F-beta 进行组合，其中 beta 等于一以得到对称的 F1 形式。

```python
def lcs_length(a, b):
    n, m = len(a), len(b)
    # 使用 numpy 二维表记录 DP 状态
    dp = numpy.zeros((n + 1, m + 1), dtype=int)
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i+1, j+1] = dp[i, j] + 1
            else:
                dp[i+1, j+1] = max(dp[i+1, j], dp[i, j+1])
    return int(dp[n, m])
```

numpy 表格使实现更加清晰；纯 Python 列表也可以。选择使用 ROUGE-L 的任务需要为每个任务承担 O(n m) 的开销。对于典型的摘要长度，这保持在 1 毫秒以内。

## 准确率

对于多目标分类任务，准确率简化为对单个归一化目标的精确匹配。我们将其暴露为单独的函数，以便分发器可以根据 `metric_name` 进行分发，而无需在运行器内部进行字符串比较。

## 分发契约

单一入口点是 `score(metric_name, prediction, targets)`。它返回一个 `[0, 1]` 范围内的浮点数。运行器不根据指标名称进行分支。它将调用传递给相应的指标函数并记录结果。这就是第 75 课将胶水到第 70 课任务规格的表面。

```python
def score(metric_name, pred, targets):
    # 精确匹配：候选必须完全等于某个目标
    if metric_name == "exact_match":
        return exact_match(pred, targets)
    # F1：取所有目标中最高的 F1
    if metric_name == "f1":
        return max(f1_score(pred, t) for t in targets)
    # BLEU-4：取所有目标中最高的 BLEU
    if metric_name == "bleu_4":
        return max(bleu4(pred, t) for t in targets)
    # ROUGE-L：取所有目标中最高的 ROUGE-L
    if metric_name == "rouge_l":
        return max(rouge_l(pred, t) for t in targets)
    # 准确率：用于单目标分类
    if metric_name == "accuracy":
        return accuracy(pred, targets)
    raise ValueError(f"未知的 metric_name: {metric_name}")
```

`code_exec` 在第 72 课中处理，并在那里嵌入到分发器中。

## 本节不做的事情

它不调用模型。它不对生成内容进行超出第 70 课后处理规则的归一化。它不计算置信区间。它不做 BLEURT 或 BERTScore（这些需要模型，属于另一节）。关键是打好基础：五个指标、一个分词器、一张分发表。

## 如何阅读代码

`main.py` 定义了每个指标作为独立函数以及分发器。参考向量位于文件底部的 `_reference_examples` 块中。演示程序运行分发器并针对八个示例打印每个指标的分数。`code/tests/test_metrics.py` 中的测试固定参考向量并压力测试每个边界情况（空预测、空参考、无共享 token、精确匹配、重复短语截断）。

从头到尾阅读 `main.py`。函数按复杂度排序。exact_match 和 accuracy 各一行。F1 六行。BLEU 和 ROUGE-L 是主要部分，它们包含关于平滑规则和 LCS 递推关系的详细注释。

## 进一步探索

经典指标是必要的，但并非充分的。它们奖励表面重叠而忽略语义。解决方案是在你信任经典基座的基础上，叠加基于模型的指标（BLEURT、BERTScore、GEval）。那是后续的课程。现在：让这些五个指标工作，用测试固定它们，你就拥有一个可审计、快速且可复现的指标栈。
```
