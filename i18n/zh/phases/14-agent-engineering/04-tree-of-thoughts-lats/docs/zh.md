# 思维树与LATS：审慎搜索

> 单一链式思维轨迹没有回溯空间。ToT（Yao 等，2023）将推理转化为树结构，在每个节点进行自我评估。LATS（Zhou 等，2024）在蒙特卡洛树搜索框架下统一了 ToT、ReAct 与 Reflexion。Game of 24 任务上，准确率从 4%（CoT）提升至 74%（ToT）；LATS 在 HumanEval 上达到 92.7% pass@1。

**类型：** 构建
**语言：** Python（标准库）
**先决条件：** 第14阶段 · 01（智能体循环）、第14阶段 · 03（Reflexion）
**时间：** 约75分钟

## 学习目标

- 将推理建模为搜索：节点是"思维"，边是"展开"，价值是"前景评分"。
- 使用标准库实现基于 ToT 风格的 BFS 树搜索，并带有自我评估评分。
- 扩展为简易的 LATS MCTS 循环，包含选择/展开/模拟/回传四阶段。
- 判断何时搜索值得付出 token 倍率开销（Game of 24、代码生成），何时单条轨迹就足够（简单问答）。

## 问题所在

链式思维是一条线性路径。如果第一步出错，后续每一步都在错误前提上工作。在 Game of 24（用四个数字通过 + − × ÷ 组合得到 24）任务上，GPT-4 的 CoT 准确率仅有 4%。模型在早期就选错了子表达式且无法恢复。

推理需要的是：提出多个候选方案、评估它们、挑选有前景的，以及在死胡同出现时回溯。这就是搜索。思维树（Tree of Thoughts）和 LATS 是两种经典表述形式。

## 概念

### 思维树（Tree of Thoughts，Yao 等，NeurIPS 2023）

每个节点是一个连贯的中间步骤（"一个思维"）。每个节点可扩展出 K 个子思维。LLM 通过评分提示词对每个节点进行自我评估。搜索以 BFS、DFS 或 beam 方式遍历整棵树。

```
                     (根节点："从 4 6 4 1 中找 24")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- 评分：高
              /   \              |                  |
          ...    ...          ...                完成
```

自我评估是核心机制。论文展示了三种变体：`sure / likely / impossible` 分类、`1..10` 数值评分，以及对候选方案进行投票。这三种方式在 Game of 24 上都大幅优于 CoT（GPT-4 上从 4% 提升至 74%）。

### LATS（Zhou 等，ICML 2024）

LATS 在 MCTS 框架下统一了 ToT、ReAct 与 Reflexion。LLM 扮演三个角色：

- **策略（Policy）**：提出候选下一步动作（ReAct 风格）。
- **价值函数（Value function）**：对部分轨迹打分（ToT 风格自我评估）。
- **自我反思者（Self-reflector）**：在失败时撰写自然语言反思（Reflexion 风格），并用其重新初始化后续 rollout。

环境反馈（观察结果）融入价值函数，使搜索能基于真实工具结果而非仅凭模型主观意见。论文发布时的结果：HumanEval pass@1 达 92.7%（GPT-4，SOTA），WebShop 平均 75.9（GPT-3.5，接近基于梯度的微调水平）。

### MCTS（极简版）

每次迭代包含四个阶段：

1. **选择（Select）** —— 从根节点沿 UCT（树的上界置信区间）公式走向叶节点。
2. **展开（Expand）** —— 通过策略生成 K 个子节点。
3. **模拟（Simulate）** —— 从子节点出发用策略进行 rollout，在叶节点用价值函数（或环境奖励）打分。
4. **回传（Backpropagate）** —— 沿路径更新访问次数和价值估计。

UCT 公式：`Q(s, a) + c * sqrt(ln N(s) / N(s, a))`。第一项是利用，第二项是探索。`c` 需要根据任务调优。

### 成本现实

搜索会引发 token 爆炸。ToT 在 Game of 24 上使用的 token 量是 CoT 的 100–1000 倍。LATS 情况类似。这并非免费，搜索应仅用于以下场景：

- 单条轨迹明显不足的任务（Game of 24、复杂代码）。
- 正确性比墙钟时间更重要的任务。
- 拥有廉价且可靠价值函数的任务（代码单元测试、数学明确目标）。

如果任务只有一个正确答案且评估器噪声较大，搜索往往会使结果更差——它会找到一个"评分高"但错误的解答。

### 2026 年定位

大多数生产级智能体并不运行 LATS。它们运行的是带工具接地验证的 ReAct（CRITIC，见第05课）。搜索出现在专门的细分场景中：

- 将测试用例作为价值函数的代码智能体（HumanEval 风格）。
- 探索多条查询路径的深度研究智能体。
- LangGraph 子图中重规划的工作流。

AlphaEvolve（第11课）代表了 2025 年的极端情况：对代码进行进化搜索，使用机器可判定的适应度函数，取得前沿突破（56 年来首次改进 4x4 矩阵乘法）。

```figure
tree-of-thoughts
```

## 构建

`code/main.py` 实现了：

- 一个针对"选择算术运算"简化任务的微型 ToT BFS。
- 同一任务上的玩具 LATS MCTS 循环（选择/展开/模拟/回传），含 UCT 选择策略。
- 一个组合符号评分与自我评估分的价值函数。

运行方式：

```
python3 code/main.py
```

日志输出展示了 ToT 每个节点按 BFS 展开三个候选，与 LATS 通过 MCTS 收敛到最佳 rollout 的对比，以及双方的 token 计数。

## 使用

LangGraph 将 ToT 风格探索封装为子图模式；LangChain 团队关于 LATS 的博客（2024年5月）是参考教程。LlamaIndex 提供了 `TreeOfThoughts` 智能体。对大多数 2026 年的生产智能体而言，此模式位于一个 `if task_complexity > threshold: use_search()` 门控之后——参见第05课的评估器-优化器模式。

## 交付物

`outputs/skill-search-policy.md` 根据任务形态、预算和评估器保真度，在直线型 ReAct、ToT、LATS 和进化搜索之间做出选择。

## 练习

1. 用 UCT 参数 c=0.1 与 c=2.0 分别运行玩具 LATS。日志输出有何变化？
2. 将价值函数替换为噪声较大的评分器（加入随机扰动）。MCTS 是否仍能找到最优叶节点？它能容忍的最小信噪比是多少？
3. 实现 beam-search 风格的 ToT（每层保留 top-k），并与 BFS 对比。在严格 token 预算下哪种更好？
4. 阅读 LATS 论文第5.1节。复现 HumanEval 轨迹数：需要多少次 rollout 才能达到报告的 pass@1？
5. 阅读 LATS 论文中关于"何时 LATS 帮助有限"的讨论。写一段约一句话的决策规则，将任务形态映射到搜索策略。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|----------|
| Tree of Thoughts（思维树） | "分支式 CoT" | Yao 等——带有自我评估的思维节点树 |
| LATS | "LLM 的 MCTS" | Zhou 等——在 MCTS 下统一 ToT + ReAct + Reflexion |
| UCT | "上界置信区间" | 选择公式，平衡利用（Q）与探索（ln N / n） |
| Value function（价值函数） | "这个状态有多好" | 经 LLM 提示的评分或环境奖励；驱动回传 |
| Policy（策略） | "动作提议器" | ReAct 风格生成器；输出候选下一步思维/动作 |
| Rollout（回滚/展开） | "模拟轨迹" | 从节点沿策略走向叶节点，用价值函数打分 |
| Backpropagate（回传） | "更新祖先" | 将叶节点奖励沿路径向上推送，更新访问次数和 Q 值 |
| Search cost（搜索成本） | "token 爆炸" | Game of 24 上是 CoT 的 100-1000 倍；采用前需评估预算 |

## 延伸阅读

- [Yao et al., Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) —— 该领域的经典论文
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) —— 带 Reflexion 反馈的 MCTS
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— 搜索相关的子图模式
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) —— 使用程序化评估器的进化搜索
