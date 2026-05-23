# 自主编码智能体发展全景（2026）

> SWE-bench Verified 的分数在不到三年内从 4% 飙升至 80.9%。同一款 Claude Sonnet 4.5 在 SWE-agent v1 上得分 43.2%，在 Cline autonomous 上却达到 59.8% —— 如今，模型周围的脚手架（scaffolding）与模型本身同样重要。OpenHands（前身为 OpenDevin）是最活跃的 MIT 许可平台，其 CodeAct 循环直接在沙盒中执行 Python 动作，而非使用 JSON 工具调用。亮眼的数字背后隐藏着方法论问题：SWE-bench Verified 的 500 个任务中，有 161 个只需修改 1-2 行代码；而面对更难的 SWE-bench Pro（任务需修改 10 行以上），同样的前沿模型得分仅为 23-59%。

**类型：** 学习
**语言：** Python（标准库，CodeAct 与 JSON 工具调用对比）
**先修知识：** 阶段 14 · 07（工具使用），阶段 15 · 01（长时域智能体）
**时间：** 约 45 分钟

## 核心问题

“哪个编码智能体最好”是一个错误的问题。正确的问题是：在一个匹配我工作性质的任务分布上，使用我将部署到生产环境的脚手架，我能获得怎样的端到端可靠性？

在 2022 年到 2026 年间，该领域认识到脚手架——即检索层、规划器、沙盒、编辑-验证循环、反馈格式——是承重结构。Claude Sonnet 4.5 在 SWE-agent v1 上于 SWE-bench Verified 中得分 43.2%；同一个模型在 Cline 的自主脚手架中得分却达到 59.8%。绝对分差 16.6 分，权重相同。基础模型是一个组件；而循环才是产品。

伴随的问题是基准测试的饱和掩盖了倒退。SWE-bench Verified 已接近饱和，其容易任务尾部（500 个任务中有 161 个需要 ≤2 行修改）拉高了顶级分数。现实世界的质量更适合在类似 SWE-bench Pro（需修改 10 行以上）这样的分布上衡量，而在这上面，同样的领先者分数仍停留在 23-59%。

## 核心概念

### SWE-bench，一段话概括

SWE-bench（Jimenez 等人）利用带有真实补丁的 GitHub 问题，要求智能体生成一个能让测试套件通过的补丁。SWE-bench Verified（OpenAI, 2024）是一个人工筛选的 500 个任务子集，剔除了模糊和损坏的任务。SWE-bench Pro 是其更难的后续版本——任务需要 10 行以上的修改，目前的顶尖智能体得分在 23-59% 之间。

### 2022 → 2026 年曲线实际显示了什么

- **2022**：研究模型在原始 SWE-bench 上约为 4%。
- **2024**：GPT-4 + Devin 风格脚手架约为 14%；SWE-agent 约为 12%。
- **2025**：Claude 3.5/3.7 Sonnet 在 Aider 和 SWE-agent 中推动分数进入 40-55% 区间。
- **2026**：Claude Sonnet 4.5 及前沿竞争者在 SWE-bench Verified 上达到 70-80%+。Epoch AI 的排行榜实时追踪此数据。

曲线的斜率源于三个复合作用：更好的基础模型、更好的脚手架（CodeAct、反思、验证器循环）以及更好的基准测试（Verified 版本去除了噪声）。

### CodeAct 与 JSON 工具调用

OpenHands（All-Hands-AI, arXiv:2407.16741，前身为 OpenDevin）做出了一个特定的架构赌注：模型不输出由主机解码执行的 JSON 工具调用，而是输出 Python 代码，并由一个 Jupyter 风格的内核在沙盒中运行它。智能体可以在一个动作中循环处理文件、链接工具并捕获自身的异常。

权衡如下：

- **JSON 工具调用**：每个动作一轮交互；易于审计；组合性有限；默认安全，因为每次调用都经过显式验证器。
- **CodeAct**：一个动作可以是一个完整的程序；组合性强；需要强化的沙盒（OpenHands 使用 Docker 隔离）；失败模式包括沙盒运行时允许的任何情况。

两种架构均已投入生产。CodeAct 在开源平台（OpenHands、smolagents）中占主导地位。JSON 工具调用则在托管服务（Anthropic Managed Agents、OpenAI Assistants）中占主导，因为这些服务提供商控制着执行器。

### 2026 年全景中的脚手架

| 脚手架 | 许可证 | 执行模型 | 显著特点 |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | Docker 中的 CodeAct | 最活跃的开源平台；事件流可重放 |
| SWE-agent | MIT | 智能体-计算机接口（ACI） | 首个端到端 SWE-bench 脚手架 |
| Aider | Apache-2 | 在本地仓库中通过 diff 编辑 | 极简脚手架，回归稳定性强 |
| Cline | Apache-2 | VS Code 智能体，带工具策略 | 在 Sonnet 4.5 上得分最高的开源脚手架 |
| Devin (Cognition) | 专有 | 托管 VM + 规划器 | 首个“AI 软件工程师”产品类别 |
| Claude Code | 专有 | 权限模式 + 例程 | 第 10 课详细介绍了智能体循环 |

### 为何脚手架占主导地位

一次编码运行是一个长时域轨迹（第 1 课）。可靠性在各步骤间复利增长。脚手架在三个地方获取分数：

1.  **检索**：找到正确的文件来读取是无声的瓶颈。SWE-agent 的 ACI、OpenHands 的文件索引和 Aider 的仓库地图都针对此点。
2.  **验证器循环**：运行测试、读取堆栈跟踪、重新尝试，在 SWE-bench 上带来 10 分以上的提升。
3.  **故障 containment**：一个能在出错时回滚的沙盒可防止损害累积。同一个模型，有验证器循环和没有验证器循环，看起来像是两个不同的产品。

### 基准测试饱和与真实分布

OpenHands 的作者和 Epoch AI 都指出，SWE-bench Verified 存在一个容易尾部：500 个任务中有 161 个只需 1-2 行修改。高分部分是由这个尾部驱动的。SWE-bench Pro 将任务限制在 10 行以上的修改，即使是顶尖系统，得分也回落到 23-59% 区间。你的生产环境分布几乎肯定更接近 Pro 而不是 Verified。

对选择智能体的启示：在你自己的 bug 积压任务中运行一个类似 Pro 的子集。重要的分数是在那些代表你实际发货任务的集合上获得的分数。

## 动手实践

`code/main.py` 在一个固定的小型任务分布上比较了两个玩具级智能体脚手架：

1.  一个**JSON 工具调用**脚手架，每轮执行一个动作。
2.  一个**CodeAct** 脚手架，每轮可以输出一小段 Python 代码片段。

两者都使用一个存根“模型”（确定性规则），因此比较隔离了脚手架与模型质量的影响。输出显示，CodeAct 脚手架用更少的轮次解决了更多任务，但代价是每个动作的潜在影响范围更大。

## 部署准备

`outputs/skill-scaffold-audit.md` 可帮助你在采用前审计一个拟议的编码智能体脚手架：检索质量、验证器是否存在、沙盒隔离程度，以及基准测试与分布的匹配度。

## 练习

1.  运行 `code/main.py`。每个脚手架在相同任务集上分别需要多少轮次？每个脚手架每个动作的潜在影响范围有多大？
2.  阅读 OpenHands 论文（arXiv:2407.16741）。论文认为 CodeAct 在复杂任务上优于 JSON 工具调用。找出论文承认的一种失败模式，并写一句话说明该模式在何种情况下会在生产环境中占主导。
3.  从你的 bug 积压任务中挑选一个需要跨两个文件修改 10 行以上的任务。估算前沿模型在 (a) JSON 工具调用和 (b) CodeAct 下的端到端成功概率。解释其中的差距。
4.  SWE-bench Verified 有 161 个单文件、1-2 行修改的任务。构建一个排除这些任务的分数。排行榜排名会如何变化？
5.  阅读《Introducing SWE-bench Verified》（OpenAI）。解释用于移除模糊任务的具体方法，并指出一个该筛选可能会遗漏的类别。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|---|---|---|
| SWE-bench | “编码基准” | 真实的 GitHub 问题，带有真实补丁和测试套件 |
| SWE-bench Verified | “清理过的子集” | 500 个人工筛选的任务，存在容易尾部 |
| SWE-bench Pro | “更难的子集” | 需修改 10 行以上；顶尖系统得分 23-59% |
| CodeAct | “代码即动作” | 智能体输出 Python；由 Jupyter 风格内核在沙盒中执行 |
| JSON 工具调用 | “函数调用” | 每个动作是一个结构化 JSON 载荷，执行前需验证 |
| 脚手架 | “智能体框架” | 基础模型周围的检索 + 规划器 + 执行器 + 验证器循环 |
| ACI（智能体-计算机接口） | “SWE-agent 的格式” | 为 LLM 人体工学设计的命令集，而非人类 shell |
| 验证器循环 | “测试-重试” | 运行测试、读取输出、修订补丁；最大的非模型可靠性收益 |

## 延伸阅读

- [Jimenez 等人 — SWE-bench](https://www.swebench.com/) — 原始基准和方法论。
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 人工筛选子集的构建方式。
- [Wang 等人 — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — CodeAct 架构与事件流设计。
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) — 实时追踪的分数。
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 长时域编码智能体可靠性的框架。