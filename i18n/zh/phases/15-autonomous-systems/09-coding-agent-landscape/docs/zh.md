# 自主编程智能体生态格局（2026）

> SWE-bench Verified 在不到三年的时间内从 4% 提升到了 80.9%。同一款 Claude Sonnet 4.5 模型在 SWE-agent v1 上得分 43.2%，在 Cline 自主模式下得分 59.8% —— 模型周围的脚手架（scaffolding）现在与模型本身同等重要。OpenHands（前 OpenDevin）是最活跃的使用 MIT 许可的平台，其 CodeAct 循环直接在沙箱中执行 Python 操作，而非 JSON 工具调用。 headline 数字隐藏了一个方法学问题：SWE-bench Verified 的 500 个任务中有 161 个只需要 1–2 行的修改，而针对相同前沿模型，SWE-bench Pro（10+ 行修改的任务）的得分仅处于 23–59% 区间。

**类型：** 学习
**语言：** Python（标准库，CodeAct vs JSON tool-call comparison）
**前置要求：** Phase 14 · 07（Tool use），Phase 15 · 01（Long-horizon agents）
**时间：** 约 45 分钟

## 问题所在

“哪种编程智能体最好”是个错误的问题。正确的问题是：在与我的工作相匹配的任务分布上，使用我将在生产环境中运行的脚手架，我能获得怎样的端到端可靠性？

2022 年至 2026 年间，该领域认识到脚手架——检索层、规划器、沙箱、edit-verify loop、反馈格式——起着关键支撑作用。Claude Sonnet 4.5 在 SWE-agent v1 上的 SWE-bench Verified 得分为 43.2%；同一模型在 Cline 自主脚手架中的得分为 59.8%。相同的权重，相差 16.6 个百分点。基础模型只是一个组件；循环才是产品。

伴生的问题是，基准测试的饱和现象掩盖了性能倒退。SWE-bench Verified 已接近饱和，而简单任务尾段（500 个任务中有 161 个仅需 ≤2 行修改）拉高了头部分数。真实世界的质量更适合在 SWE-bench Pro（10+ 行修改）等分布上进行衡量，在此榜单上，同样的领先模型得分仍处于 23–59%。

## 核心概念

### SWE-bench：一段话概述

SWE-bench（Jimenez et al.）选取带有 ground-truth patch 的 GitHub Issue，要求智能体生成能让测试套件通过的 patch。SWE-bench Verified（OpenAI，2024）是一个人工筛选的 500 任务子集，已移除歧义和损坏的任务。SWE-bench Pro 是更难的继任版本——任务需要 10+ 行的修改，当前前沿智能体在此的得分处于 23–59%。

### 2022 → 2026 曲线实际揭示了什么

- **2022**：原始 SWE-bench 上，研究模型得分约 4%。
- **2024**：GPT-4 + Devin-style scaffolding 约 14%；SWE-agent 约 12%。
- **2025**：Aider 和 SWE-agent 内的 Claude 3.5/3.7 Sonnet 推入 40–55% 区间。
- **2026**：Claude Sonnet 4.5 及前沿竞品在 SWE-bench Verified 上达到 70–80%+。Epoch AI 的排行榜实时追踪此数据。

分数的攀升来自三个叠加来源：更好的基础模型、更好的脚手架（CodeAct、reflection、verifier loops）以及更好的基准测试（Verified 移除了噪声）。

### CodeAct vs JSON tool calls

OpenHands（All-Hands-AI，arXiv:2407.16741，前 OpenDevin）采取了一项特定的架构抉择：模型不输出由宿主解码执行的 JSON 工具调用，而是直接输出 Python 代码，由类 Jupyter 的内核在沙箱中运行。智能体可以在一个 action 内循环处理文件、串联工具，并捕获自身抛出的异常。

权衡如下：

- **JSON tool calls**：每个 action 对应一次交互；易于 audit；组合性有限；默认安全，因为每次调用都经过显式 validator。
- **CodeAct**：一个 action 可以是一个完整程序；具备组合性；需要加固的沙箱（OpenHands 使用 Docker isolation）；failure modes 涵盖沙箱 runtime 允许的任何操作。

两种架构均已投入生产。CodeAct 在开源平台中占据主导（如 OpenHands、smolagents）。JSON tool calls 在托管服务中仍占主导（如 Anthropic Managed Agents、OpenAI Assistants），因为提供方控制着 executor。

### 2026 年格局中的脚手架

| Scaffold | License | Execution model | Notable property |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | Docker 中的 CodeAct | 最活跃的开源平台；event-stream 可重放 |
| SWE-agent | MIT | Agent-Computer Interface (ACI) | 首个端到端 SWE-bench scaffold |
| Aider | Apache-2 | 本地仓库 diff 编辑 | 极简 scaffold，回归稳定性强 |
| Cline | Apache-2 | 带 tool policy 的 VS Code agent | 在 Sonnet 4.5 上得分最高的开源 scaffold |
| Devin (Cognition) | Proprietary | Managed VM + planner | 首个“AI software engineer”产品类别 |
| Claude Code | Proprietary | Permission modes + routines | Lesson 10 详细讲解 agent loop |

### 为什么脚手架占据主导

一次 coding run 是一条 long-horizon trajectory（Lesson 1）。可靠性会随步骤累积。scaffolding 能带来提分的三个关键环节：

1. **Retrieval**：找到需要阅读的正确文件是隐形的瓶颈。SWE-agent 的 ACI、OpenHands 的 file-index 以及 Aider 的 repo-map 均致力于解决此问题。
2. **Verifier loop**：运行测试、阅读 stack trace 并重新尝试，在 SWE-bench 上能带来 10+ 个百分点的提升。
3. **Failure containment**：能在出错时回滚的沙箱可防止损害累积。有无 verifier loop 的同一模型，表现宛如两款不同的产品。

### Benchmark saturation and the real distribution

OpenHands 的作者与 Epoch AI 均指出 SWE-bench Verified 存在简单任务尾段：500 个任务中有 161 个仅需 1–2 行修改。高分部分由该尾段驱动。SWE-bench Pro 将任务限定为 10+ 行修改，即便对于前沿系统，得分也仅返回 23–59% 的区间。你的生产环境任务分布几乎肯定更接近 Pro 而非 Verified。

对选择 agent 的启示：运行一个与你自身 bug backlog 相匹配的 Pro-like subset。真正有意义的分数，是那些代表你实际交付内容的任务得分。

```figure
a5-scaffold-delta
```

## 实践应用

`code/main.py` 在固定的微型任务分布上对比了两种 toy agent scaffolds：

1. 一个 **JSON tool-call** scaffold：每次交互执行一个 action。
2. 一个 **CodeAct** scaffold：每个 action 可输出一个小型 Python snippet。

两者均使用占位符“model”（确定性规则），以便将对比隔离于 model quality 之外。输出结果显示，CodeAct scaffold 以更少的 turns 解决了更多任务，代价是每个 action 的 blast radius 更大。

## 部署落地

`outputs/skill-scaffold-audit.md` 帮助你在采用前 audit 候选的 coding-agent scaffold：retrieval quality、verifier presence、sandbox isolation，以及 benchmark-to-distribution fit。

## 练习

1. 运行 `code/main.py`。每种 scaffold 在相同任务集上各需多少次 turns？各自的 per-action blast radius 是多少？

2. 阅读 OpenHands paper（arXiv:2407.16741）。论文论证 CodeAct 在 complex tasks 上优于 JSON tool calls。指出论文承认的一种 failure mode，并用一句话说明该 mode 在 production 环境中何时会占主导。

3. 从你的 bug backlog 中选取一个需要跨两个文件修改 10+ 行的任务。估算 frontier model 在 (a) JSON tool calls 和 (b) CodeAct 下的端到端成功概率。论证两者差距的原因。

4. SWE-bench Verified 包含 161 个单文件、1–2 行的 tasks。构建一个剔除它们的 score metric。排行榜会发生怎样的 shuffle？

5. 阅读《Introducing SWE-bench Verified》（OpenAI）。解释用于移除 ambiguous tasks 的具体 methodology，并列举一个会被 curation 遗漏的 category。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| SWE-bench | “Coding benchmark” | 带有 ground-truth patch 和 test suite 的真实 GitHub Issue |
| SWE-bench Verified | “Cleaned subset” | 500 个人工筛选 tasks，存在 easier-tail |
| SWE-bench Pro | “Harder subset” | 10+ line changes；frontier sits at 23–59% |
| CodeAct | “Code-as-action” | Agent emits Python；Jupyter-style kernel executes in sandbox |
| JSON tool call | “Function calling” | 每个 action 是结构化 JSON payload，执行前需验证 |
| Scaffold | “Agent framework” | 围绕 base model 的 Retrieval + planner + executor + verifier loop |
| ACI (Agent-Computer Interface) | “SWE-agent's format” | 为 LLM ergonomics 设计的 command set，而非 human shells |
| Verifier loop | “Test-and-retry” | 运行测试、读取 output、修订 patch；最大的 non-model reliability gain |

## 延伸阅读

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — 原始 benchmark 及 methodology。
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — curated subset 的构建过程。
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — CodeAct architecture 与 event-stream design。
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) — live-tracked scores。
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — long-horizon coding-agent reliability framing。
