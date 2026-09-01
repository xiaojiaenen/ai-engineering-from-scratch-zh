# 基准测试：SWE-bench、GAIA、AgentBench

> 2026 年，三个基准锚定了智能体评估。SWE-bench 测试代码补丁能力。GAIA 测试通用工具使用。AgentBench 测试多环境推理。了解它们的构成、污染历史以及它们无法测量的内容。

**类型：** 学习
**语言：** Python（标准库）
**前置知识：** Phase 14 · 06（工具使用）
**时间：** 约 60 分钟

## 学习目标

- 说出 SWE-bench 的测试机制（FAIL_TO_PASS），并解释它为何依赖单元测试。
- 说明 SWE-bench Verified（OpenAI，500 个任务）存在的意义以及它过滤掉了什么。
- 描述 GAIA 的设计哲学：对人类简单、对 AI 困难；三个难度等级。
- 列举 AgentBench 的八个环境，以及开源 LLM 面临的主要瓶颈。
- 总结 SWE-bench+ 的污染发现及其含义。

## 问题所在

排行榜能告诉你哪个模型在一个基准上得分最高。但它不会告诉你：

- 基准是否被污染（训练数据中存在解法、测试数据泄露）。
- 基准是否测量了你关心的能力（编码 vs 浏览 vs 通用）。
- 评估器是否鲁棒（AST 匹配、状态检查、人工审核）。

在你引用任何一个数字之前，先了解这三个锚定基准及其失效模式。

## 概念

### SWE-bench（Jimenez 等，ICLR 2024 oral）

- 来自 12 个热门 Python 仓库的 2,294 个真实 GitHub issue。
- 智能体获得：修复前的代码库 + 自然语言 issue 描述。
- 智能体产出：一个补丁（patch）。
- 评估器：应用补丁，运行仓库的测试套件。补丁必须将 FAIL_TO_PASS 测试（此前失败，现在通过）翻转过来，同时不破坏 PASS_TO_PASS 测试。

SWE-agent（Yang 等，2024）在发布时达到 12.5%，其重点在于智能体-计算机接口（模型能理解的文件编辑器命令、搜索语法）。

### SWE-bench Verified

OpenAI，2024 年 8 月。人工精选的 500 个任务子集。去除了模糊的 issue、不可靠的测试以及修复方案不明确的任务。是"你的智能体能否交付真实补丁"的主基准。

### 污染问题

- 超过 94% 的 SWE-bench issue 早于大多数模型的截断日期。
- **SWE-bench+** 发现，32.67% 的成功补丁中存在解法泄露（模型在 issue 描述中看到了修复方案），另有 31.08% 因测试覆盖薄弱而存疑。
- Verified 更干净，但并非无污染。

实际影响：一个在 SWE-bench 上得分为 50% 的模型，在 SWE-bench+ 上可能只有 35%。如果你声称 SWE-bench 成绩，应同时报告两个结果。

### GAIA（Mialon 等，2023 年 11 月）

- 466 道题目；其中 300 道保留用于 huggingface.co/gaia-benchmark 的私有排行榜。
- 设计哲学："对人类概念上简单（92% 的人类能答对），但对 AI 困难（GPT-4 with plugins：仅 15%）"。
- 测试推理、多模态、网页浏览、工具使用。
- 三个难度等级；Level 3 需要跨模态的长工具链。

GAIA 是用来衡量"通用能力"的基准。不要将其与代码专用基准混淆。

### AgentBench（Liu 等，ICLR 2024）

- 覆盖 8 个环境：编码（Bash、DB、KG）、游戏（Alfworld、LTP）、网页（WebShop、Mind2Web）和开放式生成。
- 多轮交互，每个分区约 4k–13k 步。
- 主要发现：长期推理、决策制定和指令遵循是开源 LLM 追赶商业模型的主要瓶颈。

### 这些基准无法测量的内容

- 真实世界的运行成本（token 消耗、 wall-clock 时间）。
- 对抗条件下的安全行为。
- 你在特定领域的表现（使用你自己的评估，见 Lesson 30）。
- 尾部故障（基准取平均值；生产运营者关心的是最差的 1%）。

### 基准测试常见的错误

- **单数字执念。** SWE-bench 50% 远不如 P50/P75/P95 的成本 + 步骤分布有信息量。
- **受污染的声明。** 报告 SWE-bench 成绩却不提及 Verified 或 SWE-bench+，具有误导性。
- **以基准为开发目标。** 针对基准优化会偏离生产实用性。

```figure
ae-swebench-gate
```

## 动手实践

`code/main.py` 实现了一个玩具版 SWE-bench 风格评估框架：

- 合成的 bug 修复任务（3 个任务）。
- 一个脚本化"智能体"，负责提出补丁。
- 一个测试运行器，检查 FAIL_TO_PASS（bug 已修复）和 PASS_TO_PASS（未引入回归）。
- 一个基于问题分解深度的 GAIA 风格难度分类器。

运行方式：

```
python3 code/main.py
```

输出将展示每个任务的解决率及按难度分组的统计，使评估规则具体化。

## 如何使用

- **SWE-bench Verified** 用于代码智能体。务必报告 Verified 得分。
- **GAIA** 用于通用智能体。使用私有排行榜分区。
- **AgentBench** 用于多环境对比。
- **自定义评估**（Lesson 30）用于你产品的真实形态。

## 交付物

`outputs/skill-benchmark-harness.md` 为任意代码库-任务对构建一个 SWE-bench 风格的评估框架，支持 FAIL_TO_PASS / PASS_TO_PASS 门控。

## 练习

1. 将玩具评估框移植到一个真实仓库（选择你自己的）。为已知 bug 编写 3 个 FAIL_TO_PASS 测试。
2. 添加步骤计数指标。在你的 3 个任务上，每个成功修复平均需要多少智能体步骤？
3. 阅读 SWE-bench+ 论文。实现一个解法泄露检查（对 issue 文本与 diff 进行模式匹配）。
4. 从公开分区下载一道 GAIA 题目。追踪一个 GPT-4 级别的智能体会如何作答。它需要哪些工具？
5. 阅读 AgentBench 的各环境细分数据。哪个环境与你的产品形态最相似？那里的"SOTA"表现如何？

## 关键术语

| 术语 | 人们通常的说法 | 实际含义 |
|------|---------------|---------|
| SWE-bench | "代码智能体基准" | 2,294 个 GitHub issue；补丁必须翻转 FAIL_TO_PASS 测试 |
| SWE-bench Verified | "干净的 SWE-bench" | 500 个人工精选任务，由 OpenAI 提供 |
| FAIL_TO_PASS | "修复门控" | 此前失败的测试，补丁后必须通过 |
| PASS_TO_PASS | "无损门控" | 原本通过的测试，补丁后仍须通过 |
| GAIA | "通用基准" | 466 道对人类简单、对 AI 困难的多工具题目 |
| AgentBench | "多环境基准" | 8 个环境；长周期多轮交互 |
| Contamination | "训练集泄露" | 基准任务出现在模型训练数据中 |
| SWE-bench+ | "污染审计" | 发现 SWE-bench 成功补丁中有 32.67% 存在解法泄露 |

## 延伸阅读

- [Jimenez 等，SWE-bench（arXiv:2310.06770）](https://arxiv.org/abs/2310.06770) — 原始基准论文
- [OpenAI，SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 精选子集
- [Mialon 等，GAIA（arXiv:2311.12983）](https://arxiv.org/abs/2311.12983) — 通用能力基准
- [Liu 等，AgentBench（arXiv:2308.03688）](https://arxiv.org/abs/2308.03688) — 多环境评测套件
