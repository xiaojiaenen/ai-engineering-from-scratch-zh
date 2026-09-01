# Anthropic 负责扩展政策 v3.0

> RSP v3.0 于 2026 年 2 月 24 日正式生效，取代了 2023 年的政策。双层缓解机制：Anthropic 将单方面采取的行动 vs 作为行业范围建议提出（包括 RAND SL-4 安全标准）。新增前沿安全路线图和风险报告作为常规文件，而非一次性交付物。删除了 2023 年的暂停承诺。引入 AI R&D-4 门槛：一旦跨越，Anthropic 必须发布明确论证，说明对齐风险和缓解措施。Claude Opus 4.6 未跨越此门槛。Anthropic 在 v3.0 公告中表示" confidently ruling this out is becoming difficult." SaferAI 对 2023 年 RSP 评分为 2.2；他们给 v3.0 降至 1.9，使 Anthropic 与 OpenAI 和 DeepMind 一同被列为"weak"RSP 类别。定性阈值取代了 2023 年的定量承诺；移除暂停条款是最显著的倒退。

**类型：** 学习
**语言：** Python (stdlib, RSP 门槛决策引擎)
**前置条件：** Phase 15 · 06 (AAR), Phase 15 · 07 (RSI)
**时间：** 约 45 分钟

## 问题

前沿实验室发布的扩展政策既是技术文档，也是治理文件，同时是对监管机构的信号。RSP v3.0 是当前的 Anthropic 文档。仔细阅读它很重要，不是因为遵守它是强制性的（并非如此），而是因为框架塑造了实验室如何构想灾难性风险以及如何向公众沟通权衡。

v3.0 与 v2.0 的差异是有用的分析单位。新增内容：前沿安全路线图、风险报告、AI R&D-4 门槛。删除内容：2023 年暂停承诺。重构内容：双层缓解计划，分为 Anthropic 单方面行动和行业建议两部分。外部评审——SaferAI——将评分从 2.2（v2）降至 1.9（v3.0）。这就是一个扩展政策如何在看起来更精致的同时变得不够严格。

## 概念

### 双层缓解计划

- **Anthropic 单方面行动**：无论其他实验室做什么，Anthropic 都将采取的行动。训练在阈值之上停止，特定的安全措施，特定的部署门槛。
- **行业范围建议**：Anthropic 认为整个行业应该集体采取的行动。包括 RAND SL-4 安全标准。这些不是 Anthropic 方面的承诺；它们是政策倡导。

双层结构在 v2 中不存在。这意味着读者需要查看每个承诺属于哪一列。在"行业范围建议"列中的安全措施不是 Anthropic 的承诺；它是 Anthropic 的希望。

### AI R&D-4 门槛

这是 RSP v3.0 命名的下一个重要门槛的能力水平。具体来说：能够在竞争性成本下自动化大量 AI 研究的研究级模型。一旦 Anthropic 认为一个模型跨越了该门槛，他们必须在继续扩展之前发布明确论证，说明对齐风险和缓解措施。

根据 v3.0 公告，Claude Opus 4.6 未跨越该门槛。文档补充道："confidently ruling this out is becoming difficult."该措辞很重要；它承认该门槛已足够接近，成为一个现实关注点，而非假设性限制。

第 6 课（自动化对齐研究）和第 7 课（递归自我改进）直接与此门槛相关。自动化对齐研究者达到研究质量水平是 AI R&D-4 门槛正在逼近的证据。

### 前沿安全路线图与风险报告

v3.0 将两类产物提升为常规文件：

- **前沿安全路线图**：前瞻性文档，描述计划的安全工作、能力预期和缓解研究。
- **风险报告**：针对特定模型发布后的回顾性文档，描述观察到的能力和残留风险。

两者都是公开的。两者都按既定周期更新。其用处在于：读者可以追踪 Anthropic 在路线图中所说会做什么与他们风险报告中报告的内容相比如何。

### 移除暂停条款

2023 年 RSP 包含明确的暂停承诺：如果模型跨越特定能力门槛，训练将暂停，直到缓解措施到位。v3.0 用较软的表述替代了明确暂停（发布明确论证，如果缓解措施充分则继续）。SaferAI 和其他分析师直接将此指为文档中最强的倒退。

该变更的政策论证：2023 年的定量阈值在 2026 年的能力基准测试中变得无法达到，因为基准本身已被重新缩放。反驳论点：扩展政策中的暂停条款是一种承诺机制；移除它消除了政策的可信度。

### SaferAI 降级

SaferAI 是一个独立组织，对 RSP 类文档进行评级。他们的公开评级：2023 年 Anthropic RSP 得分为 2.2（满分 4.0，最低 1.0）。v3.0 得分为 1.9。这使 Anthropic 从"moderate"降级到"weak"，与 OpenAI 和 DeepMind 同列。

根据 SaferAI 的分析，降级因素包括：
- 定性阈值取代了定量阈值。
- 暂停承诺被移除。
- AI R&D-4 门槛的缓解措施被描述为"明确论证"而非具体措施。
- 审查机制依赖于 Anthropic 的安全咨询委员会，独立监督有限。

### 本课内容不包括什么

这不是合规课程。RSP v3.0 不是法规；没有任何力量迫使 Anthropic 遵守它。本课的目的是以应有的精确性和怀疑精神阅读该文档。扩展政策是前沿实验室就灾难性风险立场发出的主要公开信号。对于工作依赖前沿能力的人来说，良好阅读它们是一项实用技能。

```figure
a5-rsp-ladder
```

## 实践使用

`code/main.py` 实现了一个小型决策引擎，镜像 RSP 门槛评估结构：给定候选模型和能力测量集，返回是否跨越 AI R&D-4 门槛、所需的明确论证部分，以及部署是否可以继续。它刻意保持简单；目的是使文档的逻辑显式化。

## 产出交付

`outputs/skill-scaling-policy-review.md` 对照 v3.0 参考对扩展政策（Anthropic、OpenAI、DeepMind 或内部）进行评审：双层结构、门槛、暂停承诺、独立审查。

## 练习

1. 运行 `code/main.py`。输入三个不同能力水平的合成模型。确认门槛评估器按预期工作并生成正确的明确论证模板。

2. 完整阅读 RSP v3.0（32 页）。找出所有位于"行业范围建议"层的承诺。其中哪些在 v2 中会是"Anthropic 单方面"承诺？

3. 阅读 SaferAI 的 RSP 评级方法论。通过对其评分标准应用于该文档，重现他们对 v3.0 的 1.9 分。哪个评分行驱动了最多降级？

4. 2023 年的暂停承诺被移除。提出一个替代承诺，在承认 2026 年基准重新缩放问题的同时保留政策的可信度。

5. 将 RSP v3.0 与 OpenAI 准备框架 v2（第 20 课）进行比较。选出 v3.0 更强的一项，再选出准备框架更强的一项。

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|---|---|---|
| RSP | "Anthropic 的扩展政策" | 负责扩展政策；v3.0 于 2026 年 2 月 24 日生效 |
| AI R&D-4 | "研究自动化门槛" | 能够在竞争性成本下自动化大量 AI 研究的能力 |
| Affirmative case | "安全论证" | 发表的论证，说明风险已被识别且缓解措施充分 |
| Frontier Safety Roadmap | "前进计划" | 关于计划安全工作和预期能力的常规文件 |
| Risk Report | "模型回顾" | 发布后关于观察到的能力和残留风险的常规文件 |
| Two-tier mitigation | "单方面 vs 行业" | 区分 Anthropic 承诺与行业建议 |
| Pause commitment | "2023 条款" | 明确承诺暂停训练；在 v3.0 中已移除 |
| SaferAI rating | "独立 RSP 评级" | 第三方评分标准；v3.0 得分为 1.9（v2 为 2.2）|

## 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 完整的 32 页政策。
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) — 从 v2 变更的摘要。
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/research/frontier-safety) — 来自 RSP v3.0 链接的常规文件。
- [Anthropic — Risk Report: Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) — 对当前前沿模型的回顾。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 将 AI R&D-4 与实际测量的自主性连接。
