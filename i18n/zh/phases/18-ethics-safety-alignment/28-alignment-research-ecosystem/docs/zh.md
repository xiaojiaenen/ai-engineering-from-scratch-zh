# 对齐研究生态 — MATS、Redwood、Apollo、METR

> 五家机构定义了2026年的非实验室对齐研究层。MATS（ML Alignment & Theory Scholars）：自2021年底以来已有527+研究员，发表180+论文，引用超1万次，h指数47；2024年暑期批次作为501(c)(3)注册，拥有约90名学者和40名导师；2025年前的毕业生中80%从事安全/安保工作，其中200+人在Anthropic、DeepMind、OpenAI、英国AISI、RAND、Redwood、METR、Apollo任职。Redwood Research：应用对齐实验室，由Buck Shlegeris创立；引入AI Control（第10课）；与英国AISI合作开展控制安全案例研究。Apollo Research：面向前沿实验室的前部署欺骗评估；撰写了《In-Context Scheming》（第8课）和《Towards Safety Cases for AI Scheming》。METR（Model Evaluation and Threat Research）：基于任务的 Capabilities 评估、自主任务时间跨度研究；《Common Elements of Frontier AI Safety Policies》对比各实验室框架。Eleos AI Research：模型福利前部署评估（第19课）；开展了Claude Opus 4福利评估。

**类型：** 学习
**语言：** 无
**前置条件：** 第18阶段 · 01-27（之前第18阶段课程）
**时间：** 约45分钟

## 学习目标

- 识别非实验室对齐研究生态系统的五家机构及其核心产出。
- 描述MATS的规模（学者人数、论文数、h指数）及其作为人才管道的作用。
- 描述Redwood的AI Control议程及其与英国AISI的合作。
- 描述METR的基于任务的评估方法。

## 问题

前沿实验室（第18课）内部进行安全评估并发布精选结果。实验室外部的生态系统才是验证评估、首次发现新型失效模式以及培养人才的场所。理解生态系统有助于解读哪些研究成果被谁所信任。

## 概念

### MATS（ML Alignment & Theory Scholars）

始于2021年底。研究导师项目；学者与高级研究员合作10-12周，解决具体的对齐问题。

规模（2026年）：
- 自成立以来的研究员超过527人。
- 发表论文超过180篇。
- 引用超过1万次。
- h指数47。
- 2024年暑期：90名学者 + 40名导师；注册为501(c)(3)。

职业成果：约80%的2025年前毕业生从事安全/安保工作。200+人在Anthropic、DeepMind、OpenAI、英国AISI、RAND、Redwood、METR、Apollo任职。

### Redwood Research

应用对齐实验室。由Buck Shlegeris创立。提出了AI Control议程（第10课）。与英国AISI合作开展控制安全案例。就评估设计为DeepMind和Anthropic提供咨询。

经典论文：Greenblatt、Shlegeris等人，《AI Control》（arXiv:2312.06942，ICML 2024）；Alignment Faking（Greenblatt、Denison、Wright等人，arXiv:2412.14093，与Anthropic联合）。

风格：具体的威胁模型、最坏情况下的对手、可压测的具体协议。

### Apollo Research

为前沿实验室提供前部署欺骗评估。撰写了《In-Context Scheming》（第8课，arXiv:2412.04984）。是2025年OpenAI反欺骗训练合作的合作伙伴。制作了《Towards Safety Cases for AI Scheming》（2024年）。

风格：代理设定下的评估，其中欺骗可能涌现；三支柱分解（未对齐、目标导向性、情境意识）。

### METR（Model Evaluation and Threat Research）

基于任务的 Capabilities 评估。自主任务完成时间跨度研究。《Common Elements of Frontier AI Safety Policies》（metr.org/common-elements，2025）对比各实验室框架。

与Apollo合作撰写了AI Scheming安全案例概要。

风格：长跨度任务评估、实证能力测量、框架综合。

### Eleos AI Research

模型福利前部署评估。开展了记录在系统卡第5.3节的Claude Opus 4福利评估。为第19课的福利相关主张提供外部方法学核查。

### 流程

MATS培养研究员。毕业生流向Anthropic、DeepMind、OpenAI（实验室安全团队）或Redwood、Apollo、METR、Eleos（外部评估）。外部评估者与实验室及英国AISI / CAISI合作。出版物回馈生态系统至下一届MATS。

### 为何这一层很重要

单一来源的评估不可靠：实验室自我评估存在结构性利益冲突。外部评估者可以提出并验证实验室可能低报的失效模式。2024年《Sleeper Agents》论文（第7课）是Anthropic + Redwood的成果；《Alignment Faking》是Anthropic + Redwood；《In-Context Scheming》出自Apollo；《Anti-Scheming》是Apollo + OpenAI。多机构结构即是质量控制。

### 在Phase 18中的位置

第7-11课引用了Redwood和Apollo的工作；第18课引用了METR的框架对比；第19课引用了Eleos。第28课是这一生态系统的明确组织映射，整个Phase的其他课程都依赖于此。

```figure
sae-features
```

## 使用它

无代码。阅读METR的《Common Elements of Frontier AI Safety Policies》，作为外部综合如何为实验室内部政策工作增添价值的示例。

## 完成它

本课程产出 `outputs/skill-ecosystem-map.md`。给定一个对齐主张或评估，它识别涉及的机构、发表渠道和方法论风格，并与已知对应机构进行交叉核查。

## 练习

1. 从第7-15课中挑选一篇论文，识别涉及的机构。交叉核查作者与MATS毕业生及当前生态系统隶属关系。

2. 阅读METR的《Common Elements of Frontier AI Safety Policies》。识别它们强调的三个跨实验室共识点和两个最大分歧点。

3. MATS的职业成果约80%从事安全/安保工作。论证这种选择压力是适应性的（培养领域）还是存在偏差的（过滤掉异端立场）。

4. Redwood和Apollo都从事控制/欺骗相关工作，但风格不同。选择一个失效模式，描述它们各自会如何调查。

5. Eleos AI是唯一纯粹的模型福利机构。设计一个假设的第二机构，专注于不同的福利相关问题（认知自由、机器人具身化等），并阐明其方法学。

## 关键术语

| 术语 | 人们说什么 | 实际含义 |
|------|-----------------|------------------------|
| MATS | "导师项目" | ML Alignment & Theory Scholars；自2021年以来527+研究员 |
| Redwood Research | "控制实验室" | 应用对齐；AI Control作者；英国AISI合作伙伴 |
| Apollo Research | "欺骗评估" | 面向前沿实验室的前部署欺骗评估 |
| METR | "任务跨度评估" | 基于任务的 Capabilities 评估；框架综合 |
| Eleos AI | "福利实验室" | 模型福利前部署评估 |
| 人才管道 | "MATS -> 实验室" | MATS毕业生流向Anthropic、DM、OpenAI、Redwood、Apollo、METR |
| 外部评估 | "非实验室核查" | 不由模型生产者进行的评估；增加可信度 |

## 延伸阅读

- [MATS（ML Alignment & Theory Scholars）](https://www.matsprogram.org/) — 导师项目
- [Redwood Research](https://www.redwoodresearch.org/) — AI Control论文
- [Apollo Research](https://www.apolloresearch.ai/) — 欺骗评估
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 框架对比
- [Eleos AI Research](https://www.eleosai.org/research) — 模型福利方法学
