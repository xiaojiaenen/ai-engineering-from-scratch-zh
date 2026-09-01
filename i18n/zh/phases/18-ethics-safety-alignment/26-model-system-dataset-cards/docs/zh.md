# Model Card、System Card 与 Dataset Card

> 三种文档格式构建 AI 透明度。Model Card（Mitchell 等，2019）——模型的"营养成分标签"：训练数据、定量分层分析、伦理考量、注意事项；Hugging Face 上的 model card 中仅有 0.3% 记录了伦理考量（Oreamuno 等，2023）。Datasheets for Datasets（Gebru 等，2018，CACM）——动机、构成、收集流程、标注、分发、维护；类比电子数据手册。Data Card（Pushkarna 等，Google 2022）——模块化分层细节（ telescopic / periscopic / microscopic 三层缩放），作为面向不同读者的边界对象。2024–2025 年发展：通过 LLM 自动生成（CardGen，Liu 等，2024）；model card 详细程度与 HF 上下载量最高增加 29% 相关（Liang 等，2024）；可验证声明（Laminator，Duddu 等，2024）；碳排放/用水可持续性报告新增（Jouneaux 等，2025 年 7 月）；欧盟/ISO 监管卡片相继出现。System Card（Sidhpurwala 2024；Meta 系统级透明度；"Blueprints of Trust" arXiv:2509.20394）——覆盖端到端 AI 系统的文档，包括安全能力、提示注入防护、数据外泄检测、与人类价值观的对齐。

**类型：** Build
**语言：** Python（stdlib，含 model-card + datasheet + system-card 生成器）
**前置条件：** Phase 18 · 18（安全框架），Phase 18 · 24（监管合规）
**时间：** 约 60 分钟

## 学习目标

- 描述 Mitchell 等 2019 的 Model Card 与 Gebru 等 2018 的 Datasheet。
- 描述 Data Card 的 telescopic / periscopic / microscopic 分层机制。
- 描述 System Card 及其端到端覆盖范围。
- 列出三项 2024–2025 年的发展（自动化生成、可验证声明、可持续性报告）。

## 问题背景

监管框架（第 24 课）和实验室安全策略（第 18 课）均要求文档化。文档格式从模型专用（model cards）演进到数据集专用（datasheets）再到系统专用（system cards）。每种格式对应不同范围的透明度。2024–2025 年的自动化与可验证声明工作旨在解决长期存在的采纳难题。

## 概念讲解

### Model Card（Mitchell 等，2019）

章节包括：
- 模型详情
- 预期用途
- 影响因素（评估相关的人口统计或环境变量）
- 指标
- 评估数据
- 训练数据
- 定量分析（按影响因素分层）
- 伦理考量
- 注意事项与建议

采纳难题：Oreamuno 等 2023 对 Hugging Face model cards 的审计发现，仅 0.3% 记录了伦理考量。

### Datasheets for Datasets（Gebru 等，2018）

类比电子数据手册。章节包括：
- 动机（为何创建该数据集）
- 构成（其中包含什么）
- 收集流程（如何组装）
- 标注（如适用）
- 用途（预期用途、禁用用途、风险）
- 分发
- 维护

发表于 CACM 2021。Datasheet 是上游文档；model card 的正确性依赖于 datasheet 的准确性。

### Data Card（Pushkarna 等，Google 2022）

模块化分层细节。三个缩放层级：
- **Telescopic（望远级）**：面向非专家的高层摘要。
- **Periscopic（潜望级）**：面向 ML 从业者中级概览。
- **Microscopic（显微级）**：面向审计者的特征级详细文档。

边界对象视角：不同读者从同一文档中提取不同信息。

### System Card

范围：端到端 AI 系统，包括模型 + 安全栈 + 部署上下文。章节通常包括：
- 安全能力
- 提示注入防护
- 数据外泄检测
- 与既定人类价值观的对齐
- 事件响应

出自 Sidhpurwala 2024 及 Meta 系统级透明度工作。"Blueprints of Trust"（arXiv:2509.20394）将 System Card 形式化为 Model Card 的部署层补充。

### 2024–2025 年发展

- **CardGen（Liu 等，2024）**：通过 LLM 自动生成 model card；在 Mitchell 2019 标准字段上报告了比许多人工撰写卡片更高的客观性。
- **下载量相关性（Liang 等，2024）**：详细的 model card 与 HF 上高达 29% 的下载量提升相关——采纳动力现已由市场驱动，而不仅限于合规驱动。
- **Laminator（Duddu 等，2024）**：通过硬件 TEE / 密码学签名实现可验证声明——使 model card 携带"声明的证明"而不仅是声明本身。
- **可持续性（Jouneaux 等，2025 年 7 月）**：新增碳、水、算力能耗足迹字段；新兴 ISO 标准。
- **监管卡片**。欧盟《人工智能法案》（第 24 课）GPAI 行为准则透明度章节要求 model card 作为合规产物。

### 在 Phase 18 中的位置

第 24–25 课为监管与 CVE 层；第 26 课为文档层；第 27 课为训练数据治理，是 datasheet 的上游；第 28 课是产生卡片中所引用评估的研究生态。

```figure
an-card-scopes
```

## 实践操作

`code/main.py` 为一个玩具部署生成最小的 model card、datasheet 和 system card。每份卡片均遵循标准章节结构。你可以查看格式并比较三者的覆盖范围。

## 成果交付

本课产出 `outputs/skill-card-audit.md`。给定一份 model card、datasheet 或 system card，它会审计章节覆盖率、定量分层数据，以及是否存在可验证声明。

## 练习

1. 运行 `code/main.py`。检查生成的卡片，识别薄弱章节（仅占位符），并说明哪些证据可以加强它们。

2. 在第 20 课基础上，为 model card 补充跨两个群体的人口统计定量分层分析。

3. 阅读 Oreamuno 等 2023 关于 0.3% 采纳率的研究。提出一项能使 model card 规范中伦理考量采纳率提升的结构化修改方案。

4. Laminator（Duddu 等，2024）使用 TEE 实现可验证声明。设计一个携带评估结果密码学证明的 model card 字段，并描述验证者的角色。

5. 为过往项目或假设部署撰写一份 System Card（注意是 System Card，而非 Model Card）。指出对第三方审计者最有价值的章节。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|------------|----------|
| Model Card | "Mitchell 卡片" | Mitchell 等 2019 的 ML 模型标准文档 |
| Datasheet | "Gebru 数据表" | Gebru 等 2018 的数据集标准文档 |
| Data Card | "Pushkarna 卡片" | Google 2022 的分层模块化数据文档 |
| System Card | "部署卡片" | 含安全栈的端到端 AI 系统文档 |
| Boundary object | "一份文档，不同读者" | Data Card 视角：同一文档服务多元受众 |
| Verifiable attestation | "Laminator 声明" | 附加于文档声明的密码学或 TEE 证明 |
| Sustainability field | "碳/水足迹" | 2025 年新出现的的环境核算字段 |

## 延伸阅读

- [Mitchell 等 — Model Cards for Model Reporting（arXiv:1810.03993，FAT* 2019）](https://arxiv.org/abs/1810.03993) —— 标准 model card 文献
- [Gebru 等 — Datasheets for Datasets（CACM 2021，arXiv:1803.09010）](https://arxiv.org/abs/1803.09010) —— datasheet 论文
- [Pushkarna 等 — Data Cards（Google 2022）](https://arxiv.org/abs/2204.01075) —— 分层数据文档
- [Sidhpurwala 等 — Blueprints of Trust（arXiv:2509.20394）](https://arxiv.org/abs/2509.20394) —— System Card 形式化
