# 顶石项目 17 — 个人 AI 导师（自适应、多模态、带记忆）

> Khanmigo（可汗学院）、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat 和 Synthesis Tutor 均在 2026 年实现了大规模自适应多模态辅导。其共同形态包括：苏格拉底式策略（绝不直接给出答案）、每次交互后更新的学习者模型（贝叶斯知识追踪风格）、语音+文字+拍照数学输入、课程图谱检索、间隔重复调度，以及严格的年龄适当内容安全过滤器。本顶石项目旨在推出一款特定科目的辅导工具（K-12 代数或 Python 入门），对 10 名学习者进行为期两周的效果研究，并通过内容安全审计。

**类型：** 顶石项目
**语言：** Python（后端、学习者模型）、TypeScript（Web 应用）、SQL（通过 Postgres + Neo4j 构建课程图谱）
**先决条件：** 阶段 5（NLP）、阶段 6（语音）、阶段 11（LLM 工程）、阶段 12（多模态）、阶段 14（智能体）、阶段 17（基础设施）、阶段 18（安全）
**涉及阶段：** P5 · P6 · P11 · P12 · P14 · P17 · P18
**时间：** 30 小时

## 问题

自适应辅导曾是一个教育科技研究的小众领域。到 2026 年，它已成为消费级产品。Khanmigo 在美国大多数学区部署。Duolingo Max 达到了数千万月活跃用户。Google 的 LearnLM / Gemini for Education 为 Google Classroom 提供辅导支持。Quizlet Q-Chat 与闪卡功能并列。Synthesis Tutor 因面向好奇儿童的辅导功能而爆红。共同要素包括：多模态输入（打字、语音、拍照方程）、苏格拉底教学法（先提问，后解释）、每次交互后更新的学习者模型，以及严格的年龄适当安全措施。

你将为特定人群构建这样一款工具。衡量标准是一项实际的效果研究：对 10 名学习者进行两周的学前测试和学后测试，比较分数。语音循环必须自然（复用顶石项目 03 的子栈）。记忆必须尊重隐私。安全过滤器必须通过面向 K-12 的、知晓 COPPA 的红队测试。

## 概念

四个组件。**辅导策略**是苏格拉底式循环：当学习者索要答案时，策略会提出引导性问题；当他们答对时，转向下一个概念；当他们卡住时，提供脚手架式提示。**学习者模型**是贝叶斯知识追踪（或其简单变体），在每次交互后更新每个课程节点的掌握概率。**课程图谱**是概念的 Neo4j 图，包含先决条件边；策略遍历该图以选择下一个概念。**记忆**是情境式+语义式存储（类似 agentmemory 风格），保存过去的交互、错误和偏好。

用户体验是多模态的。文字输入用于键入答案。语音输入通过 LiveKit + Whisper（复用顶石项目 03）。拍照输入用于数学题，通过 dots.ocr 或 PaliGemma 2。语音输出通过 Cartesia Sonic-2。安全方面使用 Llama Guard 4 加上年龄适当过滤器（屏蔽成人内容、暴力、自残），并遵守 COPPA 知晓的记忆保留策略。

效果研究是交付成果。10 名学习者，学前测试和学后测试，为期两周。报告学习增益增量和置信区间。与非自适应基线（相同内容以线性方式交付，无辅导策略）进行比较。

## 架构

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## 技术栈

- 科目选择：K-12 代数或 Python 入门（二选一以深入研究）
- 辅导策略：基于 LangGraph 的 Claude Sonnet 4.7（带提示缓存）
- 学习者模型：贝叶斯知识追踪（经典版）或用于间隔调度的 FSRS
- 课程图谱：Neo4j 中的概念 + 先决条件边 + OER 内容
- 记忆：agentmemory 风格的持久化向量 + 情境式 + 语义式存储
- 语音：LiveKit Agents 1.0 + Cartesia Sonic-2（复用顶石项目 03 子栈）
- 拍照数学：dots.ocr 或 PaliGemma 2 用于方程识别
- 安全：Llama Guard 4 + 自定义年龄适当过滤器
- 评估：布卢姆级别问题生成、前测/后测工具、效果研究工具

## 实现步骤

1.  **课程图谱。** 构建一个包含 50-150 个概念节点的 Neo4j（例如，从“数轴”到“二次公式”的 K-12 代数），带有先决条件边。每个节点关联 OER 内容（Open Textbook, OpenStax）。

2.  **学习者模型。** 初始化贝叶斯知识追踪，设置先验值：猜测、失误、学习率。每次交互后更新每个概念的掌握度。按学习者持久化存储。

3.  **辅导策略。** 使用 LangGraph，节点包括：`read_signal`（学习者的答案正确/部分正确/卡住了？），`select_concept`（遍历课程图谱选择最高优先级的概念），`scaffold`（苏格拉底式提示），`update_mastery`。

4.  **记忆。** 每次交互写入情境存储。错误和偏好提升到语义记忆。遵守 COPPA 的保留策略：1 年后自动删除，父母可访问。

5.  **语音路径。** LiveKit Agents worker 附加到辅导策略。ASR 通过 Whisper-v3-turbo。TTS 通过 Cartesia Sonic-2。支持打断（复用顶石项目 03 机制）。

6.  **拍照数学路径。** 上传或捕获图像；运行 dots.ocr 或 PaliGemma 2 识别方程；作为结构化输入提供给辅导工具。

7.  **安全。** 每个模型输出都通过 Llama Guard 4 和年龄适当过滤器（屏蔽自残、成人内容、暴力）。记忆访问按学习者 ID 范围限定；提供家长访问界面用于删除。

8.  **效果研究。** 10 名学习者，学前测试（标准化 30 题基线），两周辅导交互（每周 3 次），学后测试。与由 10 名学习者组成的、学习相同内容的非自适应基线组进行比较。

9.  **每周进度报告。** 为每个学习者自动生成 PDF 摘要，包括探讨的主题、掌握轨迹和建议的后续步骤。

## 使用方式

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## 交付成果

`outputs/skill-ai-tutor.md` 是交付成果。一个具有多模态输入、学习者模型、记忆、安全和效果测量的特定科目自适应辅导工具。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 学习增益增量 | 在 10 名学习者两周研究中的前测/后测增量 |
| 20 | 苏格拉底式保真度 | 对对话记录样本的评估量表评分 |
| 20 | 多模态用户体验 | 语音 + 拍照 + 文字端到端的连贯性 |
| 20 | 安全与隐私状况 | Llama Guard 4 通过率 + COPPA 知晓的保留策略 |
| 15 | 课程广度和图谱质量 | 概念覆盖度 + 先决条件图谱一致性 |
| **100** | | |

## 练习

1.  在有和没有自适应学习者模型（随机概念顺序）的情况下运行效果研究。报告增量差值。预计自适应模型会获胜，但增量大小才是有趣的数据。

2.  添加多模态探测：以文字、语音和照片形式提出相同的概念问题。测量学习者是否在其偏好的模态下收敛更快。

3.  构建家长仪表板：练习的主题、掌握轨迹、即将学习的概念、安全事件（任何护栏触发）。符合 COPPA 标准。

4.  添加语言切换模式：辅导工具接受西班牙语输入并用西班牙语教学。衡量 X-Guard 覆盖率。

5.  测试记忆隐私：验证学习者 A 无法看到学习者 B 的数据，即使通过语音片段重新摄取攻击。记录尝试的访问并发出警报。

## 关键术语

| 术语 | 人们如何称呼它 | 其实际含义 |
|------|-----------------|------------------------|
| 苏格拉底式策略 | “提问，不倾倒” | 辅导工具提出引导性问题，而不是直接给出答案 |
| 贝叶斯知识追踪 | “BKT” | 用于计算每个概念掌握概率的经典学习者模型方程 |
| FSRS | “自由间隔重复调度器” | 2024 年的间隔重复调度算法，优于 SM-2 |
| 课程图谱 | “概念 DAG” | 包含先决条件边的 Neo4j 概念图 |
| 情境记忆 | “每次交互日志” | 存储的每次交互，供日后检索 |
| 语义记忆 | “学习到的模式存储” | 从情境记忆中提炼提升的错误和偏好 |
| COPPA | “儿童隐私法” | 限制从 13 岁以下儿童收集数据的美国法律 |

## 延伸阅读

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — 参考消费级 K-12 辅导工具
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — 参考语言学习辅导工具
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — 托管参考模型
- [Quizlet Q-Chat](https://quizlet.com) — 替代参考
- [Synthesis Tutor](https://www.synthesis.com) — 初创公司参考
- [FSRS algorithm](https://github.com/open-spaced-repetition/fsrs4anki) — 间隔重复调度算法
- [Bayesian Knowledge Tracing](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — 学习者模型经典算法
- [LiveKit Agents](https://github.com/livekit/agents) — 语音技术栈