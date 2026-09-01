# 毕业设计17 — 个人AI导师（自适应、多模态、带记忆）

> Khanmigo（可汗学院）、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat 和 Synthesis Tutor 都在2026年规模化推出了自适应多模态辅导产品。其共同形态为：苏格拉底式政策（绝不直接抛答案）、每次交互后更新的 learner model（贝叶斯知识追踪风格）、语音+文本+拍照输入、课程图检索、间隔重复调度，以及面向适龄内容的严格安全过滤。毕业设计的任务是交付一个特定学科的导师（K-12代数或入门Python），开展为期两周的10人功效研究，并通过内容安全审计。

**类型：** 毕业设计
**语言：** Python（后端、learner model）、TypeScript（Web应用）、SQL（通过 Postgres + Neo4j 实现课程图）
**先修要求：** Phase 5（NLP）、Phase 6（语音）、Phase 11（LLM工程）、Phase 12（多模态）、Phase 14（Agent）、Phase 17（基础设施）、Phase 18（安全）
**涉及的Phase：** P5 · P6 · P11 · P12 · P14 · P17 · P18
**时间：** 30小时

## 问题背景

自适应辅导曾经是教育科技的研究 niche，到2026年已成为消费级产品。Khanmigo 已部署至美国大多数学区。Duolingo Max 达到数千万MAU。Google 的 LearnLM / Gemini for Education 支撑 Google Classroom 中的辅导功能。Quizlet Q-Chat 与抽认卡并肩。Synthesis Tutor 因"给好奇孩子的导师"概念而走红。这些产品的共同要素包括：多模态输入（打字、语音、拍摄方程）、苏格拉底教学法（先提问再解释）、每次交互后更新的 learner model，以及严格的适龄安全机制。

你将为特定群体构建其中一个产品。衡量标准是一场真实的功效研究：为期两周、10名学习者的前后测分数。语音交互需自然流畅（参考毕业设计03的子栈）。记忆模块须尊重隐私。安全过滤器须通过面向 K-12 的 COPPA 合规红队测试。

## 概念设计

四个核心组件。**Tutor policy（导师策略）** 采用苏格拉底循环：当学习者询问答案时，策略会提出引导性问题；答对时推进到下一个概念；卡住时提供分步脚手架提示。**Learner model（学习者模型）** 采用贝叶斯知识追踪（或简化变体），每次交互后更新每个课程节点的平均掌握概率。**Curriculum graph（课程图）** 是基于 Neo4j 的概念图谱，包含先修关系边；策略沿图遍历以选择下一个概念。**Memory（记忆）** 是 episodic + semantic 存储（类似 agentmemory 风格），记录历史交互、错误和学习偏好。

UX 支持多模态：文本输入用于键入答案；语音输入基于 LiveKit + Whisper（复用毕业设计03）；数学题拍照输入基于 dots.ocr 或 PaliGemma 2；语音输出基于 Cartesia Sonic-2。安全方面使用 Llama Guard 4 加上适龄过滤器（拦截成人内容、暴力、自残相关内容），并配合 COPPA 合规的记忆保留策略。

功效研究是最终交付物：10名学习者，前后测，周期两周。报告学习增益 delta 及置信区间，并与非自适应对照组（相同内容线性呈现、无导师策略）进行比较。

## 架构

```
学习者设备
  |
  +-- 文本         -> Web应用
  +-- 语音         -> LiveKit Agents（ASR + TTS）
  +-- 拍照数学     -> dots.ocr / PaliGemma 2
       |
       v
  导师策略（LangGraph）
       - 苏格拉底决策头
       - 下一概念选择器（课程图遍历）
       - 脚手架提示生成
       - 掌握度更新
       |
       v
  学习者模型（BKT / 项目反应理论）
       - 按概念掌握的预估概率
       - 间隔重复调度器（SM-2 或 FSRS）
       |
       v
  记忆（agentmemory 风格）
       - 情景记忆：每次交互
       - 语义记忆：已学习的错误、偏好
       - 保留策略：COPPA / GDPR 合规
       |
       v
  课程图（Neo4j）
       - 先修关系边
       - 挂载 OER 内容
       |
       v
  安全过滤：
    Llama Guard 4 + 适龄过滤器
    记忆访问按学习者ID作用域限制
```

## 技术栈

- 学科选择：K-12代数 或 入门Python（选择其一深耕）
- 导师策略：基于 Claude Sonnet 4.7（支持prompt缓存）的 LangGraph
- 学习者模型：经典贝叶斯知识追踪（BKT）或 FSRS 间隔调度
- 课程图：Neo4j 概念节点 + 先修边 + OER 内容
- 记忆：agentmemory 风格的持久化向量 + 情景 + 语义存储
- 语音：LiveKit Agents 1.0 + Cartesia Sonic-2（复用毕业设计03子栈）
- 拍照数学：dots.ocr 或 PaliGemma 2 方程识别
- 安全：Llama Guard 4 + 自定义适龄过滤器
- 评测：Bloom层级题目生成、前后测框架、功效研究工具链

```figure
cf-tutor-loop
```

## 构建步骤

1. **课程图。** 构建一个 Neo4j 图谱，包含 50-150 个概念节点（如 K-12代数从"数轴"到"求根公式"），附带先修关系边。为每个节点挂载 OER 内容（Open Textbook、OpenStax）。

2. **学习者模型。** 初始化贝叶斯知识追踪，设置先验参数：猜测率（guess）、失误率（slip）、学习率（learn-rate）。每次交互后更新按概念维度的掌握度。按学习者持久化存储。

3. **导师策略。** 基于 LangGraph 实现以下节点：`read_signal`（判断学习者答案是否正确/部分正确/卡住）、`select_concept`（遍历课程图选择最高优先级概念）、`scaffold`（生成苏格拉底式提问）、`update_mastery`（更新掌握度）。

4. **记忆。** 每次交互写入情景存储。错误与偏好升级至语义记忆。COPPA 合规保留策略：1年后自动删除，家长可访问。

5. **语音通路。** LiveKit Agents worker 接入导师策略。ASR 使用 Whisper-v3-turbo，TTS 使用 Cartesia Sonic-2。支持打断（复用毕业设计03机制）。

6. **拍照数学通路。** 上传或拍摄图片；运行 dots.ocr 或 PaliGemma 2 识别方程；将结构化输入传递给导师策略。

7. **安全。** 所有模型输出须通过 Llama Guard 4 + 适龄过滤器（拦截自残、成人内容、暴力）。记忆访问按学习者ID作用域隔离；提供家长端删除接口。

8. **功效研究。** 10名学习者，前测（标准化30题基线），两周导师交互（每周3次），后测。与另一组10名学习者的非自适应对照组（相同内容）进行比较。

9. **周报。** 为每位学习者自动生成PDF摘要： explored topics、掌握曲线、下一步建议。

## 使用示例

```
学习者："我不明白为什么 3x + 6 = 12 意味着 x = 2"
[信号]   卡住
[概念]  'isolating variables'（先修：addition-subtraction-equality）
[脚手架] "你会从两边减去什么数来开始？"
学习者："6"
[信号]   正确
[掌握度] addition-subtraction-equality: 0.62 -> 0.77
[概念]  继续 'isolating variables'
[脚手架] "很好。现在 3x / 3 等于多少？"
```

## 交付要求

`outputs/skill-ai-tutor.md` 为最终交付物：一个特定学科的自适应导师，支持多模态输入，包含学习者模型、记忆、安全机制，并附有效功效数据。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 学习增益 delta | 10人两周研究的前后测差值 |
| 20 | 苏格拉底忠实度 | 基于对话转录样本的量表评分 |
| 20 | 多模态UX | 语音+拍照+文本端到端连贯性 |
| 20 | 安全与隐私立场 | Llama Guard 4 通过率 + COPPA 合规保留策略 |
| 15 | 课程广度与图质量 | 概念覆盖率 + 先修关系一致性 |
| **100** | | |

## 练习题

1. 对比带与不带自适应 learner model 的功效研究（随机概念顺序）。报告 delta。预期自适应更优，但幅度是有趣的数据。

2. 增加多模态探测：同一概念题目以文本、语音、照片三种形式呈现。测量学习者是否倾向于用自己偏好的模态更快收敛。

3. 构建家长仪表板：练习主题、掌握曲线、即将学习概念、安全事件（任何护栏命中）。符合 COPPA 规范。

4. 增加语言切换模式：导师接受西班牙语输入并用西班牙语授课。测量 X-Guard 覆盖率。

5. 压力测试记忆隐私：验证学习者可A无法通过语音片段重新注入攻击访问学习者B的数据。记录尝试访问并触发告警。

## 术语表

| 术语 | 通常说法 | 实际含义 |
|------|-----------------|------------------------|
| Socratic policy | "先问不直接给" | 导师提出引导性问题而非直接给出答案 |
| Bayesian knowledge tracing | "BKT" | 经典 learner model 方程，用于计算每概念的掌握概率 |
| FSRS | "Free Spaced Repetition Scheduler" | 2024年间隔重复调度器，优于 SM-2 |
| Curriculum graph | "概念DAG" | 基于 Neo4j 的概念图谱加先修边 |
| Episodic memory | "每次交互日志" | 存储每次交互以便后续检索 |
| Semantic memory | "已学模式存储" | 从情景记忆中提炼的错误与偏好 |
| COPPA | "儿童隐私法" | 美国限制收集13岁以下儿童数据的法律 |

## 延伸阅读

- [Khanmigo（可汗学院）](https://www.khanmigo.ai) — 消费级 K-12 导师参考
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — 语言学习导师参考
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — 托管参考模型
- [Quizlet Q-Chat](https://quizlet.com) — 备选参考
- [Synthesis Tutor](https://www.synthesis.com) — 初创公司参考
- [FSRS算法](https://github.com/open-spaced-repetition/fsrs4anki) — 间隔重复调度器
- [贝叶斯知识追踪](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — learner model 经典文献
- [LiveKit Agents](https://github.com/livekit/agents) — 语音技术栈
