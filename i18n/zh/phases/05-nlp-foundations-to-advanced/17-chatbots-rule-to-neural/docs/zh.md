# 聊天机器人 — 从规则到神经到 LLM 智能体

> ELIZA 通过模式匹配回复。DialogFlow 映射意图。GPT 从权重中生成回答。Claude 运行工具并验证。每个时代都解决了前一个时代最致命的缺陷。

**类型：** 学习
**语言：** Python
**前置知识：** 第 5 阶段 · 13（问答）、第 5 阶段 · 14（信息检索）
**时间：** 约 75 分钟

## 问题所在

用户说"我想改航班"。系统需要弄清楚他们想要什么、缺少什么信息、如何获取这些信息，以及如何完成操作。然后用户又说"等等，如果取消呢？"系统必须记住上下文、切换任务并保存状态。

对话对机器学习系统来说很难。输入是开放式的。输出必须在多个回合中保持一致性。系统可能需要在真实世界中采取行动（更改航班、扣款）。用户的每一步错误都清晰可见。

聊天机器人的架构经历了四个范式的循环，每个新范式都是因为前一个范式失败得过于明显而诞生的。本课按顺序介绍它们。2026 年的生产环境是最后两个范式的混合体。

## 概念解析

![聊天机器人演变：规则 → 检索 → 神经 → 智能体](../assets/chatbot.svg)

### 规则主导的半个世纪，1950-2001

第一个范式没有坚持五年。它坚持了五十年。了解它的演进轨迹很重要，因为其中的每个系统都是同一台机器——匹配输入、发出预设回复、更新少量状态——在这台机器上添加规则的五十年从未产生出通用解决方案。这就是第二到第四范式存在的原因。

**1950 年。** 图灵绕开了"机器能否思考？"这个问题，提出了一个操作性替代方案：如果测试者无法通过电话打字机区分机器和人类，那么这个哲学问题就变得无关紧要。对话成为该领域的基准测试，尽管该领域还没有正式命名。

**1956 年。** 这个名称出现了——达特茅斯夏季研讨会构思了"人工智能"这一术语，基于这样的猜想：智能的每个特征"原则上可以被精确描述，从而制造一台机器来模拟它"。该提案预算两个月用于取得实质性进展。

**1966 年。** ELIZA 推出了你在步骤 1 中构建的反射技巧：分解规则从输入中提取片段，重组规则将它们作为问题回声返回。总共约 200 个模式，零状态，零理解——但用户仍然向它倾诉。韦泽鲍姆（Weizenbaum）余生都在为如此少的机械结构就能产生这样的效果而感到震惊。

**1972 年。** PARRY 在斯坦福大学构建，用于模拟偏执，添加了 ELIZA 所缺乏的东西：内部状态。用于恐惧、愤怒和不信任的数值变量在每个回合中更新，并决定哪个脚本下一步触发，因此相同的输入会根据对话历史产生不同的回复。在一项双盲转录测试中，精神病学家区分 PARRY 和人类患者的能力仅处于随机水平。它是人格条件化的直接祖先——三个浮点数实现的系统提示。同年，这两个机器人通过 ARPANET 相互连接：一个治疗师脚本正在采访一个偏执状态机，这是网络上首次机器人之间的对话。

**1995 年。** ALICE 用 AIML（一种用于模式-模板对的 XML 方言）扩展了 ELIZA 的方案。大约 40,000 个人工编写的类别，三次获得Loebner奖。它证明了规则系统的扩展规律：更多规则带来覆盖率，而非通用性。每条规则都是某人必须维护的负担。

**2001 年。** SmarterChild 将该方案带给了 3000 万即时通讯用户，并添加了后端查询——天气、股票、电影时间——拼接到模板中。眯起眼睛看，它就是穿着 2001 年服装的工具调用：解析意图、调用服务、将结果渲染到回复中。

五十年，一种机制，不断上升的规则数量。这个范式之所以终结，不是因为有人推翻了它，而是因为手写的状态机的维护成本随覆盖率线性增长，而用户期望却随着他们上周看到的东西而增长。

```figure
chatbot-lineage
```

**基于规则的（ELIZA、AIML、DialogFlow）。** 手工编写的模式匹配用户输入并产生回复。意图分类器路由到预定义的流程。槽位填充状态机收集所需信息。在它设计的小范围内表现卓越。一旦超出范围立即失败。仍然在不容忍幻觉的安全关键领域（银行身份验证、航空公司预订）中部署。

**基于检索的。** 一种 FAQ 式系统。编码每个（话语，响应）对。运行时，编码用户消息并检索最近的存储响应。想象 Zendesk 经典的"类似文章"功能。比规则更好地处理 paraphrase（改写）。没有生成，所以没有幻觉。

**神经的（seq2seq）。** 在对话日志上训练的编码器 - 解码器。从头生成回复。流利但容易陷入泛泛的输出（"我不知道"）和事实漂移。永远无法可靠地保持在主题上。这就是为什么 Google、Facebook 和 Microsoft 在 2016-2019 年都拥有令人失望的聊天机器人的原因。

**LLM 智能体。** 一个包裹在循环中的语言模型，该循环进行规划、调用工具并验证结果。不是带有长提示的聊天机器人。是一个智能体循环：规划 → 调用工具 → 观察结果 → 决定下一步。基于检索的 grounding（RAG）防止其产生幻觉。工具调用让它真正能够做事。这是 2026 年的架构。

这四种范式不是顺序替换的。2026 年的生产聊天机器人会依次经过所有四种：基于规则用于身份验证和破坏性操作，基于检索用于 FAQ，神经生成用于自然措辞，LLM 智能体用于模糊的开放式查询。

## 构建它

### 步骤 1：基于规则的 패턴 매칭

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    # 遍历所有规则模式
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

用 20 行代码实现的 ELIZA。反射技巧（"我感觉悲伤" → "你为什么感觉悲伤？"）来自 Weizenbaum 1966 年的经典心理治疗师演示。仍然具有教学价值。

### 步骤 2：基于检索的（FAQ）

这个示例代码片段需要 `pip install sentence-transformers`（会拉取 torch）。本课程的运行版 `code/main.py` 使用 stdlib Jaccard 相似度，因此课程无需外部依赖即可运行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("如何重置密码", "前往设置 > 安全 > 重置密码。"),
    ("如何取消订单", "前往订单，找到订单，点击取消。"),
    ("退货政策是什么", "未使用商品 30 天内退货，需原始包装。"),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    # 如果最佳匹配不够接近，返回 None 让系统升级
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于阈值的拒绝是关键设计选择。如果最佳匹配不够接近，返回 `None` 并让系统升级。

### 步骤 3：神经生成（基线）

使用小型指令微调的编码器 - 解码器（FLAN-T5）或微调的对话模型。在 2026 年单独用于生产不可用（矛盾、偏离主题漂移、事实荒谬），但在混合系统中用于自然措辞。类似 DialoGPT 的解码器-only 模型需要明确的回合分隔符和 EOS 处理才能产生连贯的回复；FLAN-T5 text2text 管道开箱即用，适合教学示例。

```python
from transformers import pipeline

# 初始化聊天机器人管道
chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("礼貌回应：你好！", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 步骤 4：LLM 智能体循环

2026 年的生产形态：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            # 验证工具名称
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"错误：未知工具 {tool_name!r}"})
                continue
            # 验证参数类型
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"错误：参数必须是字典，收到 {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "无法在步骤预算内完成任务。"
```

三个要点：工具是 LLM 可调用的函数。循环在 LLM 返回最终答案而不是工具调用时终止。步骤预算防止在模糊任务上无限循环。

真实生产环境增加：基于检索的 grounding（在每次 LLM 调用前注入相关文档）、护栏（未经确认拒绝破坏性操作）、可观测性（记录每个步骤）和评估（自动化检查智能体行为是否保持规范）。

### 步骤 5：混合路由

```python
def hybrid_chat(user_input):
    # 破坏性操作走结构化流程
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    # 先尝试 FAQ 检索
    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    # 否则走智能体循环
    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式：任何破坏性操作使用确定性规则，常见问题解答使用检索，其余使用 LLM 智能体。这就是 2026 年客服系统中部署的方案。

## 使用它

2026 年技术栈：

| 用例 | 架构 |
|---------|---------------|
| 预订、支付、身份验证 | 基于规则的状态机 + 槽位填充 |
| 客户支持 FAQ | 基于 curated 答案的检索 |
| 开放式帮助聊天 | 带 RAG + 工具调用的 LLM 智能体 |
| 内部工具 / IDE 助手 | 带工具调用的 LLM 智能体（搜索、读取、写入） |
| 陪伴 / 角色聊天机器人 | 带人格系统提示的 tuned LLM，知识检索 |

在生产中始终使用混合路由。没有单一架构能很好地处理所有请求。路由层本身通常是一个小型意图分类器。

## 仍在生产中出现的故障模式

- **自信地编造。** LLM 智能体声称完成了它并未执行的操作。缓解措施：验证结果，记录工具调用，不要让 LLM 在没有成功工具返回的情况下声称完成了某事。
- **提示注入。** 用户插入覆盖系统提示的文本。在 2025 年 LLM 应用 OWASP Top 10 中排名第 1。两种类型：直接注入（粘贴到聊天中）和间接注入（隐藏在智能体读取的文档、电子邮件或工具输出中）。

  攻击成功率因场景而异。在通用工具使用和编码基准测试中，前沿模型的成功率约为 0.5-8.5%。特定高风险设置（针对 AI 编码智能体的自适应攻击、脆弱的编排）已达到约 84%。生产 CVE 包括 EchoLeak（CVE-2025-32711，CVSS 9.3）——一个由攻击者控制的电子邮件触发的 Microsoft 365 Copilot 零点击数据外泄漏洞。

  缓解措施：在整个循环中将用户输入视为不受信任；在工具调用前清理；将工具输出与主提示隔离；使用计划 - 验证 - 执行（PVE）模式，其中智能体首先规划，然后在执行前根据该计划验证每个操作（这阻止工具结果注入新的未规划操作）；对破坏性操作要求用户确认；对工具范围应用最小权限。

  无论多少提示工程都无法完全消除此风险。需要外部运行时防御层（LLM Guard、白名单验证、语义异常检测）。
- **范围蔓延。** 智能体偏离任务，因为工具调用返回了边缘相关的信息。缓解措施：缩小工具合同；保持系统提示聚焦；添加偏离任务率的评估。
- **无限循环。** 智能体持续调用同一个工具。缓解措施：步骤预算、工具调用去重、LLM 判断"我们是否在取得进展"。
- **上下文窗口耗尽。** 长对话将最早的回合挤出上下文。缓解措施：摘要较旧的回合、通过相似度检索相关的过去回合，或使用长上下文模型。

## 交付它

保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: 为给定用例设计聊天机器人技术栈。
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

给定产品上下文（用户需求、合规约束、可用工具、数据量），输出：

1. 架构。基于规则、基于检索、神经、LLM 智能体或混合（指定哪些路径去哪）。
2. 如果适用，LLM 选择。命名模型家族（Claude、GPT-4、Llama-3.1、Mixtral）。匹配工具使用质量和成本。
3. 接地策略。RAG 来源、检索方法（见第 14 课）、工具合同。
4. 评估计划。任务成功率、工具调用正确率、偏离任务率、在保留对话上的幻觉率。

拒绝推荐对任何破坏性操作（支付、账户删除、数据修改）使用纯 LLM 智能体，除非有结构化确认流程。拒绝跳过提示注入审计，如果智能体对任何内容具有写入访问权限。
```

## 练习

1. **简单。** 用 10 个模式实现上面的基于规则的响应，用于咖啡店点餐机器人。测试边界情况：重复订单、修改、取消、意图不明确。
2. **中等。** 构建混合 FAQ + LLM 后备。50 个 SaaS 产品的 canned FAQ 条目，带文档站点检索的 LLM 后备。在 100 个真实支持问题上测量拒绝率和准确率。
3. **困难。** 用三个工具（搜索、读取用户数据、发送邮件）实现上面的智能体循环。使用 50 个测试场景运行评估，包括提示注入尝试。报告偏离任务率、任务失败率和任何注入成功。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|--------------------------|---------------------------------------------------|
| 意图 | 用户想要什么 | 分类标签（book_flight, reset_password）。路由到处理器。 |
| 槽位 | 信息片段 | 机器人需要的参数（日期、目的地）。槽位填充是询问序列。 |
| RAG | 检索加生成 | 检索相关文档，然后 grounding LLM 的回复。 |
| 工具调用 | 函数调用 | LLM 发出带名称 + 参数的结构化调用。运行时执行，返回结果。 |
| 智能体循环 | 规划、行动、验证 | 控制器，交错运行 LLM 调用和工具调用直到任务完成。 |
| 提示注入 | 用户攻击提示 | 试图覆盖系统提示的恶意输入。 |

## 延伸阅读

- [图灵（1950）。计算机与智能](https://academic.oup.com/mind/article/LIX/236/433/986238) — 使对话成为该领域基准的论文。
- [韦泽鲍姆（1966）。ELIZA — 用于自然语言通信研究的计算机程序](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 原始基于规则的聊天机器人论文。
- [科尔比、韦伯、希尔夫（1971）。人工偏执](https://doi.org/10.1016/0004-3702(71)90002-6) — PARRY 的情感变量架构，首个有状态的聊天机器人。
- [Thoppilan 等人（2022）。LaMDA：用于对话应用的语言模型](https://arxiv.org/abs/2201.08239) — Google 后期的神经聊天机器人论文，就在 LLM 智能体接管之前。
- [Yao 等人（2022）。ReAct：在语言模型中协同推理和行动](https://arxiv.org/abs/2210.03629) — 命名智能体循环模式的论文。
- [Anthropic 的构建有效智能体指南](https://www.anthropic.com/research/building-effective-agents) — 2024 年的生产指导，在 2026 年仍然适用。
- [Greshake 等人（2023）。不是你注册的东西：通过间接提示注入损害真实世界的 LLM 集成应用](https://arxiv.org/abs/2302.12173) — 提示注入论文。
- [OWASP LLM 应用 Top 10 2025 — LLM01 提示注入](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — 使提示注入成为首要安全关切的排名。
- [AWS — 保护 Amazon Bedrock 智能体免受间接提示注入](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — 包括计划和验证执行以及用户确认流程在内的实用编排层防御。
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — 来自间接提示注入的经典零点击数据外泄 CVE。为什么具有写入访问权限的智能体需要运行时防御的参考案例。
