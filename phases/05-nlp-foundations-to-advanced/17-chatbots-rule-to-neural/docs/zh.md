# 聊天机器人 —— 从规则基础到神经网络再到LLM智能体

> ELIZA通过模式匹配应答，DialogFlow映射意图，GPT基于权重回答，Claude运行工具并进行验证。每个时代都解决了上一个时代最显著的失败。

**类型：** 学习
**语言：** Python
**先决知识：** 第5阶段 · 13（问答系统），第5阶段 · 14（信息检索）
**时长：** ~75分钟

## 问题所在

用户说“我想改签机票”。系统必须弄清楚他们想要什么，缺少什么信息，如何获取，以及如何完成操作。接着用户说“等等，如果取消呢？”，系统需要记住上下文，切换任务，并保留状态。

对话对机器学习系统来说很难。输入是开放的。输出必须在多轮对话中保持连贯。系统可能需要作用于世界（改签机票、刷卡扣费）。每一步的错误都对用户可见。

聊天机器人架构已经经历了四个范式，每个新范式的引入都是因为前一个失败得太明显。本课按顺序讲解它们。2026年的生产环境是最后两者的混合体。

## 核心概念

![聊天机器人演进：规则基础 → 检索基础 → 神经网络 → 智能体](../assets/chatbot.svg)

**基于规则（ELIZA，AIML，DialogFlow）。** 人工编写的模式匹配用户输入并生成响应。意图分类器路由到预定义流程。槽填充状态机收集所需信息。在其设计的狭窄范围内运行良好。一旦超出范围则立即失败。仍在安全性要求高的领域（银行认证、机票预订）使用，这些领域不容忍幻觉。

**基于检索。** 一种FAQ式系统。将每一对（话语，响应）编码。运行时，编码用户消息并检索最接近的已存储响应。类似Zendesk经典的“相似文章”功能。比规则更好地处理释义。没有生成，因此没有幻觉。

**神经网络（seq2seq）。** 在对话日志上训练的编码器-解码器。从零开始生成响应。流利但容易产生通用输出（“我不知道”）和事实漂移。从未可靠地保持主题。这是Google、Facebook和Microsoft在2016-2019年所有聊天机器人表现不佳的原因。

**LLM智能体。** 包装在循环中的语言模型，负责规划、调用工具和验证结果。不是带有长提示的聊天机器人。一个智能体循环：规划 → 调用工具 → 观察结果 → 决定下一步。检索优先的接地（RAG）防止其产生幻觉。工具调用使其能够实际执行操作。这是2026年的架构。

这四个范式并非顺序替代。一个2026年的生产聊天机器人会路由经过所有四个：规则基础用于认证和破坏性操作，检索用于FAQ，神经生成用于自然措辞，LLM智能体用于模糊的开放查询。

## 动手构建

### 第1步：基于规则的模式匹配

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
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20行代码实现的ELIZA。反射技巧（“我感到难过” → “你为什么感到难过”）是Weizenbaum 1966年经典的治疗师演示。至今仍有启发意义。

### 第2步：基于检索（FAQ）

此示例代码片段需要 `pip install sentence-transformers`（会引入torch）。本课程可运行的 `code/main.py` 使用标准库的Jaccard相似度代替，因此课程无需外部依赖即可运行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于阈值的拒绝是关键设计决策。如果最佳匹配不够接近，则返回 `None` 并让系统升级处理。

### 第3步：神经生成（基线）

使用小型指令调整的编码器-解码器（FLAN-T5）或微调后的对话模型。在2026年单独使用在生产中不可用（矛盾、偏离主题、事实错误），但作为混合系统的一部分用于自然措辞。DialoGPT风格的纯解码器模型需要显式的轮次分隔符和EOS处理才能生成连贯的回复；FLAN-T5的text2text流水线作为教学示例可直接使用。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 第4步：LLM智能体循环

2026年的生产形态：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

三点需要注意。工具是LLM可以调用的可执行函数。当LLM返回最终答案而非工具调用时，循环终止。步骤预算防止在模糊任务上出现无限循环。

实际生产环境会增加：检索优先接地（在每次LLM调用前注入相关文档）、防护措施（拒绝破坏性操作，除非确认）、可观察性（记录每一步）和评估（自动检查智能体行为是否符合规范）。

### 第5步：混合路由

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式是：破坏性操作使用确定性规则，预置的FAQ使用检索，其他一切使用LLM智能体。这就是2026年客户支持系统上线的内容。

## 实际应用

2026年的技术栈：

| 用例 | 架构 |
|---------|---------------|
| 预订、支付、认证 | 基于规则的状态机 + 槽填充 |
| 客户支持FAQ | 在精选答案上进行检索 |
| 开放式帮助聊天 | 带有RAG和工具调用的LLM智能体 |
| 内部工具 / IDE助手 | 带有工具调用（搜索、读取、写入）的LLM智能体 |
| 伴侣 / 角色聊天机器人 | 调整过的LLM，带有角色系统提示，基于知识检索 |

在生产环境中始终使用混合路由。没有单一架构能很好地处理所有请求。路由层本身通常是一个小型意图分类器。

## 仍然会出现的故障模式

- **自信地编造。** LLM智能体声称它完成了某个操作，但实际上没有。缓解措施：验证结果，记录工具调用，绝不让LLM声称做了某事，除非有成功的工具返回。
- **提示注入。** 用户插入覆盖系统提示的文本。在2025年OWASP LLM应用十大风险中排名第一。两种类型：直接注入（粘贴到聊天中）和间接注入（隐藏在智能体读取的文档、电子邮件或工具输出中）。
  攻击成功率因场景而异。在通用工具使用和编码基准测试中，前沿模型的实测成功率范围约为0.5-8.5%。特定高风险设置（针对AI编码智能体的自适应攻击、脆弱的编排）成功率可达约84%。生产环境的CVE包括EchoLeak（CVE-2025-32711，CVSS 9.3）—— 这是微软365 Copilot中一个由攻击者控制的电子邮件触发的零点击数据泄露漏洞。
  缓解措施：在整个循环中将用户输入视为不可信；在工具调用前进行清理；将工具输出与主提示隔离；使用“计划-验证-执行”（PVE）模式，智能体先计划，然后在执行前根据该计划验证每个操作（这可以阻止工具结果注入新的计划外操作）；对破坏性操作要求用户确认；对工具作用域应用最小权限原则。
  任何程度的提示工程都无法完全消除此风险。需要外部运行时防御层（LLM Guard、白名单验证、语义异常检测）。
- **范围蔓延。** 智能体因工具调用返回了相关性不大的信息而偏离任务。缓解措施：缩小工具契约；保持系统提示集中；添加离题率评估。
- **无限循环。** 智能体持续调用同一工具。缓解措施：步骤预算、工具调用去重、使用LLM判断“我们是否在取得进展”。
- **上下文窗口耗尽。** 长对话将最早的轮次挤出上下文。缓解措施：总结早期轮次，通过相似性检索相关的过去轮次，或使用长上下文模型。

## 上线部署

保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## 练习

1. **简单。** 使用10个模式实现上述基于规则的响应，用于咖啡店点餐机器人。测试边缘情况：重复下单、修改、取消、意图不明确。
2. **中等。** 构建一个混合FAQ + LLM后备系统。为一个SaaS产品准备50条预置FAQ条目，使用检索文档网站的LLM后备。在100个真实支持问题上测量拒绝率和准确性。
3. **困难。** 使用三个工具（搜索、读取用户数据、发送电子邮件）实现上述智能体循环。使用50个测试场景（包括提示注入尝试）进行评估。报告离题率、任务失败率和任何注入成功情况。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 意图 (Intent) | 用户想要什么 | 分类标签（book_flight, reset_password）。路由到处理器。 |
| 槽 (Slot) | 一条信息 | 机器人需要的参数（日期、目的地）。槽填充是一系列询问。 |
| RAG | 检索加生成 | 检索相关文档，然后为LLM的响应提供依据。 |
| 工具调用 (Tool call) | 函数调用 | LLM发出带有名称和参数的结构化调用。运行时执行并返回结果。 |
| 智能体循环 (Agent loop) | 计划、执行、验证 | 运行LLM调用并与工具调用交织，直到任务完成的控制器。 |
| 提示注入 (Prompt injection) | 用户攻击提示 | 试图覆盖系统提示的恶意输入。 |

## 延伸阅读

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 最早的基于规则的聊天机器人论文。
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — 谷歌的晚期神经聊天机器人论文，就在LLM智能体接管之前。
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — 命名了智能体循环模式的论文。
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2024年的生产指南，在2026年仍然适用。
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — 关于提示注入的论文。
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — 使提示注入成为首要安全问题的排名。
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — 实用的编排层防御，包括“计划-验证-执行”和用户确认流程。
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — 间接提示注入导致的零点击数据泄露CVE典型案例。这是为什么具有写入权限的智能体需要运行时防御的参考案例。