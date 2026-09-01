# Reflexion：语言强化学习

> 基于梯度的强化学习需要数千次试验和 GPU 集群才能修复一种失败模式。Reflexion（Shinn 等人，NeurIPS 2023）用自然语言做到这一点：每次试验失败后，智能体写下反思，存入情景记忆，并在下一次试验中基于该记忆进行条件生成。这是 Letta 的睡眠计算、Claude Code 的 CLAUDE.md 学习记录和 pro-workflow 的 learn-rule 背后的模式。

**类型：** 构建
**语言：** Python（标准库）
**前置知识：** 第 14 阶段 · 01（智能体循环）、第 14 阶段 · 02（ReWOO）
**时间：** 约 60 分钟

## 学习目标

- 说出 Reflexion 的三个组成部分（演员、评估器、自我反思器）以及情景记忆的作用。
- 用标准库实现一个带有二元评估器、反思缓冲区和新重试的 Reflexion 循环。
- 针对给定任务选择标量、启发式和自评反馈来源。
- 解释为什么语言强化学习能捕捉到梯度强化学习需要数千次试验才能修复的错误。

## 问题

智能体在一次任务中失败了。在标准 RL 中，你会运行数千次更多试验，计算梯度，更新权重。昂贵、缓慢，而且大多数生产环境中的智能体并没有为每次失败配备训练预算。

Reflexion（Shinn 等人，arXiv:2303.11366）提出了一个不同的问题：如果智能体思考它为什么会失败，然后在提示中加入这个思考再试一次呢？不需要权重更新。不需要梯度。只需在不同试验之间存储自然语言即可。

结果：在 ALFWorld 上，它击败了 ReAct 和其他未微调的基线。在 HotpotQA 上，它优于 ReAct。在代码生成（HumanEval/MBPP）上，它在当时设立了最先进水平。全程无需一次梯度步骤。

## 概念

### 三个组成部分

```
Actor         : 生成轨迹（类 ReAct 循环）
Evaluator     : 评分轨迹——二元、启发式或自评
Self-Reflector: 对失败写出一条自然语言反思
```

再加一个数据结构：

```
Episodic memory: 之前反思的列表，前置到下一次试验的提示中
```

一次试验运行 Actor。Evaluator 对它评分。如果分数低，Self-Reflector 生成一条反思（"我选错了工具，因为我把问题误读为在问 X，而它实际上是在问 Y"）。反思进入情景记忆。下一次试验从头开始，但能看到那条反思。

### 三种评估器类型

1. **标量（Scalar）**——外部二元信号。ALFWorld 成功或失败。HumanEval 测试通过或不通过。最简单，信息量最高。
2. **启发式（Heuristic）**——预定义的失败特征。"如果智能体连续产生相同的动作两次，标记为卡住。""如果轨迹超过 50 步，标记为低效。"
3. **自评（Self-evaluated）**——LLM 对自己轨迹评分。在没有 ground truth 时需要。信号较弱；与工具落地验证配合良好（第 05 课 — CRITIC）。

2026 年的默认方案是混合使用：有 ground truth 时用标量，没有时用自评，用启发式作为安全护栏。

### 为什么它能泛化

Reflexion 与其说是一种新算法，不如说是一个命名的模式。几乎每个生产环境中的"自修复"智能体都在运行某种变体：

- Letta 的睡眠计算（第 08 课）：一个独立智能体反思过去的对话并写入记忆块。
- Claude Code 的 `CLAUDE.md` / "保存记忆"模式：将反思捕获为学习记录，前置到未来会话中。
- pro-workflow 的 `/learn-rule` 命令：将修正捕获为显式规则。
- LangGraph 的反思节点：一个对输出评分并在需要时路由至改进的节点。

它们都源于同一个洞察：自然语言是一种足够丰富的媒介，可以在不同运行之间传递"我从失败中学到了什么"。

### 何时有效，何时无效

Reflexion 有效的场景：

- 存在清晰的失败信号（测试失败、工具错误、答案错误）。
- 任务类型可复现（同一类问题可以再次被提出）。
- 反思有改进空间（有足够的动作预算）。

Reflexion 无帮助的场景：

- 智能体第一次就成功了。
- 失败来自外部（网络中断、工具损坏）——反思"网络中断了"对未来运行无益。
- 反思变成了迷信——存储一次偶发不稳定运行的叙述。

2026 年的陷阱：记忆腐烂。反思不断积累；有些已过时或错误；随着情景缓冲区增长，重试变得越来越慢。缓解措施：定期压缩（第 06 课）、对反思设置 TTL，或用一个独立的睡眠清理智能体（Letta）。

```figure
react-trace
```

## 构建它

`code/main.py` 在一个玩具谜题上实现了 Reflexion：生成一个元素之和等于目标值的 3 元素列表。Actor 输出候选列表；Evaluator 检查总和；Self-Reflector 写一行关于哪里出错了的内容。反思进入情景记忆供下一次试验使用。

组件：

- `Actor`——一个脚本化策略，在看到反思时能自我改进。
- `Evaluator.binary()`——对目标总和做通过/失败判断。
- `SelfReflector`——生成一行对失败的诊断。
- `EpisodicMemory`——具有 TTL 语义的有界列表。

运行它：

```
python3 code/main.py
```

追踪显示三次试验。试验 1 失败，一条反思被存入，试验 2 看到反思后改进但仍失败，试验 3 成功。与基线运行（无反思）对比——它一直卡在试验 1 的答案。

## 使用它

LangGraph 以节点模式提供反射。Claude Code 的 `/memory` 命令和 pro-workflow 的 `/learn-rule` 将情景缓冲区外化为一个 markdown 文件。Letta 的睡眠计算在空闲时间运行 Self-Reflector，使主智能体保持低延迟。OpenAI Agents SDK 不直接提供 Reflexion；你需要用一个自定义 Guardrail 来实现它，根据分数拒绝轨迹，并用跨运行存活的记忆 `Session`。

## 交付

`outputs/skill-reflexion-buffer.md` 创建并维护一个具有反思捕获、TTL 和去重的场景缓冲区。给定一个任务类和一次失败，它输出一条真正对下一次试验有帮助的反思（而不是通用的"更小心"）。

## 练习

1. 从二元评估器切换到返回距离度量（距目标值多远）的标量评估器。收敛更快吗？
2. 给反思添加 10 次试验的 TTL。超过那个点之后，旧反思是帮倒忙还是有帮助？
3. 实现启发式评估器：如果相同动作重复出现则标记为卡住。这与 Self-Reflector 如何交互？
4. 用一个无视反思的对手 Actor 运行 Reflexion。迫使 Actor 注意到反思的最小反思提示工程是什么？
5. 阅读 Reflexion 论文的第 4 节关于 AlfWorld 的内容。从概念上复现 130% 成功率提升：与原始 ReAct 的关键差异是什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Reflexion | "自我修正" | Shinn 等人 2023 —— Actor、Evaluator、Self-Reflector 加上情景记忆 |
| Verbal reinforcement | "无梯度的学习" | 将自然语言反思前置到下一次试验的提示中 |
| Episodic memory | "按任务的反思" | 针对一类任务的有界缓冲反思列表 |
| Scalar evaluator | "二元成功信号" | 来自 ground truth 的通过/失败或数值评分 |
| Heuristic evaluator | "基于模式的检测器" | 预定义的失败特征（如卡住循环、步数过多） |
| Self-evaluator | "LLM 自评自身轨迹" | 无 ground truth 时的弱信号后备——与工具落地验证配合使用 |
| Memory rot | "过期的反思" | 情景缓冲区充满过时条目；通过压缩/TTL 修复 |
| Sleep-time reflection | "异步自我反思" | 在非主路径上运行 Self-Reflector，使主智能体保持快速 |

## 延伸阅读

- [Shinn 等人，Reflexion：具有语言强化学习的智能体（arXiv:2303.11366）](https://arxiv.org/abs/2303.11366)——经典论文
- [Letta，睡眠计算](https://www.letta.com/blog/sleep-time-compute)——生产环境中的异步反思
- [Anthropic，AI 智能体的有效上下文工程](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)——将情景缓冲区作为上下文的一部分进行管理
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview)——反思节点模式
