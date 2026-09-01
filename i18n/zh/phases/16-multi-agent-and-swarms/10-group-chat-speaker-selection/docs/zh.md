# 群聊与发言者选择

> 共享对话编排将 N 个 Agent 放在一个对话中；由一个选择函数（LLM、轮询或自定义）来决定下一个发言者。这是涌现式多 Agent 对话的典型原型——Agent 不知道自己在静态图中的角色，它们只是对共享池做出反应。AutoGen 的 GroupChat 和 AG2 的 GroupChat 是参考实现：AutoGen v0.2 的 GroupChat 语义被保留在 AG2 分支中；AutoGen v0.4 则将其重写为事件驱动 Actor 模型。Microsoft 于 2026 年 2 月将 AutoGen 放入维护模式，并将其合并到 Microsoft Agent Framework（2026 年 2 月 RC 版）。GroupChat 原语在 AG2 和 Microsoft Agent Framework 中都得以保留——学一次，到处可用。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置条件：** Phase 16 · 04（Primitive Model）
**时间：** 约 60 分钟

## 问题

静态图（LangGraph）在workflow已知的情况下非常好用。但真实对话不是静态的：有时是编码者问审查者，有时是研究者，有时是写作者。硬编码所有可能的交接会导致边爆炸。你想要的是 *对共享池做出反应的 Agent*，由某个函数决定谁接下来发言。

这正是 AutoGen GroupChat 所做的。

## 概念

### 结构

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

每个 Agent 都能看到所有消息。每一轮都会调用选择函数来决定谁接下来发言。

### 三种选择器风格

**轮询。** 固定循环。确定性。规模随 N 线性增长，但忽略上下文——即使主题是法律审查，编码者仍会获得发言机会。

**LLM 选择。** 调用 LLM，读取最近的共享池，返回最佳下一个发言者。感知上下文但较慢：每一轮都增加一次 LLM 调用。这是 AutoGen 的默认方式。

**自定义。** 带有任意逻辑的 Python 函数。典型用法：LLM 选择 + 回退规则（例如，"编码者发言后，总是让验证者获得发言权"）。

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 持有选择器。当一个 Agent 完成一轮发言后，管理程序调用选择器，选择器返回下一个 Agent。循环持续进行，直到满足终止条件。

### 终止条件

三种常见模式：

- **最大轮数。** 对总轮数设置硬性上限。
- **"TERMINATE" 标记。** Agent 可以发出哨兵消息；当出现此类消息时，管理程序停止。
- **目标达成检查。** 每个回合运行一个轻量级验证器，完成后停止对话。

### 传承：分支与合并

2025 年初，Microsoft 开始对 AutoGen（v0.4）进行重大重写，围绕事件驱动 Actor 模型。社区将 AutoGen v0.2 的 GroupChat 语义 fork 为 AG2，保留了早期采用者已集成的 API。

2026 年 2 月，Microsoft 宣布 AutoGen 将进入维护模式，事件驱动的 Actor 模型将合并到 **Microsoft Agent Framework** 中（2026 年 2 月 RC 版，现已与 Semantic Kernel 合并）。GroupChat 概念在两个方向中都得以保留；实现细节不同。AG2 是 v0.2 兼容代码的首选上游。

### GroupChat 适用场景

- **涌现式对话。** 你不想预先把所有可能的下一个发言者都连线好。
- **角色混合任务。** 编码者询问研究者，研究者询问档案员，档案员再反问编码者。流程不是 DAG。
- **探索式问题解决。** 更像是"头脑风暴会议"，而非"装配线"。

### GroupChat 失效场景

- **严格确定性。** LLM 选择器可能不一致。相同提示词，不同运行，下一个发言者不同。
- **阿谀级联。** Agent 倾向于 defer 到发言最自信的人。需在提示中明确反对此行为。
- **上下文膨胀。** 每个 Agent 都读取每条消息；10 轮后上下文会非常大。使用投影（Lesson 15）来限制视图范围。
- **热发言者。** 一个 Agent 主导了对话，因为选择器偏向其擅长领域。引入发言者平衡作为选择器特性。

### 群聊 vs 监督者

相同的原语，不同的默认设置：

- 监督者：一个 Agent 规划，其他 Agent 执行。选择器逻辑是"询问规划者该做什么"。
- 群聊：所有 Agent 都是对等的；选择器是作用于共享池的函数。

两者都使用 Lesson 04 中的四个原语。群聊默认使用 LLM 选择编排和全池共享状态。

```figure
swarm-speaker
```

## 动手实现

`code/main.py` 用 stdlib 从头实现了一个 GroupChat。三个 Agent（coder、reviewer、manager），轮询和 LLM 选择两种变体，以及基于 `TERMINATE` 标记的终止逻辑。

演示会打印对话转录记录，以及两种变体的选择器决策追踪。

运行：

```
python3 code/main.py
```

## 使用它

`outputs/skill-groupchat-selector.md` 配置了针对特定任务的 GroupChat 选择器——轮询、LLM 选择和自定义，以及应使用哪些选择器输入（最近消息、Agent 专长、发言次数）。

## 交付清单

检查项：

- **最大轮数上限。** 始终设置。典型任务 10-20 轮。
- **发言者平衡指标。** 跟踪每个 Agent 的发言次数；当失衡超过阈值时告警。
- **终止标记。** `TERMINATE` 或专用验证 Agent。
- **投影或受限内存。** 约 10 条消息后，考虑让每个 Agent 只获得受限视图，防止上下文膨胀。
- **选择器日志记录。** 对于 LLM 选择变体，记录选择器的输入和决策。否则无法调试。

## 练习

1. 运行 `code/main.py`。比较轮询与 LLM 选择下的对话。哪种变体下哪个 Agent 占主导？
2. 在选择器中添加"每 Agent 最大发言次数"规则。对话记录会发生什么变化？
3. 实现目标达成终止：当 review 者返回"approved"时停止。在达到轮次上限前，有多少比例会触发？
4. 阅读 AutoGen 稳定版的 GroupChat 文档（https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html）。找出 `GroupChatManager` 使用的默认选择器。
5. 阅读 AG2 仓库（https://github.com/ag2ai/ag2），比较其 v0.2 GroupChat 与 v0.4 事件驱动版本。v0.4 增加了哪个具体特性（吞吐量、容错性、可组合性）？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| GroupChat | "一群 Agent 在一个聊天室里" | 共享消息池 + 选择函数。AutoGen / AG2 原语。 |
| Speaker selection | "下一个谁说话" | 决定下一个 Agent 的函数。轮询、LLM 选择或自定义。 |
| GroupChatManager | "会议主持人" | 持有选择器并循环控制轮次的 AutoGen 组件。 |
| ConversableAgent | "基础 Agent" | AutoGen 基类；能收发消息的 Agent。 |
| Termination token | "'停止'词" | 哨兵字符串（通常是 `TERMINATE`），用于结束对话。 |
| Hot speaker | "一个 Agent 占主导" | 选择器反复选择同一 Agent 的失效模式。 |
| Context bloat | "池无限增长" | 每个 Agent 读取所有先前消息；上下文随轮次增长。 |
| Projection | "受限视图" | 角色特定的共享池视图，用于防止上下文膨胀。 |

## 延伸阅读

- [AutoGen group chat 文档](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) —— 参考实现
- [AG2 仓库](https://github.com/ag2ai/ag2) —— 社区 AutoGen v0.2 延续
- [Microsoft Agent Framework 文档](https://learn.microsoft.com/en-us/agent-framework/) —— 合并后的继任者，2026 年 2 月 RC
- [AutoGen v0.4 发布说明](https://microsoft.github.io/autogen/stable/) —— 事件驱动 Actor 模型重写详情
