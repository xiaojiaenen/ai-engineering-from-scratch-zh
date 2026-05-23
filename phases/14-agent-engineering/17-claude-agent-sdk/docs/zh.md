# Claude Agent SDK：子代理与会话存储

> Claude Agent SDK 是 Claude Code 工具链的库形式。内置工具、用于上下文隔离的子代理、钩子、W3C 跟踪传播、会话存储一致性。Claude Managed Agents 是针对长时间异步任务的托管替代方案。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**先决条件：** 第 14 阶段 · 01（代理循环）、第 14 阶段 · 10（技能库）
**时间：** 约 75 分钟

## 学习目标

- 解释 Anthropic 客户端 SDK（原始 API）和 Claude Agent SDK（工具链形态）之间的区别。
- 描述子代理 —— 并行化和上下文隔离 —— 以及何时使用它们。
- 说出 Python SDK 的会话存储接口（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）以及 `--session-mirror` 的作用。
- 实现一个标准库工具链，包含内置工具、具有隔离上下文的子代理生成、生命周期钩子和会话存储。

## 问题所在

原始 LLM API 只能完成一次请求-响应往返。生产环境中的代理需要工具执行、MCP 服务器、生命周期钩子、子代理生成、会话持久化、跟踪传播。Claude Agent SDK 将此形态作为库提供 —— 与 Claude Code 使用的相同工具链，为自定义代理而开放。

## 核心概念

### 客户端 SDK 与代理 SDK

- **客户端 SDK（`anthropic`）。** 原始消息 API。你掌控循环、工具和状态。
- **代理 SDK（`claude-agent-sdk`）。** 内置工具执行、MCP 连接、钩子、子代理生成、会话存储。作为库的 Claude Code 循环。

### 内置工具

SDK 开箱即提供 10 多个工具：文件读写、shell、grep、glob、网络获取等。自定义工具通过标准工具模式接口注册。

### 子代理

Anthropic 记录了两个用途：

1.  **并行化。** 并发运行独立任务。"为这 20 个模块中的每一个查找测试文件" 是 20 个并行子代理任务。
2.  **上下文隔离。** 子代理使用自己的上下文窗口；只有结果返回给编排器。编排器的预算得到保留。

Python SDK 近期新增：`list_subagents()`、`get_subagent_messages()`，用于读取子代理转录记录。

### 会话存储

与 TypeScript 协议一致：

- `append(session_id, message)` —— 添加一轮对话。
- `load(session_id)` —— 恢复会话。
- `list_sessions()` —— 枚举。
- `delete(session_id)` —— 级联至子代理会话。
- `list_subkeys(session_id)` —— 列出子代理键。

`--session-mirror`（CLI 标志）在流式传输时将转录记录镜像到外部文件，用于调试。

### 钩子

可注册的生命周期钩子：

- `PreToolUse`、`PostToolUse` —— 门控或审计工具调用。
- `SessionStart`、`SessionEnd` —— 设置和拆卸。
- `UserPromptSubmit` —— 在模型看到用户输入前对其执行操作。
- `PreCompact` —— 在上下文压缩前运行。
- `Stop` —— 代理退出时清理。
- `Notification` —— 旁路告警。

钩子是专业工作流（第 14 阶段课程参考）及类似系统添加横切关注点行为的方式。

### W3C 跟踪上下文

调用者上活跃的 OTel span 通过 W3C 跟踪上下文头传播到 CLI 子进程。整个多进程跟踪在你的后端中显示为一个跟踪。

### Claude Managed Agents

托管替代方案（测试版头信息 `managed-agents-2026-04-01`）。长时间异步工作、内置提示缓存、内置压缩。用控制权换取托管基础设施。

### 此模式可能出错之处

- **子代理过度生成。** 为 100 个微小任务生成 100 个子代理。开销占主导。应进行批处理。
- **钩子膨胀。** 每个团队都添加钩子；启动时间激增。应每季度审查钩子。
- **会话臃肿。** 会话累积；大小增长。使用 `list_sessions` + 过期策略。

## 构建它

`code/main.py` 在标准库中实现了 SDK 形态：

- `Tool`、`ToolRegistry`，带有内置 `read_file`、`write_file`、`list_dir`。
- `Subagent` —— 私有上下文、隔离运行、返回结果。
- `SessionStore` —— 追加、加载、列出、删除、列出子键。
- `Hooks` —— `pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- 一个演示：主编排器并行生成 3 个子代理（各自隔离），聚合结果，持久化会话。

运行它：

```
python3 code/main.py
```

跟踪显示了子代理上下文隔离（编排器上下文大小保持有界）、钩子执行和会话持久化。

## 使用它

- **Claude Agent SDK**：用于希望采用 Claude Code 工具链形态的 Claude 优先产品。
- **Claude Managed Agents**：用于托管的长时间异步工作。
- **OpenAI Agents SDK**（第 16 课）：用于 OpenAI 优先的对应方案。
- **LangGraph + 自定义工具**：如果你更想要图形态的状态机。

## 交付它

`outputs/skill-claude-agent-scaffold.md` 搭建一个包含子代理、钩子、会话存储、MCP 服务器附加和 W3C 跟踪传播的 Claude Agent SDK 应用程序。

## 练习

1.  添加一个子代理生成器，将 20 个任务批处理为每组 5 个并行子代理。测量编排器上下文大小与每个任务一个子代理的对比。
2.  实现一个 `PreToolUse` 钩子，对 `write_file` 调用进行速率限制（每分钟每个会话 5 次）。跟踪其行为。
3.  连接 `list_subkeys` 以渲染子代理树。深层嵌套是什么样子？
4.  将演示移植到真正的 `claude-agent-sdk` Python 包。工具注册有何变化？
5.  阅读 Claude Managed Agents 文档。何时应从自托管切换到托管？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Agent SDK | “作为库的 Claude Code” | 工具链形态：工具、MCP、钩子、子代理、会话存储 |
| Subagent | “子代理” | 独立的上下文、自身的预算；结果向上冒泡 |
| Session store | “对话数据库” | 持久化、加载、列出、删除轮次，支持子代理级联 |
| Hook | “生命周期回调” | 工具前/后、会话、提示提交、压缩、停止 |
| W3C trace context | “跨进程跟踪” | 父 span 传播到 CLI 子进程 |
| Managed Agents | “托管工具链” | Anthropic 托管的长时间异步工作 |
| `--session-mirror` | “转录镜像” | 在流式传输时将会话轮次写入外部文件 |
| MCP server | “工具接口” | 附加到代理的外部工具/资源源 |

## 扩展阅读

- [Claude Agent SDK 概述](https://platform.claude.com/docs/en/agent-sdk/overview) —— Claude Code 的库形式
- [Anthropic, 使用 Claude Agent SDK 构建代理](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) —— 生产模式
- [Claude Managed Agents 概述](https://platform.claude.com/docs/en/managed-agents/overview) —— 托管替代方案
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) —— 对应方案