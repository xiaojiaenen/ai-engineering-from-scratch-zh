# Harness 作为库 — 子代理与会话存储

> 可导入的 harness：内置工具、用于上下文隔离的子代理、钩子、W3C 链路追踪传播、会话持久化。Claude Agent SDK 是参考示例——即 Claude Code harness 的库形式——而 Claude Managed Agents 则是用于长时间运行异步工作的托管方案。

**类型：** 学习 + 实践
**语言：** Python (stdlib)
**前置知识：** Phase 14 · 01 (Agent Loop), Phase 14 · 10 (Skill Libraries)
**时间：** ~75 分钟

## 学习目标

- 说明 Anthropic Client SDK（原始 API）与 Claude Agent SDK（harness 形态）之间的区别。
- 描述子代理——并行化与上下文隔离——以及何时使用它们。
- 说出 Python SDK 的会话存储接口（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）以及 `--session-mirror` 的作用。
- 实现一个带有内置工具、隔离上下文的子代理启动、生命周期钩子和会话存储的 stdlib harness。

## 问题所在

原始 LLM API 只提供一次往返调用。生产级 agent 需要工具执行、MCP 服务器、生命周期钩子、子代理启动、会话持久化和链路追踪传播。Claude Agent SDK 将这个形态作为库提供——与 Claude Code 相同的 harness，暴露给自定义 agent 使用。

## 概念解析

### Client SDK vs Agent SDK

- **Client SDK (`anthropic`)。** 原始 Messages API。你自己掌控循环、工具和状态。
- **Agent SDK (`claude-agent-sdk`)。** 内置工具执行、MCP 连接、钩子、子代理启动、会话存储。将 Claude Code 循环作为库使用。

### 内置工具

SDK 开箱即用 10+ 个工具：文件读写、shell、grep、glob、web fetch 等。自定义工具通过标准 tool-schema 接口注册。

### 子代理

Anthropic 文档记录了两大用途：

1. **并行化。** 并发运行独立工作。"为这 20 个模块分别找到测试文件"就是 20 个并行子代理任务。
2. **上下文隔离。** 子代理使用自己的上下文窗口；只有结果返回给编排器。编排器的预算得以保留。

Python SDK 近期新增：`list_subagents()`、`get_subagent_messages()` 用于读取子代理记录。

### 会话存储

与 TypeScript 协议对齐：

- `append(session_id, message)` — 添加一轮对话。
- `load(session_id)` — 恢复会话。
- `list_sessions()` — 枚举会话。
- `delete(session_id)` — 级联删除子代理会话。
- `list_subkeys(session_id)` — 列出子代理键。

`--session-mirror`（CLI 标志）在流式传输时将对话记录镜像到外部文件，用于调试。

### 钩子（Hooks）

可注册的生命周期钩子：

- `PreToolUse`、`PostToolUse` — 门禁或审计工具调用。
- `SessionStart`、`SessionEnd` — 设置和清理会话。
- `UserPromptSubmit` — 在模型看到用户输入前进行处理。
- `PreCompact` — 在上下文压缩前运行。
- `Stop` — agent 退出时清理。
- `Notification` — 旁路通知。

钩子是 pro-workflow（Phase 14 课程参考）及类似系统添加横切关注点行为的方式。

### W3C 链路上下文

调用方活跃的 OTel span 通过 W3C trace context 头传播到 CLI 子进程。整个多进程链路在您的后端中显示为一条完整 trace。

### Claude Managed Agents

托管方案（beta 头 `managed-agents-2026-04-01`）。支持长时间运行的异步工作，内置提示缓存、内置上下文压缩。用控制力换取托管基础设施。

### 此模式常见误区

- **子代理滥用。** 为 100 个微小任务启动 100 个子代理。开销占主导。请改为批处理。
- **钩子膨胀。** 每个团队都添加钩子；启动时间暴增。建议每季度审查钩子。
- **会话膨胀。** 会话不断累积；体积增长。使用 `list_sessions` + 过期策略。

```figure
ae-subagent-isolation
```

## 动手实现

`code/main.py` 以 stdlib 实现了 SDK 形态：

- `Tool`、`ToolRegistry`，内置 `read_file`、`write_file`、`list_dir`。
- `Subagent`——私有上下文、隔离运行、结果返回。
- `SessionStore`——append、load、list、delete、list_subkeys。
- `Hooks`——`pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- 演示：主代理并行启动 3 个子代理（各自隔离），聚合结果，持久化会话。

运行：

```
python3 code/main.py
```

Trace 显示子代理上下文隔离（编排器上下文大小保持有界）、钩子执行和会话持久化。

## 使用场景

- **Claude Agent SDK** 适用于希望使用 Claude Code harness 形态的 Claude-first 产品。
- **Claude Managed Agents** 适用于托管的长时间运行异步工作。
- **OpenAI Agents SDK**（Lesson 16）适用于 OpenAI-first 的对等方案。
- **LangGraph + 自定义工具** 如果你想要以图形式状态机而非循环形态。

## 交付物

`outputs/skill-claude-agent-scaffold.md` 脚手架化一个 Claude Agent SDK 应用，包含子代理、钩子、会话存储、MCP 服务器挂载和 W3C 链路传播。

## 练习

1. 添加一个子代理启动器，将 20 个任务分批为每组 5 个并行子代理。测量编排器上下文大小与单任务一对一方式的对比。
2. 实现一个 `PreToolUse` 钩子，对 `write_file` 调用进行速率限制（每个会话每分钟 5 次）。追踪行为。
3. 将 `list_subkeys` 接入以渲染子代理树。深层嵌套看起来是什么样的？
4. 将此玩具程序移植到真实的 `claude-agent-sdk` Python 包。工具注册方面有什么变化？
5. 阅读 Claude Managed Agents 文档。你何时会从自托管切换到托管？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Agent SDK | "Claude Code 作为库" | Harness 形态：工具、MCP、钩子、子代理、会话存储 |
| 子代理 (Subagent) | "子 agent" | 独立上下文，独立预算；结果向上汇总 |
| 会话存储 (Session store) | "对话数据库" | 持久化、加载、枚举、删除轮次，含子代理级联 |
| 钩子 (Hook) | "生命周期回调" | 工具前后、会话、提示提交、压缩、停止 |
| W3C trace context | "跨进程链路" | 父 span 传播到 CLI 子进程 |
| Managed Agents | "托管 harness" | Anthropic 托管的长时间运行异步工作 |
| `--session-mirror` | "对话镜像" | 流式传输时将会话轮次写入外部文件 |
| MCP server | "工具表面" | 附加到 agent 的外部工具/资源源 |

## 延伸阅读

- [Claude Agent SDK 概览](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Code 的库形式
- [Anthropic, 使用 Claude Agent SDK 构建 agent](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 生产模式
- [Claude Managed Agents 概览](https://platform.claude.com/docs/en/managed-agents/overview) — 托管方案
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 对等方案
