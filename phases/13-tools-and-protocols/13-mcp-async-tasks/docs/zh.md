# 异步任务（SEP-1686） —— 先调用，后获取，应对长时间运行的工作

> 真正的代理工作可能需要几分钟到几小时：CI运行、深度研究综合、批量导出。同步工具调用会断开连接、超时或阻塞UI。于2025年11月25日合并的SEP-1686引入了一个任务原语：任何请求都可以被增强为一个任务，其结果可以稍后获取或通过状态通知进行流式传输。漂移风险提示：任务功能在2026年上半年仍是实验性的；SDK接口仍围绕规范进行设计。

**类型：** 构建
**语言：** Python（标准库，异步任务状态机）
**先决条件：** 第13阶段 · 07（MCP服务器），第13阶段 · 09（传输层）
**时间：** 约75分钟

## 学习目标

- 识别何时应将工具从同步模式提升为任务增强模式（服务器端工作超过30秒）。
- 走过任务生命周期：`working` → `input_required` → `completed` / `failed` / `cancelled`。
- 持久化任务状态，以便在崩溃时不会丢失进行中的工作。
- 正确地轮询 `tasks/status` 并获取 `tasks/result`。

## 问题所在

一个 `generate_report` 工具运行一个需要数分钟的提取流水线。在同步模型下的选项有：

1.  保持连接打开三分钟。远程传输层会断开它；客户端会超时；UI会卡住。
2.  立即返回一个占位符；要求客户端轮询一个自定义端点。这破坏了MCP的一致性。
3.  发送即忘；没有结果。

这些都不好。SEP-1686增加了第四种：任务增强。任何请求（通常是 `tools/call`）都可以被标记为任务。服务器立即返回一个任务ID。客户端轮询 `tasks/status` 并在完成后获取 `tasks/result`。服务器端状态在重启后依然存在。

## 核心概念

### 任务增强

通过设置 `params._meta.task.required: true`（或 `optional: true`，由服务器决定）将请求转变为任务。服务器立即响应：

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` 是服务器保留状态的承诺；过了ttl后，任务结果将被丢弃。

### 工具级选择加入

工具注解可以声明任务支持：

- `taskSupport: "forbidden"` —— 此工具始终同步运行。适用于快速工具。
- `taskSupport: "optional"` —— 客户端可以请求任务增强。
- `taskSupport: "required"` —— 客户端必须使用任务增强。

一个 `generate_report` 工具会是 `required`。一个 `notes_search` 工具会是 `forbidden`。

### 状态

```
working  -> input_required -> working  (loop via elicitation)
working  -> completed
working  -> failed
working  -> cancelled
```

状态机是仅追加的：一旦变为 `completed`、`failed` 或 `cancelled`，任务即为终止状态。

### 方法

- `tasks/status {taskId}` —— 返回当前状态和一个进度提示。
- `tasks/result {taskId}` —— 阻塞或如果任务未完成则返回404。
- `tasks/cancel {taskId}` —— 幂等；终止状态会忽略此请求。
- `tasks/list` —— 可选；枚举活跃和最近完成的任务。

### 流式状态变更

当服务器支持时，客户端可以订阅状态通知：

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

采用流式而非轮询方式的客户端能获得更好的用户体验。轮询始终作为最小接口得到支持。

### 持久化状态

规范要求声明支持任务的服务器必须持久化状态。在ttl时间内，崩溃不应导致已完成的结果丢失。存储范围从SQLite到Redis再到文件系统。第13课的练习框架使用文件系统。

### 取消语义

`tasks/cancel` 是幂等的。如果任务正在执行中，服务器会尝试停止（检查执行器的协作取消）。如果已经是终止状态，该请求则是一个空操作。

### 崩溃恢复

当服务器进程重启时：

1.  加载所有已持久化的任务状态。
2.  将任何进程死亡导致的 `working` 任务标记为 `failed`，错误为 `CRASH_RECOVERY`。
3.  在其ttl时间内保留 `completed` / `failed` / `cancelled`。

### 异步任务与采样

任务本身可以调用 `sampling/createMessage`。长时间运行的研究任务就是这么工作的：服务器的任务线程根据需要采样客户端的模型，而客户端的UI显示任务为 `working` 并附带定期的进度更新。

### 为何是实验性的

SEP-1686于2025年11月25日发布，但更广泛的路线图指出了三个未解决的问题：持久订阅原语、子任务（父子任务关系）以及结果TTL标准化。预计规范将在2026年持续演进。生产代码应仅将任务视为通用场景下的稳定功能，并需防范未来SDK关于子任务的变更。

## 实践应用

`code/main.py` 实现了一个持久化任务存储（基于文件系统）和一个在后台线程中运行的 `generate_report` 工具。客户端调用该工具，立即获得一个任务ID，在工作者更新进度时轮询 `tasks/status`，并在完成后获取 `tasks/result`。取消功能有效；通过杀死工作者线程并重新加载状态来模拟崩溃恢复。

观察要点：

- 任务状态JSON持久化到 `/tmp/lesson-13-tasks/<id>.json`。
- 工作者线程更新 `progress` 字段；轮询显示其正在推进。
- 来自客户端的取消会设置一个事件；工作者检查并提前退出。
- “崩溃”后的状态重载会将进行中的任务标记为 `failed`，错误为 `CRASH_RECOVERY`。

## 交付成果

本课程产出 `outputs/skill-task-store-designer.md`。给定一个长时间运行的工具（研究、构建、导出），技能包括设计任务存储（状态结构、ttl、持久性），选择正确的taskSupport标志，并草拟进度通知。

## 练习

1.  运行 `code/main.py`。启动一个 `generate_report` 任务，轮询状态，然后获取结果。

2.  在运行中途添加一个 `tasks/cancel` 调用。验证工作者遵循该请求且状态变为 `cancelled`。

3.  模拟崩溃恢复：杀死工作者线程，重新启动加载器，并观察 `CRASH_RECOVERY` 的失败模式。

4.  将存储扩展为SQLite。持久性优势相同；查询选项得以扩展（列出会话X的所有任务）。

5.  阅读2026年MCP路线图文章。找出未来一年最可能影响SDK API设计的、与任务相关的未解决问题。

## 关键术语

| 术语 | 人们怎么说 | 其实际含义 |
|------|------------|------------|
| 任务 | “长时间运行的工具调用” | 通过 `_meta.task` 增强用于异步执行的请求 |
| SEP-1686 | “任务规范” | 于2025年11月25日引入任务的规范演进提案 |
| `_meta.task` | “任务信封” | 包含id、状态、ttl的每请求元数据 |
| taskSupport | “工具标志” | 每个工具的 `forbidden` / `optional` / `required` |
| `tasks/status` | “轮询方法” | 获取当前状态和可选的进度提示 |
| `tasks/result` | “获取结果” | 返回已完成的有效载荷，如果未完成则返回404 |
| `tasks/cancel` | “停止它” | 幂等的取消请求 |
| ttl | “保留预算” | 服务器承诺保留任务状态的毫秒数 |
| `notifications/tasks/updated` | “状态推送” | 服务器发起的状态变更事件 |
| 持久化存储 | “崩溃安全状态” | 文件系统/SQLite/Redis 持久层 |

## 扩展阅读

- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) —— 原始提案及完整讨论
- [WorkOS — MCP async tasks for AI agent workflows](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) —— 包含设计原理的设计讲解
- [DeepWiki — MCP task system and async operations](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) —— 机制与状态机
- [FastMCP — Tasks](https://gofastmcp.com/servers/tasks) —— SDK级别的任务实现模式
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) —— 未解决问题与2026年优先事项，包括子任务