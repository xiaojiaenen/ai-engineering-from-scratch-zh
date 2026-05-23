# 毕业项目 01 — 终端原生编码代理

> 到 2026 年，编码代理的形态已经确定。一个 TUI 框架、一个有状态的计划、一个沙箱化的工具界面、一个“计划-执行-观察-恢复”的循环。Claude Code、Cursor 3 和 OpenCode 从 50 英尺外看去都长得一样。这个毕业项目要求你从头到尾构建一个 —— CLI 输入，Pull Request 输出 —— 并在 SWE-bench Pro 上将其与 mini-swe-agent 和 Live-SWE-agent 进行比较。你将明白难点不在于模型调用，而在于工具循环、沙箱以及 50 轮运行的成本上限。

**类型：** 毕业项目
**语言：** TypeScript / Bun（框架），Python（评估脚本）
**先修课程：** 阶段 11（LLM 工程），阶段 13（工具与协议），阶段 14（代理），阶段 15（自治系统），阶段 17（基础设施）
**涉及阶段：** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**时间：** 35 小时

## 问题

到 2026 年，编码代理已成为主导的 AI 应用类别。Claude Code (Anthropic)、配备 Composer 2 和 Agent Tabs 的 Cursor 3 (Cursor)、Amp (Sourcegraph)、OpenCode (11.2 万星)、Factory Droids 和 Google Jules 都发布了相似的架构变体：一个终端框架、一个权限化的工具界面、一个沙箱，以及围绕前沿模型构建的计划-执行-观察循环。前沿差距狭窄 —— Live-SWE-agent 使用 Opus 4.5 在 SWE-bench Verified 上达到了 79.2% —— 但工程技艺范围广泛。大多数失败模式并非模型错误，而是工具循环不稳定、上下文污染、token 成本失控以及破坏性文件系统操作。

你无法仅从外部理解这些代理。你必须亲自构建一个，观察循环在第 47 轮因 ripgrep 返回 8MB 匹配结果而崩溃，然后重建截断层。这就是这个毕业项目的意义所在。

## 概念

框架包含四个界面。**计划** 维护一个类似 TodoWrite 的状态对象，模型每轮都会重写它。**执行** 分派工具调用（读取、编辑、运行、搜索、git）。**观察** 捕获 stdout / stderr / 退出码，截断后将摘要反馈。**恢复** 处理工具错误，避免上下文窗口耗尽或无限循环。2026 年的形态增加了一样东西：**钩子**。`PreToolUse`, `PostToolUse`, `SessionStart`, `SessionEnd`, `UserPromptSubmit`, `Notification`, `Stop`, 和 `PreCompact` — 可配置的扩展点，操作员在此注入策略、遥测和防护栏。

沙箱是 E2B 或 Daytona。每个任务在一个全新的开发容器中运行，该容器以读写方式挂载了一个 git 工作树。框架从不接触主机文件系统。工作树在成功或失败时都会被拆除。成本控制在三层强制执行：单轮 token 上限、单会话美元预算以及硬性轮数限制（通常为 50）。可观测性层是采用 GenAI 语义规范的 OpenTelemetry span，发送到自托管的 Langfuse。

## 架构

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## 技术栈

- 框架运行时：Bun 1.2 + Ink 5 (终端内的 React)
- 模型访问：OpenRouter 统一 API，支持 Claude Sonnet 4.7, GPT-5.4-Codex, Gemini 3 Pro, Opus 4.5（用于最难的任务）
- 工具传输：模型上下文协议可流式传输 HTTP (MCP 2026 修订版)
- 沙箱：E2B 沙箱 (JS SDK) 或 Daytona 开发容器
- 代码搜索：ripgrep 子进程，tree-sitter 解析器支持 17 种语言（预编译）
- 隔离：`git worktree add` 每个任务，成功/失败时清理
- 评估框架：SWE-bench Pro (已验证子集) + Terminal-Bench 2.0 + 你自己的 30 个任务留出集
- 可观测性：带有 `gen_ai.*` 语义规范的 OpenTelemetry SDK → 自托管 Langfuse
- PR 发布：具有细粒度令牌的 GitHub App，范围限定在目标仓库

## 构建步骤

1.  **TUI 和命令循环。** 使用 Ink 搭建一个 Bun 项目脚手架。接受 `agent run <repo> "<task>"`。打印一个分屏视图：计划窗格（顶部）、工具调用流（中部）、token 预算（底部）。添加 Ctrl-C 取消功能，退出前触发 `SessionEnd` 钩子。

2.  **计划状态。** 定义一个类型化的 TodoWrite 模式（待处理 / 进行中 / 已完成的项目及备注）。模型每轮将完整状态作为工具调用进行重写 —— 不要让它增量修改。将计划持久化到 `.agent/state.json`，以便崩溃时可以恢复。

3.  **工具界面。** 定义六个工具：`read_file`, `edit_file`（带差异预览）, `ripgrep`, `tree_sitter_symbols`, `run_shell`（带超时）, `git`（状态/差异/提交/推送）。通过 MCP 可流式传输 HTTP 暴露接口，使框架与传输协议无关。每个工具返回截断的输出（每次调用上限 4k token）。

4.  **沙箱封装。** 每个任务生成一个 E2B 沙箱。`git worktree add -b agent/$TASK_ID` 一个全新的分支。所有工具调用在沙箱内执行。主机文件系统不可访问。

5.  **钩子。** 实现所有八种 2026 年钩子类型。至少连接四个用户编写的钩子：(a) `PreToolUse` 破坏性命令防护，阻止工作树外的 `rm -rf`，(b) `PostToolUse` token 计费，(c) `SessionStart` 预算初始化，(d) `Stop` 写入最终的跟踪包。

6.  **评估循环。** 克隆 SWE-bench Pro Python 的一个 30 个问题子集。针对每个问题运行你的框架。在 pass@1（单次尝试通过率）、任务轮数和每任务成本 ($) 方面与 mini-swe-agent（最小基线）进行比较。将结果写入 `eval/results.jsonl`。

7.  **成本控制。** 硬性截止：50 轮，200k 上下文，每任务 $5。`PreCompact` 钩子在 150k 处将较旧的轮次总结为先前状态块，为新观察腾出空间而不丢失计划。

8.  **PR 发布。** 成功后，最后一步是 `git push` + 一个 GitHub API 调用，该调用打开一个 PR，正文中包含计划和差异摘要。

## 使用示例

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## 交付

交付技能存放在 `outputs/skill-terminal-coding-agent.md`。给定一个仓库路径和一个任务描述，它在沙箱中运行完整的计划-执行-观察循环，并返回一个 PR URL 加上一个跟踪包。此毕业项目的评分标准：

| 权重 | 标准 | 衡量方法 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 与基线比较 | 你的框架与 mini-swe-agent 在 30 个匹配 Python 任务上的对比 |
| 20 | 架构清晰度 | 计划/执行/观察分离、钩子界面、工具模式 —— 与 Live-SWE-agent 布局对照评审 |
| 20 | 安全性 | 沙箱逃逸测试、权限提示、破坏性命令防护通过红队测试 |
| 20 | 可观测性 | 跟踪完整性（100% 工具调用被 span 覆盖）、每轮 token 计费 |
| 15 | 开发者体验 | 冷启动 < 2 秒，崩溃恢复可续接计划，Ctrl-C 可在工具执行中途干净取消 |
| **100** | | |

## 练习

1.  将底层模型从 Claude Sonnet 4.7 替换为在 vLLM 上运行的 Qwen3-Coder-30B。比较 pass@1 和每任务成本 ($)。报告开放模型表现不佳之处。

2.  添加一个 `reviewer` 子代理，该子代理在 PR 发布前读取差异，并可以请求修订循环。测量误报审查是否会将 SWE-bench 通过率降至单代理基线以下（提示：通常会）。

3.  对沙箱进行压力测试：编写一个尝试 `curl` 外部 URL 的任务和一个尝试在工作树外写入的任务。确认两者都被 PreToolUse 钩子阻止。记录这些尝试。

4.  使用较小的模型（Haiku 4.5）实现 `PreCompact` 摘要功能。测量在 3 倍压缩下计划保真度的损失程度。

5.  将 MCP 可流式传输 HTTP 传输替换为 stdio。对冷启动和单次调用延迟进行基准测试。为本地专用场景选择一个胜出者。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Harness | "代理循环" | 模型周围的代码，负责分派工具、维护计划状态和强制执行预算 |
| Hook | "代理事件监听器" | 由框架在八种生命周期事件之一运行的用户编写脚本 |
| Worktree | "Git 沙箱" | 位于独立路径的链接 git 检出；用完即弃而不影响主克隆 |
| TodoWrite | "计划状态" | 模型每轮重写的类型化待处理/进行中/已完成项目列表 |
| StreamableHTTP | "MCP 传输" | 2026 年 MCP 修订版：支持双向流的长连接 HTTP；取代了 SSE |
| Token 上限 | "上下文预算" | 输入+输出 token 的单轮或单会话限制；触发压缩或终止 |
| pass@1 | "单次尝试通过率" | 在未重试或未窥视测试集的情况下首次运行解决的 SWE-bench 任务比例 |

## 延伸阅读

- [Claude Code 文档](https://docs.anthropic.com/en/docs/claude-code) — Anthropic 的参考框架
- [Cursor 3 更新日志](https://cursor.com/changelog) — Agent Tabs 和 Composer 2 产品说明
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — 用于 SWE-bench 框架比较的最小基线
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 使用 Opus 4.5 在 SWE-bench Verified 上达到 79.2%
- [OpenCode](https://opencode.ai) — 开放框架，11.2 万星
- [SWE-bench Pro 排行榜](https://www.swebench.com) — 本毕业项目目标的评估基准
- [模型上下文协议 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、能力元数据
- [OpenTelemetry GenAI 语义规范](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 工具调用和 token 使用的 span 模式