# Capstone 01 — 终端原生编码智能体

> 到 2026 年，编码智能体的形态已经定型。一个 TUI 外壳、一个有状态计划、一个沙箱化工具表面、一个规划-行动-观察-恢复循环。Claude Code、Cursor 3 和 OpenCode 从远处看都差不多。这个 capstone 要求你从头到尾构建一个——输入 CLI，输出 pull request——并在 SWE-bench Pro 上与 mini-swe-agent 和 Live-SWE-agent 进行对比评估。你将学到难的不是模型调用，而是工具循环、沙箱和 50 轮运行的成本上限。

**类型：** Capstone
**语言：** TypeScript / Bun（外壳），Python（评估脚本）
**前置条件：** 第 11 阶段（LLM 工程）、第 13 阶段（工具和协议）、第 14 阶段（智能体）、第 15 阶段（自主系统）、第 17 阶段（基础设施）
**涉及阶段：** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**时间：** 35 小时

## 问题

编码智能体在 2026 年成为主导的 AI 应用类别。Claude Code（Anthropic）、带有 Composer 2 和 Agent Tabs 的 Cursor 3（Cursor）、Amp（Sourcegraph）、OpenCode（112k star）、Factory Droids 和 Google Jules 都配备了同一架构的变体：终端外壳、权限工具表面、沙箱和围绕前沿模型构建的规划-行动-观察循环。前沿很窄——Live-SWE-agent 使用 Opus 4.5 在 SWE-bench Verified 上达到了 79.2%——但工程工艺很广。大多数故障模式不是模型错误。它们是工具循环不稳定、上下文中毒、令牌成本失控和破坏性文件系统操作。

你无法从外部推理这些智能体。你必须构建一个，观察循环在第 47 轮因 ripgrep 返回 8MB 匹配而崩溃，然后重建截断层。这就是这个 capstone 的目的。

## 概念

外壳有四个表面。**计划**维护 TodoWrite 风格的有状态对象，模型每轮重写它。**行动**分派工具调用（读、编辑、运行、搜索、git）。**观察**捕获 stdout / stderr / 退出码，截断后将摘要反馈回去。**恢复**处理工具错误而不撑爆上下文窗口或无限循环。2026 年的形态还增加了一个东西：**钩子**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop` 和 `PreCompact` ——可配置的扩展点，运营商在此注入策略、遥测和护栏。

沙箱是 E2B 或 Daytona。每个任务都在新鲜的 devcontainer 中运行，挂载一个可读写的 git worktree。外壳从不触碰宿主机文件系统。worktree 在成功或失败时被销毁。成本控制在三个层面强制执行：每轮令牌上限、每会话美元预算和硬性轮次限制（通常为 50）。可观测性层是带有 GenAI 语义约定的 OpenTelemetry span，传送到自托管 Langfuse。

## 架构

```
  用户 CLI  ->  外壳（Bun + Ink TUI）
                  |
                  v
           计划/行动/观察循环  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          （通过 OpenRouter，模型无关）
                  v
           工具分派器（MCP StreamableHTTP 客户端）
                  |
     +------------+------------+----------+
     v            v            v          v
  读/编辑      ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona 沙箱  （worktree 隔离）
                  |
                  v
           钩子：Pre/Post、Session、Prompt、Compact
                  |
                  v
           OpenTelemetry -> Langfuse（span、令牌、$）
                  |
                  v
           通过 GitHub App 发布 PR
```

## 技术栈

- 外壳运行时：Bun 1.2 + Ink 5（终端中的 React）
- 模型访问：OpenRouter 统一 API，支持 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5（用于最困难的场景）
- 工具传输：Model Context Protocol StreamableHTTP（MCP 2026 修订版）
- 沙箱：E2B sandboxes（JS SDK）或 Daytona devcontainers
- 代码搜索：ripgrep 子进程、tree-sitter 解析器（预编译，支持 17 种语言）
- 隔离：每个任务 `git worktree add`，成功/失败时清理
- 评估外壳：SWE-bench Pro（已验证子集）+ Terminal-Bench 2.0 + 自定义 30 任务保留集
- 可观测性：OpenTelemetry SDK，带 `gen_ai.*` semconv → 自托管 Langfuse
- PR 发布：GitHub App，细粒度令牌，范围限于目标仓库

```figure
ce-agent-loop
```

## 构建它

1. **TUI 和命令循环。** 用 Ink 搭建 Bun 项目。接受 `agent run <repo> "<task>"`。打印分屏视图：计划窗格（顶部）、工具调用流（中部）、令牌预算（底部）。添加 Ctrl-C 取消，退出前触发 `SessionEnd` 钩子。

2. **计划状态。** 定义类型化的 TodoWrite schema（待处理 / 进行中 / 已完成项，带注释）。模型每轮作为工具调用重写完整状态——不要让它增量修改。将计划持久化到 `.agent/state.json`，以便崩溃后可以恢复。

3. **工具表面。** 定义六个工具：`read_file`、`edit_file`（带 diff 预览）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带超时）、`git`（状态/diff/提交/推送）。通过 MCP StreamableHTTP 暴露，使外壳与传输无关。每个工具返回截断的输出（每次调用上限 4k 令牌）。

4. **沙箱封装。** 每个任务启动一个 E2B 沙箱。`git worktree add -b agent/$TASK_ID` 一个新鲜分支。所有工具调用在沙箱内执行。宿主机文件系统不可访问。

5. **钩子。** 实现全部八个 2026 钩子类型。连接至少四个用户编写的钩子：(a) `PreToolUse` 破坏性命令守卫，阻止 worktree 外的 `rm -rf`；(b) `PostToolUse` 令牌记账；(c) `SessionStart` 预算初始化；(d) `Stop` 写入最终追踪束。

6. **评估循环。** 克隆 SWE-bench Pro Python 的 30 个议题子集。对你的外壳逐个运行。与 mini-swe-agent（最小基线）在 pass@1、每任务轮次和每任务成本上对比。将结果写入 `eval/results.jsonl`。

7. **成本控制。** 硬截止线：50 轮、200k 上下文、每个任务 $5。`PreCompact` 钩子在 150k 标记处将较早的轮次总结为前置状态块，释放空间供新观察使用而不丢失计划。

8. **PR 发布。** 成功后，最后一步是 `git push` + GitHub API 调用，打开 PR 并在正文中包含计划和 diff 摘要。

## 使用它

```
$ agent run ./my-repo "修复 worker.rs 中的竞态条件"
[plan]  1 定位 worker.rs 并列举互斥锁使用
        2 识别竞争下的共享状态
        3 提出修复方案，验证测试
[tool]  ripgrep mutex.*lock -t rust           (44 个匹配，已截断)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (通过)
[plan]  1 完成 · 2 完成 · 3 完成
[done]  PR 已打开：#482   轮次=9   令牌=38k   成本=$0.41
```

## 交付物

交付技能位于 `outputs/skill-terminal-coding-agent.md`。给定仓库路径和任务描述，它在沙箱中运行完整的计划-行动-观察循环，并返回 PR URL 和追踪束。本 capstone 的评分标准：

| 权重 | 标准 | 如何测量 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 对比基线 | 你的外壳 vs mini-swe-agent 在 30 个匹配的 Python 任务上 |
| 20 | 架构清晰度 | 计划/行动/观察分离、钩子表面、工具 schema —— 对照 Live-SWE-agent 布局审查 |
| 20 | 安全性 | 沙箱逃逸测试、权限提示、破坏性命令守卫通过红队测试 |
| 20 | 可观测性 | 追踪完整性（100% 工具调用带 span）、每轮令牌记账 |
| 15 | 开发者体验 | 冷启动 < 2s、崩溃恢复能继续计划、Ctrl-C 在中途取消工具 |
| **100** | | |

## 练习

1. 将基础模型从 Claude Sonnet 4.7 替换为 vLLM 上的 Qwen3-Coder-30B。对比 pass@1 和每任务成本。报告开源模型的不足之处。

2. 添加一个 `reviewer` 子智能体，在 PR 发布前读取 diff 并可请求修订循环。测量误报评审是否使 SWE-bench 通过率低于单智能体基线（提示：通常会）。

3. 压力测试沙箱：编写一个尝试 `curl` 外部 URL 的任务和一个在 worktree 外写入的任务。确认两者都被 PreToolUse 钩子阻止。记录这些尝试。

4. 使用较小模型（Haiku 4.5）实现 `PreCompact` 总结。测量在 3x 压缩时损失了多少计划保真度。

5. 将 MCP StreamableHTTP 传输替换为 stdio。基准测试冷启动和每次调用延迟。为仅本地使用选择一个赢家。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------------------|------------------------------------------|
| Harness | "智能体循环" | 包围模型的代码，负责分派工具、维护计划状态并执行预算 |
| Hook | "智能体事件监听器" | 用户在八个生命周期事件之一上运行的脚本，由外壳执行 |
| Worktree | "Git 沙箱" | 链接到单独路径的 git 签出；可丢弃而不触碰主克隆 |
| TodoWrite | "计划状态" | 模型每轮重写的类型化列表，包含待处理/进行中/已完成项 |
| StreamableHTTP | "MCP 传输" | 2026 MCP 修订版：长连接 HTTP，带双向流；替代 SSE |
| Token ceiling | "上下文预算" | 每轮或每会话的输入+输出令牌上限；触发压缩或终止 |
| pass@1 | "单次尝试通过率" | 首次运行即解决的 SWE-bench 任务比例，无重试或窥探测试集 |

## 延伸阅读

- [Claude Code 文档](https://docs.anthropic.com/en/docs/claude-code) —— Anthropic 参考外壳
- [Cursor 3 更新日志](https://cursor.com/changelog) —— Agent Tabs 和 Composer 2 产品说明
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) —— 用于 SWE-bench 外壳对比的最小基线
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) —— 使用 Opus 4.5 在 SWE-bench Verified 上达到 79.2%
- [OpenCode](https://opencode.ai) —— 开源外壳，112k stars
- [SWE-bench Pro 排行榜](https://www.swebench.com) —— 本 capstone 针对的评估
- [Model Context Protocol 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) —— StreamableHTTP、能力元数据
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— 工具调用和令牌使用的 span schema
