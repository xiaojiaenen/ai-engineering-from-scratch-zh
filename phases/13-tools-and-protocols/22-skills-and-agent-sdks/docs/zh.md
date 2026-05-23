# 技能与代理 SDK — Anthropic Skills、AGENTS.md、OpenAI Apps SDK

> MCP 说明“有哪些工具可用”。技能说明“如何执行任务”。2026 年的技术栈将两者分层。Anthropic 的 Agent Skills（开放标准，2025 年 12 月发布）以 SKILL.md 形式提供，支持渐进式披露。OpenAI 的 Apps SDK 是 MCP 加上小部件元数据。AGENTS.md（现已用于 60,000 多个代码仓库）位于仓库根目录，作为项目级代理上下文。本课说明了每项所涵盖的内容，并构建了一个可在不同代理间传输的最小 SKILL.md + AGENTS.md 组合包。

**类型：** 学习
**语言：** Python（标准库，SKILL.md 解析器和加载器）
**前置知识：** 阶段 13 · 07（MCP 服务器）
**时间：** 约 45 分钟

## 学习目标

- 区分三个层次：AGENTS.md（项目上下文）、SKILL.md（可复用的专业知识）、MCP（工具）。
- 使用 YAML 前置元数据和渐进式披露编写 SKILL.md。
- 以文件系统方式将技能加载到代理运行时。
- 将技能与 MCP 服务器和 AGENTS.md 组合，使得一个包能在 Claude Code、Cursor 和 Codex 中工作。

## 问题所在

一位工程师将发布说明编写工作流提炼为一个包含多个步骤的提示：“读取最新的已合并 PR。按领域分组。总结每个 PR。按照团队风格编写变更日志条目。发布到 Slack 草稿。”他们将其放在 Notion 文档中供团队使用。

现在，他们想从 Claude Code、Cursor 和 Codex CLI 中使用这个工作流。每个代理加载指令的方式都不同：Claude Code 使用斜杠命令，Cursor 使用规则，Codex 使用 `.codex.md`。工程师复制了三遍工作流并维护三个副本。

AGENTS.md 和 SKILL.md 一起解决了这个问题：

- **AGENTS.md** 位于仓库根目录。每个兼容的代理在会话开始时读取它。“这个项目如何运作？有哪些约定？哪些命令用于运行测试？”
- **SKILL.md** 是一个可移植的包：YAML 前置元数据（名称、描述）+ Markdown 正文 + 可选资源。支持技能的代理按名称按需加载它们。
- **MCP**（阶段 13 · 06-14）处理技能需要调用的工具。

三个层次，一个可移植的制品。

## 核心概念

### AGENTS.md (agents.md)

于 2025 年末推出，截至 2026 年 4 月已被 60,000 多个代码仓库采用。位于仓库根目录的一个文件。格式如下：

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

代理在会话开始时读取此文件，并用它来校准其针对该项目的行为。2026 年的每个编码代理都支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md 格式

Anthropic 的 Agent Skills（作为开放标准于 2025 年 12 月发布）：

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

前置元数据声明技能的标识。正文是技能加载时显示给模型的提示。

### 渐进式披露

技能可以引用子资源，代理仅在需要时获取这些资源。示例：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 说“参见 style-guide.md 了解样式规则。”代理仅在技能实际运行时才拉取 style-guide.md。这避免了在提示中堆满模型可能不需要的细节。

### 文件系统发现

代理运行时在已知目录中扫描 SKILL.md 文件：

- `~/.anthropic/skills/*/SKILL.md`
- 项目 `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

加载基于文件夹名称和前置元数据 `name`。Claude Code、Anthropic Claude Agent SDK 和 SkillKit（跨代理）都遵循此模式。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`（TypeScript）和 `claude-agent-sdk`（Python）在会话开始时加载技能，并在运行时中将它们公开为可调用的“代理”。当用户调用技能时，代理循环会调度到该技能。

### OpenAI Apps SDK

于 2025 年 10 月推出；直接构建在 MCP 之上。将 OpenAI 之前的 Connectors 和 Custom GPT Actions 统一到一个开发者界面下。一个 Apps SDK 应用是：

- 一个 MCP 服务器（工具、资源、提示）。
- 加上用于 ChatGPT 界面的小部件元数据。
- 加上一个可选的 MCP Apps `ui://` 资源用于交互界面。

相同的协议，更丰富的用户体验。

### 通过 SkillKit 实现跨代理可移植性

像 SkillKit 和类似的跨代理分发层这样的工具，可以将单个 SKILL.md 转换为 32 多个 AI 代理（Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）的原生格式。一个事实来源；众多消费者。

### 三层技术栈

| 层次 | 文件 | 加载时机 | 目的 |
|------|------|----------|------|
| AGENTS.md | 仓库根目录 | 会话开始 | 项目级约定 |
| SKILL.md | 技能目录 | 技能被调用时 | 可复用的工作流 |
| MCP 服务器 | 外部进程 | 需要工具时 | 可调用的操作 |

三者组合：代理在会话开始时读取 AGENTS.md，用户调用技能，技能的指令包括 MCP 工具调用，代理通过 MCP 客户端进行调度。

## 动手实践

`code/main.py` 提供了一个标准库的 SKILL.md 解析器和加载器。它发现 `./skills/` 下的技能，解析 YAML 前置元数据和 Markdown 正文，并生成一个以技能名称为键的字典。然后它模拟一个代理循环，按名称调用 `release-notes-writer`。

需要关注的点：

- 使用最小化的标准库解析器解析 YAML 前置元数据（无 `pyyaml` 依赖）。
- 技能正文原样存储；代理在调用时将其添加到系统提示的前面。
- 通过一个 `read_subresource` 函数演示渐进式披露，该函数按需拉取引用的文件。

## 交付物

本课将产生 `outputs/skill-agent-bundle.md`。给定一个工作流，该技能生成组合的 SKILL.md + AGENTS.md + MCP 服务器蓝图包，可在不同代理间移植。

## 练习

1.  运行 `code/main.py`。在 `skills/` 下添加第二个技能，并确认加载器能够识别它。
2.  为本课程仓库编写一个 AGENTS.md。包括测试命令、样式约定和阶段 13 的概念模型。
3.  将你团队内部文档中的一个多步骤工作流移植到一个 SKILL.md 中。验证它能在 Claude Code 中加载。
4.  手动将该技能翻译成 Cursor 和 Codex 的原生规则格式。计算格式之间的差异——这就是 SkillKit 自动化的翻译界面。
5.  阅读 Anthropic Agent Skills 博客文章。找出 Claude Agent SDK 中本课加载器未涵盖的一个特性。（提示：代理子调用。）

## 关键术语

| 术语 | 人们常说的 | 其实际含义 |
|------|------------|------------|
| SKILL.md | “技能文件” | YAML 前置元数据加 Markdown 正文，由代理运行时加载 |
| AGENTS.md | “仓库根目录的代理上下文” | 会话开始时读取的项目级约定文件 |
| 渐进式披露 | “延迟加载子资源” | 技能正文引用仅在需要时才拉取的文件 |
| 前置元数据 | “顶部的 YAML 块” | 包含在 `---` 分隔符中的元数据（名称、描述） |
| Claude Agent SDK | “Anthropic 的技能运行时” | `@anthropic-ai/claude-agent-sdk`，加载技能和路由 |
| OpenAI Apps SDK | “MCP 加小部件元数据” | OpenAI 基于 MCP 并集成 ChatGPT 界面钩子的开发者平台 |
| 技能发现 | “文件系统扫描” | 遍历已知目录查找 SKILL.md，按名称索引 |
| 跨代理可移植性 | “一个技能，多个代理” | 通过 SkillKit 类工具将一个 SKILL.md 翻译为 32 多个代理的格式 |
| Agent Skill | “可移植的专业知识” | MCP 工具概念之外的可复用任务模板 |
| Apps SDK | “MCP 加 ChatGPT 界面” | 基于 MCP 统一了 Connectors 和 Custom GPTs |

## 扩展阅读

- [Anthropic — Agent Skills 公告](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025 年 12 月发布
- [Anthropic — Agent Skills 文档](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md 格式参考
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — 面向 ChatGPT 的基于 MCP 的开发者平台
- [agents.md](https://agents.md/) — AGENTS.md 格式和采用列表
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — 官方技能示例