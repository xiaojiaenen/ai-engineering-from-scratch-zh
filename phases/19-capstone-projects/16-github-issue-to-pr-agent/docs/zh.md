# 毕业项目 16 — GitHub Issue 到 PR 的自主代理

> AWS 远程 SWE 代理、Cursor 后台代理、OpenAI Codex 云端版和 Google Jules 都推出了 2026 年形态的产品：标记一个 Issue，获得一个 PR。在云沙箱中运行代理，验证测试通过，并发布一个附带理由说明、可直接审查的 PR。难点在于自动重现仓库的构建环境、防止凭证泄露、实施单仓库预算，以及确保代理无法强制推送。本毕业项目构建自托管版本，并在成本和通过率上与托管方案进行对比。

**类型：** 毕业项目
**语言：** Python（代理）、TypeScript（GitHub 应用）、YAML（Actions）
**前置要求：** 阶段 11（LLM 工程）、阶段 13（工具）、阶段 14（代理）、阶段 15（自主性）、阶段 17（基础设施）
**练习阶段：** P11 · P13 · P14 · P15 · P17
**时间：** 30 小时

## 问题

异步云编码代理不同于交互式编码代理（毕业项目 01），是一个独立的产品类别。其用户体验是一个 GitHub 标签。你给 Issue 打上标签 `@agent fix this`，一个工作节点会在云沙箱中启动，克隆仓库，运行测试，编辑文件，进行验证，并开启一个 PR，理由说明会包含在正文中。没有交互循环，没有终端。AWS 远程 SWE 代理、Cursor 后台代理、OpenAI Codex 云端版、Google Jules 和 Factory Droids 都在向此形态收敛。

工程挑战非常具体：环境重现（代理必须从零开始构建仓库，不能依赖缓存的开发镜像）、不稳定测试（必须重跑或隔离）、凭证范围限定（一个拥有最小细粒度权限的 GitHub 应用）、强制执行单仓库每日预算，以及禁止强制推送策略。本毕业项目将衡量通过率、成本、安全性，并与托管方案进行对比。

## 概念

触发器是一个 GitHub webhook（Issue 标签或 PR 评论）。调度器将任务入队到 ECS Fargate 或 Lambda。工作节点将仓库拉取到一个基于仓库推断的通用 Dockerfile 构建的 Daytona 或 E2B 沙箱中。代理针对 Claude Opus 4.7 或 GPT-5.4-Codex 运行一个 mini-swe-agent 或 SWE-agent v2 循环。它迭代执行：读取代码、提出修复方案、应用补丁、运行测试。

验证是门控步骤。在 PR 开启之前，必须在沙箱内通过完整的 CI 测试。计算覆盖率变化量；如果下降超过阈值，PR 会开启但会被打上 `needs-review` 标签。代理会将理由说明作为 PR 描述发布，并附带一个 `@agent` 线程，审查者可以在此提及相关问题进行后续跟进。

安全性通过两个不同的 GitHub 界面进行范围限定：该应用程序提供一个短期有效的安装令牌，具有 `workflows: read` 以及狭窄的仓库内容/PR 范围；分支保护（而非应用程序权限）强制执行“禁止直接写入 `main`”和“禁止强制推送”——该应用程序永远不会被添加到绕过列表中。对 `.github/workflows` 的基于路径的只读访问并非真正的 GitHub 应用程序原生功能，因此代理对文件编辑的允许列表必须在工作节点侧强制执行。单仓库每日预算上限在调度器侧强制执行（例如，每个仓库每天最多 5 个 PR，每个 PR 上限 20 美元）。

## 架构

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## 技术栈

- 触发器：具有细粒度令牌的 GitHub 应用程序；通过 Lambda 或 Fly.io 接收 webhook
- 工作节点：ECS Fargate 任务（或 GitHub Actions 自托管运行器）
- 沙箱：每个任务对应一个 Daytona devcontainer 或 E2B 沙箱
- 代理循环：基于 Claude Opus 4.7 / GPT-5.4-Codex 的 mini-swe-agent 基线或 SWE-agent v2
- 检索：tree-sitter 仓库映射 + ripgrep
- 验证：沙箱内完整 CI + 覆盖率变化量门控
- 可观测性：Langfuse，每个 PR 的跟踪档案链接在 PR 正文中
- 预算：单仓库每日美元上限；每个仓库每天最大 PR 数

## 构建步骤

1.  **GitHub 应用程序。** 细粒度安装令牌：issues 读写、pull_requests 写、contents 读写、workflows 读。分支保护（唯一能实现此功能的界面）强制执行“禁止直接推送到 `main`”和“禁止强制推送”；该应用不在绕过列表中。工作节点通过检查提议的 diff 的允许列表来强制执行“禁止写入 `.github/workflows`”下的内容，因为 GitHub 应用程序权限不是基于路径的。

2.  **Webhook 接收器。** Lambda 函数接受 Issue 标签 / PR 评论 webhook。按标签 `@agent fix this` 过滤。入队到 SQS。

3.  **调度器。** 从 SQS 中取出任务。强制执行单仓库每日预算。启动一个 ECS Fargate 任务，包含仓库 URL、Issue 正文和一个全新的 Daytona 沙箱。

4.  **环境推断。** 检测语言（Python, Node, Go, Rust）和包管理器（uv, pnpm, go mod, cargo）。如果不存在 Dockerfile，则动态生成一个。

5.  **代理循环。** 使用 Claude Opus 4.7 的 mini-swe-agent 或 SWE-agent v2。工具：ripgrep、tree-sitter 仓库映射、read_file、edit_file、run_tests、git。硬性限制：成本 20 美元，挂钟时间 30 分钟，代理轮次 30 次。

6.  **验证。** 循环结束后，在沙箱内运行完整测试套件。通过 jacoco / coverage.py 计算覆盖率变化量。如果 CI 失败：停止，不开启 PR。如果覆盖率下降超过 2%：开启 PR 并打上 `needs-review` 标签。

7.  **发布 PR。** 推送代理分支。通过 GitHub API 开启 PR，包含：标题、理由说明、diff 摘要、跟踪 URL、成本、轮次。

8.  **凭证卫生。** 工作节点使用短期有效的 GitHub 应用程序安装令牌运行。日志在归档前会清理掉敏感信息。

9.  **评估。** 30 个难度各异的种子内部 Issue。衡量通过率、PR 质量（diff 大小、风格、覆盖率）、成本、延迟。与 Cursor 后台代理和 AWS 远程 SWE 代理在相同 Issue 上的表现进行比较。

## 使用

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## 交付

`outputs/skill-issue-to-pr.md` 是交付物。一个 GitHub 应用程序 + 异步云工作节点，将打了标签的 Issue 转化为成本可控、凭证范围限定的、可直接审查的 PR。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 30 个 Issue 的通过率 | 端到端成功（CI 通过 + 覆盖率达标） |
| 20 | PR 质量 | Diff 大小、覆盖率变化量、风格一致性 |
| 20 | 解决每个 Issue 的成本和延迟 | 每个 PR 的美元成本和挂钟时间 |
| 20 | 安全性 | 范围限定的令牌、单仓库预算、禁止强制推送、凭证卫生 |
| 15 | 操作员用户体验 | 理由说明评论、重试功能、@提及跟进 |
| **100** | | |

## 练习

1.  添加“修复不稳定测试”模式：标签 `@agent stabilize-flake TestX` 会在沙箱内运行该测试 50 次，并提出一个使其稳定化的最小更改方案。
2.  比较三个共享 Issue 上与 Cursor 后台代理的成本差异。报告各工具在哪方面胜出。
3.  实现一个预算仪表盘：显示单仓库每日成本、单用户成本。对异常情况进行告警。
4.  构建一个“干运行”模式，开启一个草稿 PR 但不运行 CI，以便审查者能以低成本检查计划。
5.  添加保留策略：超过 7 天未合并的 PR 分支将被自动删除。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| GitHub 应用程序 | “作用域限定的机器人身份” | 具有细粒度权限和短期安装令牌的应用程序 |
| 异步云代理 | “后台代理” | 在云沙箱而非终端中运行的非交互式工作节点 |
| 环境推断 | “Dockerfile 合成” | 检测语言和包管理器，如果缺失则生成 Dockerfile |
| 验证 | “沙箱内 CI” | 在开启 PR 前在工作节点内运行完整测试套件 |
| 覆盖率变化量 | “覆盖率保持” | 基线分支到代理分支的测试覆盖率百分比变化 |
| 单仓库预算 | “每日上限” | 在调度器侧强制执行的美元和 PR 数量上限 |
| 理由说明 | “PR 正文解释” | 代理对更改内容和原因的总结；必须包含在 PR 正文中 |

## 扩展阅读

- [AWS 远程 SWE 代理](https://github.com/aws-samples/remote-swe-agents) — 规范的异步云代理参考
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI 参考
- [Cursor 后台代理](https://docs.cursor.com/background-agent) — 商业替代方案
- [OpenAI Codex（云端版）](https://openai.com/codex) — 托管竞争对手
- [Google Jules](https://jules.google) — 谷歌的托管版本
- [Factory Droids](https://www.factory.ai) — 另一个商业参考
- [GitHub 应用程序文档](https://docs.github.com/en/apps) — 作用域限定的机器人身份
- [Daytona 云沙箱](https://daytona.io) — 参考沙箱