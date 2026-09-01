# Capstone 16 — GitHub Issue 转 PR 自治代理

> 标记一个 Issue，获得一个 PR —— 2026 年自治编码代理的产品形态：在云端沙箱中运行代理、验证测试通过，并附带理由说明发布一个可审查的 PR。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud、Google Jules 均已推出此类产品。难点在于自动复现仓库的构建环境、防止凭证泄露、执行按仓库计费预算，并确保代理无法强制推送。本 Capstone 构建自托管版本，并对比其成本与通过率与托管方案。

**类型：** Capstone
**语言：** Python（代理）、TypeScript（GitHub App）、YAML（Actions）
**前置要求：** Phase 11（LLM 工程）、Phase 13（工具）、Phase 14（代理）、Phase 15（自治）、Phase 17（基础设施）
**涉及阶段：** P11 · P13 · P14 · P15 · P17
**时间：** 30 小时

## 问题

异步云端编码代理与交互式编码代理（Capstone 01）属于不同的产品类别。其用户体验为一个 GitHub 标签。你给 Issue 打上 `@agent fix this` 标签后，一台工作节点会在云端沙箱中启动，克隆仓库、运行测试、编辑文件、验证结果，并在 PR 正文中附上代理的理由说明。没有交互循环，没有终端。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud、Google Jules 以及 Factory Droids 均趋同于此形态。

工程挑战具体而明确：环境复现（代理必须从零构建仓库，无缓存开发镜像可用）、不稳定测试（必须重跑或隔离）、凭证作用域限定（使用最小细粒度权限的 GitHub App）、每仓库每日预算执行，以及禁止强制推送策略。Capstone 将衡量通过率、成本与安全相对于托管替代方案的优劣。

## 概念

触发器为 GitHub webhook（Issue 标签或 PR 评论）。调度器将任务入队至 ECS Fargate 或 Lambda。工作节点将仓库拉取到 Daytona 或 E2B 沙箱中，沙箱基于从仓库推断的通用 Dockerfile（识别语言、框架）构建。代理在 Claude Opus 4.7 或 GPT-5.4-Codex 上运行 mini-swe-agent 或 SWE-agent v2 循环：读取代码、提出修复、应用补丁、运行测试。

验证是关键门禁。完整的 CI 必须在沙箱内通过后才能打开 PR。计算覆盖率差异；若低于阈值，PR 仍会打开但被打上 `needs-review` 标签。代理将理由说明发布为 PR 描述，并在评论区附加 `@agent` 线程供审查者 ping 以跟进。

安全性通过两个不同的 GitHub 层面进行约束：App 提供短期的安装令牌，权限包括 `workflows: read` 及狭窄的仓库内容与 PR 作用域；分支保护（而非 App 权限）强制执行"禁止直接写入 `main`"和"禁止强制推送"——该 App 永远不会加入豁免列表。`.github/workflows` 的路径级只读访问并非真实的 GitHub App 原语，因此代理对文件编辑的白名单必须在 worker 端强制执行。每日每仓库的预算上限在调度器端强制执行（例如：每个仓库每天最多 5 个 PR，每个 PR 最高 $20）。

## 架构

```
GitHub Issue 被标记 `@agent fix` 或 PR 评论
            |
            v
    GitHub App webhook -> AWS Lambda 调度器
            |
            v
    ECS Fargate 任务（或 GitHub Actions 自托管 runner）
       - 拉取仓库
       - 推断 Dockerfile（语言、包管理器）
       - 使用 Daytona / E2B 沙箱（目标运行时环境）
       - clone -> git worktree -> agent 分支
            |
            v
    mini-swe-agent / SWE-agent v2 循环
       Claude Opus 4.7 或 GPT-5.4-Codex
       工具：ripgrep、tree-sitter、read/edit、run_tests、git
            |
            v
    验证沙箱内 CI 通过 + 覆盖率差异检查
            |
            v （已验证）
    git push + 通过 GitHub App 打开 PR
       PR 正文 = 理由说明 + diff 摘要 + 追踪 URL
       标签：needs-review
            |
            v
    操作员审查；可 @提及代理以跟进
```

## 技术栈

- 触发器：具有细粒度令牌的 GitHub App；通过 Lambda 或 Fly.io 接收 webhook
- 工作节点：ECS Fargate 任务（或 GitHub Actions 自托管 runner）
- 沙箱：每个任务的 Daytona devcontainer 或 E2B 沙箱
- 代理循环：基于 Claude Opus 4.7 / GPT-5.4-Codex 的 mini-swe-agent 基线或 SWE-agent v2
- 检索：tree-sitter 仓库地图 + ripgrep
- 验证：沙箱内完整 CI + 覆盖率差异门禁
- 可观测性：Langfuse 按 PR 归档追踪记录，链接嵌入 PR 正文
- 预算：每仓库每日美元上限；每仓库每日最大 PR 数

```figure
cf-issue-to-pr
```

## 构建步骤

1. **GitHub App。** 细粒度安装令牌：issues read+write、pull_requests write、contents read+write、workflows read。分支保护（唯一能实现此功能的层面）强制执行"禁止直接推送到 `main`"和"禁止强制推送"；该 App 不在豁免列表中。工作节点通过将白名单检查应用于拟议的 diff 来强制执行"禁止写入 `.github/workflows` 之下"，因为 GitHub App 权限不支持路径级限定。

2. **Webhook 接收器。** Lambda 函数接受 Issue 标签 / PR 评论 webhook。按标签 `@agent fix this` 过滤。入队至 SQS。

3. **调度器。** 从 SQS 弹出任务。执行每仓库每日预算。启动带有仓库 URL、Issue 正文以及全新 Daytona 沙箱的 ECS Fargate 任务。

4. **环境推断。** 检测语言（Python、Node、Go、Rust）和包管理器（uv、pnpm、go mod、cargo）。若不存在 Dockerfile，则动态生成。

5. **代理循环。** mini-swe-agent 或 SWE-agent v2 配合 Claude Opus 4.7。工具：ripgrep、tree-sitter 仓库地图、read_file、edit_file、run_tests、git。硬限制：成本上限 $20、墙钟时间 30 分钟、代理轮次 30 轮。

6. **验证。** 循环结束后，在沙箱内运行完整测试套件。通过 jacoco / coverage.py 计算覆盖率差异。若 CI 失败：中止，不打开 PR。若覆盖率下降超过 2%：打开 PR 并标注 `needs-review`。

7. **PR 发布。** 推送代理分支。通过 GitHub API 打开 PR，内容包括：标题、理由说明、diff 摘要、追踪 URL、成本、轮次。

8. **凭证卫生。** 工作节点使用短期的 GitHub App 安装令牌运行。日志在归档前清除敏感信息。

9. **评估。** 30 个不同难度的 seeded 内部 Issue。衡量通过率、PR 质量（diff 大小、风格、覆盖率）、成本、延迟。在同一批 Issue 上与 Cursor Background Agents 和 AWS Remote SWE Agents 对比。

## 使用方式

```
# on github.com
  - 用户在 Issue #842 上打上标签 `@agent fix this`
  - 14 分钟后出现 PR #1903
  - 正文：
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## 交付物

`outputs/skill-issue-to-pr.md` 为交付物。一个 GitHub App + 异步云端工作节点，将带标签的 Issue 转化为预算受限、凭证作用域限定的可审查 PR。

| 权重 | 准则 | 衡量方式 |
|:-:|---|---|
| 25 | 30 个 Issue 的通过率 | 端到端成功（CI 绿色 + 覆盖率 OK） |
| 20 | PR 质量 | Diff 大小、覆盖率差异、风格合规性 |
| 20 | 每个已解决 Issue 的成本与延迟 | 每个 PR 的美元与墙钟时间 |
| 20 | 安全性 | 作用域限定令牌、每仓库预算、禁止强制推送、凭证卫生 |
| 15 | 操作员 UX | 理由说明、重试便利、@提及跟进 |
| **100** | | |

## 练习

1. 添加"修复不稳定测试"模式：标签 `@agent stabilize-flake TestX` 在沙箱内运行测试 50 次，并提出最小化变更以稳定它。

2. 在三个共享 Issue 上对比成本与 Cursor Background Agents。报告哪些工具在哪些场景占优。

3. 实现预算仪表板：每仓库每日成本、每用户成本。对异常告警。

4. 构建"试运行"模式：在不运行 CI 的情况下打开草稿 PR，使审查者能以低成本检视方案。

5. 添加保留策略：超过 7 天未合并的 PR 分支自动删除。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| GitHub App | "作用域限定的机器人身份" | 具有细粒度权限 + 短期安装令牌的 App |
| 异步云端代理 | "后台代理" | 在云端沙箱中运行的非交互式工作节点，而非终端 |
| 环境推断 | "Dockerfile 合成" | 检测语言 + 包管理器，若缺失则生成 Dockerfile |
| 验证 | "沙箱内 CI" | 在打开 PR 之前在工作节点内运行完整测试套件 |
| 覆盖率差异 | "覆盖率保持" | 从基准分支到代理分支的测试覆盖率变化百分比 |
| 每仓库预算 | "每日上限" | 在调度器端强制执行的美元与 PR 数量上限 |
| 理由说明 | "PR 正文解释" | 代理对变更内容与原因的摘要；要求在 PR 正文中提供 |

## 延伸阅读

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) —— 异步云端代理的参考实现
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) —— CLI 参考
- [Cursor Background Agents](https://docs.cursor.com/background-agent) —— 商业替代方案
- [OpenAI Codex (cloud)](https://openai.com/codex) —— 托管竞品
- [Google Jules](https://jules.google) —— Google 的托管版本
- [Factory Droids](https://www.factory.ai) —— 另一款商业参考
- [GitHub App documentation](https://docs.github.com/en/apps) —— 作用域限定的机器人身份
- [Daytona cloud sandboxes](https://daytona.io) —— 参考沙箱
