# Capstone 10 — 多智能体软件工程团队

> 2026 年多智能体工程团队的形态已趋于定型：一个架构师负责规划，N 个编码者在并行工作树中工作，一个审查者把关，一个测试者验证。SWE-AF 的工厂架构、MetaGPT 的角色提示、AutoGen 0.4 的有类型 actor 图、Cognition 的 Devin，以及 Factory 的 Droids，都独立地得出了相同的形态。并行工作树将墙钟时间转化为吞吐量。共享状态和交接协议成为故障面。Capstone 目标是构建这个团队，在 SWE-bench Pro 上评估，并报告哪些交接会失败以及失败频率。

**类型：** Capstone
**语言：** Python / TypeScript（智能体），Shell（工作树脚本）
**前置条件：** Phase 11（LLM 工程）、Phase 13（工具）、Phase 14（智能体）、Phase 15（自主）、Phase 16（多智能体）、Phase 17（基础设施）
**涉及的阶段：** P11 · P13 · P14 · P15 · P16 · P17
**时间：** 40 小时

## 问题

单智能体编码平台在处理大型任务时遇到瓶颈。不是因为任何单个智能体能力不足，而是因为 200k token 的上下文无法同时容纳架构计划、四个并行代码库切片、审查者评论以及测试输出。多智能体工厂拆分了问题：架构师拥有计划，编码者在线并行工作树中负责实现，审查者把关，测试者验证。SWE-AF 的"工厂"架构、MetaGPT 的角色、AutoGen 的有类型 actor 图——这三种表述描述的是同一种形态。

故障面在于交接环节。架构师制定了编码者无法实现的计划。编码者产生冲突的 diff。审查者批准了一个幻觉修复。测试者与正在写入的编码者发生竞争。你将构建这样一个团队，在 50 个 SWE-bench Pro 问题上运行它，跟踪每一次交接，并发布事后复盘报告。

## 概念

角色是有类型的智能体。**架构师**（Claude Opus 4.7）读取问题，编写计划，并将其拆分为具有显式接口的子任务。**编码者**（Claude Sonnet 4.7，N 个并行实例，每个位于 `git worktree` + Daytona 沙箱中）独立实现子任务。**审查者**（GPT-5.4）读取合并后的 diff，批准或要求具体修改。**测试者**（Gemini 2.5 Pro）在隔离环境中运行测试套件，报告通过/失败及产物。

通信通过共享任务板（文件或 Redis 实现）进行。每个角色消费其有权处理的子任务。交接是 A2A 协议类型的消息。协调问题：合并冲突解决（协调员角色或自动三路合并）、共享状态同步（计划一旦编码者开始就冻结；重新规划是独立事件）、以及审查者把关（审查者不能批准自己提交的更改或自己提出的更改）。

Token 放大是隐藏成本。每个角色边界都增加了摘要提示和交接上下文。一次 40 轮次的单智能体运行变成四个角色的总共 160 轮次。评分标准特别衡量 token 效率与单智能体基准的比较，因为问题不是"多智能体是否有效"，而是"它是否以更低的每美元成本取胜"。

## 架构

```
GitHub issue URL
      |
      v
架构师 (Opus 4.7)
   读取问题，生成包含子任务+接口的计划
      |
      v
任务板（文件 / Redis）
      |
   +-- 子任务 1 ---+-- 子任务 2 ---+-- 子任务 3 ---+-- 子任务 4 ---+
   v                v                v                v                v
编码者 A         编码者 B         编码者 C         编码者 D          （4 个并行）
 (Sonnet)        (Sonnet)        (Sonnet)        (Sonnet)
 worktree A      worktree B      worktree C      worktree D
 Daytona         Daytona         Daytona         Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           合并协调员（三路合并 + 冲突解决）
               |
               v
           审查者 (GPT-5.4)
               |
               v
           测试者  (Gemini 2.5 Pro)  -> 通过？-> 提交 PR
                                     -> 失败？-> 路由回编码者
```

## 技术栈

- 编排：LangGraph，共享状态 + 每智能体子图
- 消息：A2A 协议（Google 2025），用于有类型智能体间消息
- 模型：Opus 4.7（架构师）、Sonnet 4.7（编码者）、GPT-5.4（审查者）、Gemini 2.5 Pro（测试者）
- 工作树隔离：每个编码者使用 `git worktree add` + Daytona 沙箱
- 合并协调员：自定义三路合并 + LLM 调解的冲突解决
- 评估：SWE-bench Pro（50 个问题）、SWE-AF 场景、HumanEval++（单元测试）
- 可观测性：Langfuse，带角色标签的 spans，每智能体 token 核算
- 部署：K8s，每个角色作为独立的 Deployment + HPA 基于 backlog

```figure
ce-team-handoff
```

## 构建指南

1. **任务板。** 基于文件的 JSONL，包含有类型消息：`plan_request`、`subtask`、`diff_ready`、`review_needed`、`test_needed`、`approved`、`rejected`、`replan_needed`。智能体订阅对应标签。

2. **架构师。** 读取 GitHub issue，使用计划模板运行 Opus 4.7，要求显式子任务接口（修改的文件、公开函数、测试影响）。输出一个包含子任务 DAG 的 `plan_request`。

3. **编码者。** N 个并行工作者，每个从任务板认领一个子任务。每个工作者启动一个新的 `git worktree add` 分支和 Daytona 沙箱。实现子任务后，输出包含补丁 + 测试变更的 `diff_ready`。

4. **合并协调员。** 所有编码者完成后，三路合并 N 个分支到暂存分支。仅在文件级重叠时进行 LLM 调解的冲突解决。

5. **审查者。** GPT-5.4 读取合并后的 diff。不能批准自己编写的 diff。输出 `approved`（无操作）或 `review_feedback`，带有具体变更请求，路由回相关编码者。

6. **测试者。** Gemini 2.5 Pro 在干净沙箱中运行测试套件。捕获产物。输出 `test_passed` 或 `test_failed`（附带堆栈跟踪）。失败的测试循环回拥有该失败子任务的编码者。

7. **交接核算。** 每次跨越角色边界的消息都在 Langfuse 中记录一个 span，包含负载大小和使用的模型。计算每个子任务的 token 放大率（coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens）。

8. **评估。** 在 50 个 SWE-bench Pro 问题上运行。比较 pass@1 和每个解决问题的成本，与单智能体基准（一个 Sonnet 4.7 在单个工作树中）进行对比。

9. **事后复盘。** 对于每个失败的问题，识别出故障的交接点（计划过于模糊、合并冲突、审查者误批准、测试偶然失败）。生成交接失败直方图。

## 使用方式

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## 交付物

`outputs/skill-multi-agent-team.md` 是交付文件。给定一个 issue URL 和并行度，团队生成一个可合并的 PR 并附带每角色 token 核算。

| 权重 | 标准 | 测量方法 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 匹配 50 问题子集，pass@1 |
| 20 | 并行加速比 | 墙钟时间 vs 单智能体基准 |
| 20 | 审查质量 | 注入 bug 探针中的误批准率 |
| 20 | Token 效率 | 每个解决问题的总 token 数 vs 单智能体 |
| 15 | 协调工程 | 合并冲突解决、交接失败直方图 |
| **100** | | |

## 练习

1. 在运行中途向 diff 中注入一个明显的 bug（在主函数体前添加 `return None`）。测量审查者的误批准率。调整审查者提示词，使误批准率低于 5%。

2. 减少为两个编码者（架构师 + 编码者 + 审查者 + 测试者，编码者顺序执行两个子任务）。比较墙钟时间和通过率。

3. 用单写者约束替换合并协调员（子任务触碰不相交的文件集）。测量架构师的规划负担。

4. 将审查者从 GPT-5.4 换成 Claude Opus 4.7。测量误批准率和 token 成本差异。

5. 添加第五个角色：文档员（Haiku 4.5）。审查后生成 changelog 条目。测量文档质量是否值得额外的 token 开销。

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|------|-----------------|------------------------|
| 并行工作树 | "隔离分支" | `git worktree add` 为每个编码者生成独立的 working tree |
| 任务板 | "共享消息总线" | 文件或 Redis 存储的有类型消息，智能体可订阅 |
| 交接 | "角色边界" | 任何从一个角色上下文跨越到另一个角色上下文的message |
| Token 放大 | "多智能体开销" | 所有角色总 token 数 / 单智能体完成同一任务的 token 数 |
| A2A 协议 | "智能体对智能体" | Google 2025 年的有类型智能体间消息规范 |
| 合并协调员 | "集成器" | 运行三路合并并调解冲突的组件 |
| 误批准 | "审查者幻觉" | 审查者批准了一个含有已知 bug 的 diff |

## 延伸阅读

- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF) — 2026 年多智能体工厂的参考实现
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — 基于角色的多智能体框架
- [AutoGen v0.4](https://github.com/microsoft/autogen) — Microsoft 的有类型 actor 框架
- [Cognition AI (Devin)](https://cognition.ai) — 参考产品
- [Factory Droids](https://www.factory.ai) — 备选参考产品
- [Google A2A protocol](https://a2a-protocol.org/latest/) — 智能体间消息规范
- [git worktree documentation](https://git-scm.com/docs/git-worktree) — 隔离基础
- [SWE-bench Pro](https://www.swebench.com) — 评估目标
