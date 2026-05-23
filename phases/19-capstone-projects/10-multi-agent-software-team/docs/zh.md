# 毕业项目 10 —— 多智能体软件工程团队

> SWE-AF 的工厂架构、MetaGPT 的基于角色的提示、AutoGen 0.4 的类型化执行器图、Cognition 的 Devin 以及 Factory 的 Droids 都在 2026 年的形态上趋于一致：一个架构师负责规划，N 个编码员在并行工作树中工作，一个审查员负责把关，一个测试员负责验证。并行工作树将挂钟时间转化为吞吐量。共享状态和交接协议成为故障面。本毕业项目的目标是构建这样一个团队，在 SWE-bench Pro 上进行评估，并报告哪些交接环节会失败以及失败频率。

**类型：** 毕业项目
**语言：** Python / TypeScript（智能体），Shell（工作树脚本）
**前置要求：** 第 11 阶段（LLM 工程），第 13 阶段（工具），第 14 阶段（智能体），第 15 阶段（自主），第 16 阶段（多智能体），第 17 阶段（基础设施）
**涉及阶段：** P11 · P13 · P14 · P15 · P16 · P17
**时间：** 40 小时

## 问题

单智能体编码工具在处理大型任务时会遇到瓶颈。这不是因为任何单个智能体能力弱，而是因为 20 万 token 的上下文无法同时容纳一个架构计划、四个并行的代码库切片、审查意见以及测试输出。多智能体工厂将问题拆解：一个架构师负责计划，多个编码员在并行工作树中独立实现，一个审查员负责把关，一个测试员负责验证。SWE-AF 的 “工厂” 架构、MetaGPT 的角色、AutoGen 的类型化执行器图——这三种框架描述的是同一种形态。

故障面在于交接。架构师规划了编码员无法实现的内容。编码员产生了相互冲突的差异。审查员批准了一个幻觉修复。测试员与仍在编码的编码员竞争资源。你将构建这样一个团队，在 50 个 SWE-bench Pro 问题上运行它，跟踪每一次交接，并发布事后分析报告。

## 概念

角色是类型化的智能体。**架构师**（Claude Opus 4.7）阅读问题，撰写计划，并将其分解为带有明确接口的子任务。**编码员**（Claude Sonnet 4.7，N 个并行实例，每个都在 `git worktree` + Daytona 沙箱中）独立实现子任务。**审查员**（GPT-5.4）阅读合并后的差异，要么批准，要么要求进行特定修改。**测试员**（Gemini 2.5 Pro）在隔离环境中运行测试套件，并报告通过/失败状态及产物。

沟通通过一个共享任务板（文件支持或 Redis）进行。每个角色处理其被授权处理的任务。交接是 A2A 协议类型的消息。协调关注点包括：合并冲突解决（协调员角色或自动三方合并）、共享状态同步（计划在编码员开始后即冻结；重新规划是单独的事件），以及审查员把关（审查员不能批准自己提出的变更）。

Token 放大是隐藏的成本。每个角色边界都会增加摘要提示和交接上下文。一个 40 轮次的单智能体运行，在四个角色之间会变成总共 160 个轮次。评分标准特别权衡了 token 效率与单智能体基线的对比，因为问题不在于 “多智能体是否有效”，而在于 “它是否按成本效益取胜”。

## 架构

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## 技术栈

- 编排：LangGraph，共享状态 + 每个智能体的子图
- 消息传递：A2A 协议（Google 2025），用于类型化的智能体间消息
- 模型：Opus 4.7（架构师），Sonnet 4.7（编码员），GPT-5.4（审查员），Gemini 2.5 Pro（测试员）
- 工作树隔离：每个编码员使用 `git worktree add` + Daytona 沙箱
- 合并协调员：自定义三方合并 + LLM 介导的冲突解决
- 评估：SWE-bench Pro（50 个问题），SWE-AF 场景，用于单元测试的 HumanEval++
- 可观测性：Langfuse，带角色标记的跨度，每个智能体的 token 计量
- 部署：K8s，每个角色作为单独的 Deployment + 基于待办事项的 HPA

## 构建步骤

1.  **任务板。** 基于文件的 JSONL，包含类型化消息：`plan_request`、`subtask`、`diff_ready`、`review_needed`、`test_needed`、`approved`、`rejected`、`replan_needed`。智能体订阅标签。

2.  **架构师。** 阅读 GitHub issue，使用 Opus 4.7 和一个计划模板运行，该模板要求明确的子任务接口（涉及的文件、公共函数、测试影响）。发出一个包含子任务有向无环图（DAG）的 `plan_request`。

3.  **编码员。** N 个并行工作者，每个从任务板认领一个子任务。每个启动一个新的 `git worktree add` 分支和一个 Daytona 沙箱。实现该子任务。发出包含补丁和测试增量的 `diff_ready`。

4.  **合并协调员。** 当所有编码员完成后，将 N 个分支三方合并到一个暂存分支。仅当文件级重叠存在时，才使用 LLM 介导的冲突解决。

5.  **审查员。** GPT-5.4 阅读合并后的差异。不能批准由它自己撰写的差异。发出 `approved`（空操作）或 `review_feedback`，其中包含路由回相关编码员的具体修改请求。

6.  **测试员。** Gemini 2.5 Pro 在一个干净的沙箱中运行测试套件。捕获产物。发出 `test_passed` 或 `test_failed`（附带堆栈跟踪）。失败的测试循环回拥有该失败子任务的编码员。

7.  **交接计量。** 每个跨越角色边界的消息在 Langfuse 中都有一个带有有效负载大小和使用模型的跨度。计算每个子任务的 token 放大率（编码员_token + 审查员_token + 测试员_token + 架构师_份额 / 编码员_token）。

8.  **评估。** 在 50 个 SWE-bench Pro 问题上运行。将 pass@1 和每解决一个问题的成本与单智能体基线（一个 Sonnet 4.7 在单个工作树中）进行比较。

9.  **事后分析。** 对于每个失败的问题，识别失败的交接环节（计划过于模糊、合并冲突、审查员错误批准、测试员不稳定）。生成一个交接失败直方图。

## 使用

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

## 交付

`outputs/skill-multi-agent-team.md` 是可交付成果。给定一个 issue URL 和并行度，该团队将生成一个可合并的 PR，并附上每个角色的 token 计量。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 匹配的 50 个问题子集，pass@1 |
| 20 | 并行加速 | 与单智能体基线的挂钟时间对比 |
| 20 | 审查质量 | 在注入的 bug 探测下的错误批准率 |
| 20 | Token 效率 | 每个解决的问题的总 token 数与单智能体对比 |
| 15 | 协调工程 | 合并冲突解决，交接失败直方图 |
| **100** | | |

## 练习

1.  在运行中途向一个差异中注入一个明显的 bug（在主体前添加额外的 `return None`）。衡量审查员的错误批准率。调整审查员提示直到错误批准率低于 5%。

2.  减少到两个编码员（架构师 + 编码员 + 审查员 + 测试员，编码员依次运行两个子任务）。比较挂钟时间和通过率。

3.  用单写入者约束替换合并协调员（子任务涉及不相交的文件集）。衡量架构师的规划负担。

4.  将审查员从 GPT-5.4 替换为 Claude Opus 4.7。衡量错误批准率和 token 成本差异。

5.  添加第五个角色：文档员（Haiku 4.5）。审查后，它生成一个变更日志条目。衡量文档质量是否证明额外的 token 开支是合理的。

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|-----------------|------------------------|
| 并行工作树 | “隔离的分支” | 为每个编码员产生新鲜工作目录的 `git worktree add` |
| 任务板 | “共享消息总线” | 智能体订阅的类型化消息的文件或 Redis 存储 |
| 交接 | “角色边界” | 任何从一个角色的上下文传递到另一个角色上下文的消息 |
| Token 放大 | “多智能体开销” | 跨角色的总 token 数 / 完成相同任务的单智能体 token 数 |
| A2A 协议 | “智能体到智能体” | Google 2025 年关于类型化智能体间消息的规范 |
| 合并协调员 | “集成器” | 运行三方合并并调解冲突的组件 |
| 错误批准 | “审查员幻觉” | 审查员批准了一个包含已知 bug 的差异 |

## 延伸阅读

- [SWE-AF 工厂架构](https://github.com/Agent-Field/SWE-AF) —— 2026 年多智能体工厂的参考实现
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) —— 基于角色的多智能体框架
- [AutoGen v0.4](https://github.com/microsoft/autogen) —— 微软的类型化执行器框架
- [Cognition AI (Devin)](https://cognition.ai) —— 参考产品
- [Factory Droids](https://www.factory.ai) —— 替代参考产品
- [Google A2A 协议](https://developers.google.com/agent-to-agent) —— 智能体间消息传递规范
- [git worktree 文档](https://git-scm.com/docs/git-worktree) —— 隔离基底
- [SWE-bench Pro](https://www.swebench.com) —— 评估目标