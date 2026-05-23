# 毕业项目 09 — 代码迁移智能体（仓库级别语言/运行时升级）

> 亚马逊的 MigrationBench（Java 8 到 17）和谷歌的 App Engine Py2-to-Py3 迁移器设定了 2026 年的标杆。Moderne 的 OpenRewrite 可进行大规模确定性 AST 重写。Grit 以 codemod 风格的 DSL 针对相同问题。生产模式结合了两者：用于安全重写的确定性底层加上处理模糊情况的智能体层、用于每个分支构建的沙箱，以及在 PR 打开前确保测试通过的测试工具。本毕业项目的目标是迁移 50 个真实仓库，并发布通过率及失败分类。

**类型:** 毕业项目
**语言:** Python (智能体), Java / Python (目标), TypeScript (仪表板)
**先修课程:** 阶段 5 (NLP), 阶段 7 (transformers), 阶段 11 (LLM 工程), 阶段 13 (工具), 阶段 14 (智能体), 阶段 15 (自主性), 阶段 17 (基础设施)
**锻炼阶段:** P5 · P7 · P11 · P13 · P14 · P15 · P17
**时间:** 30 小时

## 问题

大规模代码迁移是 2026 年编码智能体最清晰的生产应用之一。真实情况显而易见（迁移后测试套件是否通过？），回报切实（Java-8 集群迁移是需多人完成的项目），且基准公开（MigrationBench 的 50 个仓库子集）。Moderne 的 OpenRewrite 处理确定性部分。智能体层处理 OpenRewrite 食谱无法处理的一切：模糊重写、构建系统漂移、长尾语法问题、传递性依赖断裂。

你将构建一个智能体，接收一个 Java 8 仓库（或 Python 2 仓库）并生成一个 CI 测试全绿的迁移分支。你将衡量通过率、测试覆盖率保持率、每个仓库的成本，并构建一个失败分类。与仅使用确定性方法的基线进行对比，可以明确智能体价值的真正所在。

## 概念

该流水线有两层。**确定性底层**（Java 用 OpenRewrite，Python 用 libcst）安全地执行大部分机械重写：导入、方法签名、空值安全编辑、try-with-resources、已弃用 API 替换。它速度快且生成可审计的差异。**智能体层**（OpenAI Agents SDK 或基于 Claude Opus 4.7 和 GPT-5.4-Codex 的 LangGraph）处理食谱无法解决的情况：构建文件升级（Maven/Gradle/pyproject）、传递性依赖冲突、测试不稳定、自定义注解。

每个仓库获得一个预装了目标运行时的 Daytona 沙箱。智能体进行迭代：运行构建，分类失败，应用修复，重新运行。硬性限制：每个仓库 30 分钟，$8 成本，20 次智能体交互轮次。如果所有测试通过且覆盖率未降低，则分支会开启一个 PR。如果未通过，仓库将被归入一个失败类别并附上证据。

失败分类是交付物。在 50 个仓库中，什么导致了问题？传递性依赖？自定义注解？构建工具版本？与迁移无关的测试不稳定？每个类别都有计数和示例差异。未来的食谱作者可以针对前三大类。

## 架构

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## 技术栈

- 确定性底层：OpenRewrite (Java) 或 libcst (Python)
- 智能体：OpenAI Agents SDK 或基于 Claude Opus 4.7 + GPT-5.4-Codex 的 LangGraph
- 沙箱：每个分支的 Daytona devcontainers，预装目标运行时 (Java 17 / Python 3.12)
- 构建系统：Maven, Gradle, uv (Python)
- 基准：亚马逊 MigrationBench 50 个仓库子集 (Java 8 到 17)，谷歌 App Engine Py2 到 Py3 仓库
- 测试工具：并行运行器，通过 Jacoco (Java) 或 coverage.py (Python) 进行覆盖率测试
- 可观测性：Langfuse + 每个仓库的追踪包，包含每个差异块
- 仪表板：失败分类仪表板，包含每个类别的计数和示例差异

## 构建步骤

1. **食谱遍历。** 首先运行 OpenRewrite (Java) 或 libcst (Python) 食谱。捕获 70-80% 的机械迁移。提交为“recipe”提交。

2. **构建尝试。** Daytona 沙箱：安装目标运行时，运行构建。如果通过，跳转到测试。如果失败，交给智能体处理。

3. **智能体循环。** 使用带工具的 LangGraph：`run_build`, `read_file`, `edit_file`, `run_test`, `git_diff`。智能体对失败进行分类（依赖、语法、测试、构建工具）并应用针对性修复。重新运行。

4. **预算上限。** 每个仓库 30 分钟时间限制，$8 成本，20 次智能体交互。任何超限都会停止并归档到 "budget_exhausted"，附带当前差异。

5. **测试 + 覆盖率门槛。** 构建通过后，运行测试套件。与基础仓库比较覆盖率。如果覆盖率下降超过 2%，则归档到 "coverage_regression"。

6. **开启 PR。** 成功时，推送分支，开启 PR，附带差异以及应用了哪些食谱和智能体编写了哪些提交的摘要。

7. **失败分类。** 对于每个失败的仓库，标记一个类别：`dep_upgrade_required`, `build_tool_drift`, `custom_annotation`, `test_flake`, `syntax_edge_case`, `budget_exhausted`。构建仪表板。

8. **50 个仓库运行。** 在 MigrationBench 子集上执行。报告每个类别的通过率、每个仓库的成本、覆盖率保持率，以及与仅使用确定性方法的基线对比。

## 使用它

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## 交付它

`outputs/skill-migration-agent.md` 是交付物。给定一个仓库，它执行确定性食谱，然后运行智能体循环以生成一个测试全绿的迁移分支，或将仓库归档到一个分类类别。

| 权重 | 准则 | 如何衡量 |
|:-:|---|---|
| 25 | MigrationBench 通过率 | 50 个仓库子集 pass@1 |
| 20 | 测试覆盖率保持率 | 与基准相比的平均覆盖率增量 |
| 20 | 每个迁移仓库的成本 | 通过运行的 $/仓库 |
| 20 | 智能体/确定性工具集成 | OpenRewrite 处理与智能体编写的修复比例 |
| 15 | 失败分析报告 | 含示例的分类完整性 |
| **100** | | |

## 练习

1. 仅使用 OpenRewrite（无智能体）运行迁移流水线。将通过率与完整流水线进行对比。识别出仅由智能体带来差异的情况。

2. 实现一个“lint-clean”检查：迁移后，运行样式检查工具（Java 用 spotless，Python 用 ruff）。如果出现新的 lint 错误，则 PR 失败。衡量覆盖率保持但样式回归率。

3. 添加一个“最小差异”优化器：在智能体的分支通过测试后，用第二遍修剪不必要的更改。报告差异大小缩减情况。

4. 扩展到第三次迁移：Node 18 到 Node 22。复用沙箱包装；为自定义 codemod 替换食谱层。

5. 衡量首次构建通过时间（TTFGB）作为用户体验指标。目标：p50 低于 10 分钟。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|------------|----------|
| 确定性底层 | “食谱引擎” | OpenRewrite / libcst: 具有安全保证的声明式 AST 重写 |
| Codemod | “代码修改程序” | 机械性地更改源代码的重写规则 |
| 构建漂移 | “工具版本差异” | 主版本之间 Maven / Gradle / uv 行为的微妙变化 |
| 失败类别 | “分类桶” | 仓库未能迁移的标记原因：依赖、语法、测试、构建工具、预算 |
| 覆盖率增量 | “覆盖率保持率” | 从基础分支到迁移分支测试覆盖率百分比的变化 |
| 智能体轮次 | “工具调用轮次” | 智能体循环中的一次计划->行动->观察循环 |
| 预算耗尽 | “达到上限” | 仓库耗尽其 30 分钟 / $8 / 20 轮限制而未通过 |

## 延伸阅读

- [亚马逊 MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 2026 年的权威基准
- [Moderne.io OpenRewrite 平台](https://www.moderne.io) — 确定性底层参考
- [OpenRewrite 文档](https://docs.openrewrite.org) — 食谱编写
- [Grit.io](https://www.grit.io) — 可选的 codemod DSL
- [OpenAI 沙箱化迁移食谱](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK 参考
- [谷歌 App Engine Py2 到 Py3 迁移器](https://cloud.google.com/appengine) — 可选迁移基准
- [libcst](https://github.com/Instagram/LibCST) — Python 确定性底层
- [Daytona 沙箱](https://daytona.io) — 每个分支沙箱的参考