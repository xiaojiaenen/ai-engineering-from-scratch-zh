# Capstone 09 — 代码迁移智能体（仓库级语言/运行时升级）

> Amazon 的 MigrationBench（Java 8 到 17）和 Google 的 App Engine Py2-to-Py3 迁移工具设定了 2026 年的标杆。Moderne 的 OpenRewrite 以规模化确定性 AST 重写著称。Grit 用 codemod 风格 DSL 解决同类问题。生产模式结合两者：一个用于安全重写的确定性基础层，加上一个处理模糊场景的智能体层，配合每个分支的构建沙箱，以及 PR 开启前测试就会变绿（通过）的测试框架。本 Capstone 要求迁移 50 个真实仓库，并公布通过率与失败分类学。

**类型：** Capstone
**语言：** Python（智能体）、Java / Python（目标语言）、TypeScript（仪表板）
**先决条件：** Phase 5（NLP）、Phase 7（transformers）、Phase 11（LLM engineering）、Phase 13（tools）、Phase 14（agents）、Phase 15（autonomous）、Phase 17（infrastructure）
**涉及的阶段：** P5 · P7 · P11 · P13 · P14 · P15 · P17
**耗时：** 30 小时

## Problem

大规模代码迁移是 2026 年编码智能体最清晰的生产应用场景之一。评判标准显而易见（迁移后测试套件是否通过？），回报切实可感（一套 Java 8 机群迁移是涉及人头规模的项目），且基准公开（MigrationBench 50 仓库子集）。Moderne 的 OpenRewrite 处理确定性部分。智能体层负责 OpenRewrite recipe 无法处理的所有内容：模糊重写、构建系统漂移、长尾语法、传递依赖断裂。

你将构建一个智能体，接收一个 Java 8 仓库（或 Python 2 仓库），输出一个 CI 全绿的迁移分支。你将测量通过率、测试覆盖率保持情况、单个仓库的成本，并构建失败分类学。与仅确定性方法的基线进行并列对比，能告诉你智能体的真实价值所在。

## Concept

流水线分为两层。**确定性基础层**（Java 用 OpenRewrite，Python 用 libcst）安全地执行大部分机械性重写：import 语句、方法签名、null 安全改造、try-with-resources、废弃 API 替换。它速度快且产生的 diff 可审计。**智能体层**（基于 Claude Opus 4.7 和 GPT-5.4-Codex 的 OpenAI Agents SDK 或 LangGraph）处理 recipe 无法覆盖的场景：构建文件升级（Maven/Gradle/pyproject）、传递依赖冲突、测试偶发失败、自定义注解。

每个仓库会获得一个预装目标运行时的 Daytona 沙箱。智能体循环迭代：运行构建 → 分类失败 → 应用修复 → 重新运行。硬性限制：每个仓库 30 分钟、$8、20 轮智能体交互。若所有测试通过且覆盖率变化不为负，该分支会创建 PR。否则，仓库将带着证据归入某个失败分类。

失败分类学是核心交付物。在 50 个仓库中，什么会出错？传递依赖？自定义注解？构建工具版本？与迁移无关的测试偶发失败？每个分类会有计数和示例 diff。未来的 recipe 作者可以据此优先攻克前三类。

## Architecture

```
目标仓库
      |
      v
OpenRewrite / libcst 确定性 recipes
   （安全、快速、可审计，覆盖约 70-80% 的修复）
      |
      v
按分支的 Daytona 沙箱
      |
      v
智能体循环（Claude Opus 4.7 / GPT-5.4-Codex）：
   - 运行构建 → 捕获失败信息
   - 分类失败（构建、测试、lint）
   - 应用修复（补丁或重试 recipe）
   - 重新运行
   - 预算限制：30 分钟、$8、20 轮
      |
      v
测试 + 覆盖率变化门控
      |
      v （通过）
创建 PR
      |
      v （失败）
归入失败分类 + 附加复现证据
```

## Stack

- 确定性基础层：OpenRewrite（Java）或 libcst（Python）
- 智能体：基于 Claude Opus 4.7 + GPT-5.4-Codex 的 OpenAI Agents SDK 或 LangGraph
- 沙箱：按分支的 Daytona devcontainer，预装目标运行时（Java 17 / Python 3.12）
- 构建系统：Maven、Gradle、uv（Python）
- 基准测试：Amazon MigrationBench 50 仓库子集（Java 8 到 17）、Google App Engine Py2-to-Py3 仓库
- 测试框架：并行运行器，通过 Jacoco（Java）或 coverage.py（Python）采集覆盖率
- 可观测性：Langfuse + 每个仓库附带包含所有 diff 片段的 trace bundle
- 仪表板：失败分类学仪表板，含各分类计数与示例 diff

```figure
ce-migration-funnel
```

## Build It

1. **Recipe 扫描。** 首先运行 OpenRewrite（Java）或 libcst（Python）的 recipe。捕获占 70-80% 的机械性迁移。提交为 "recipe" 提交。

2. **构建试跑。** Daytona 沙箱：安装目标运行时，运行构建。若全绿，跳过到测试；若失败，移交智能体。

3. **智能体循环。** 使用带有 `run_build`、`read_file`、`edit_file`、`run_test`、`git_diff` 工具的 LangGraph。智能体对失败进行分类（依赖、语法、测试、构建工具）并应用针对性修复。重新运行。

4. **预算上限。** 每个仓库 30 分钟实际运行时间、$8 成本、20 轮智能体交互。任何越界都会暂停，并将当前 diff 归入 "budget_exhausted"。

5. **测试与覆盖率门控。** 构建全绿后，运行测试套件。将覆盖率与基础仓库对比。若下降超过 2%，归入 "coverage_regression"。

6. **创建 PR。** 成功时，推送分支，创建 PR 并附上 diff，以及已应用的 recipe 和智能体生成的提交摘要。

7. **失败分类学。** 对每个失败的仓库打上分类标签：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。构建仪表板。

8. **50 仓库执行。** 在 MigrationBench 子集上运行。报告各类别的通过率、单仓库成本、覆盖率保持情况，并与仅确定性方法的基线进行对比。

## Use It

```bash
$ migrate legacy-java-service --target java17
[recipe]   已应用 27 处重写（JUnit 4->5、HashMap 初始化器、try-with-resources）
[build]    失败：找不到符号 sun.misc.BASE64Encoder
[agent]    第 1 轮分类：removed_jdk_api
[agent]    第 2 轮应用：sun.misc.BASE64Encoder -> java.util.Base64
[build]    通过
[tests]    412/412 通过；覆盖率 84.1% -> 84.3%
[pr]       已创建 #1841  成本=$3.20  轮次=4
```

## Ship It

``outputs/skill-migration-agent.md`` 是交付物。给定一个仓库，它会执行确定性 recipe，然后运行智能体循环，以生成 CI 全绿的迁移分支；若失败，则将仓库归入某个分类。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | MigrationBench 通过率 | 50 仓库子集的 pass@1 |
| 20 | 测试覆盖率保持 | 相对基础仓库的平均覆盖率变化 |
| 20 | 单迁移仓库成本 | 成功运行的 $/仓库 |
| 20 | 智能体/确定性工具集成 | OpenRewrite 处理 vs 智能体撰写的修复比例 |
| 15 | 失败分析报告 | 带示例的分类学完整性 |
| **100** | | |

## Exercises

1. 仅使用 OpenRewrite 运行迁移流水线（无智能体）。将通过率与完整流水线对比。找出仅靠智能体才能解决的用例。

2. 实现“lint 清洁”检查：迁移后运行风格 linter（Java 用 spotless，Python 用 ruff）。若出现新的 lint 错误则拒绝 PR。统计覆盖率保持但风格回退的比例。

3. 增加“最小化 diff”优化器：智能体分支通过测试后，进行第二轮处理以剔除无关变更。报告 diff 体积缩减比例。

4. 扩展至第三种迁移：Node 18 到 Node 22。复用沙箱封装；将 recipe 层替换为自定义 codemod。

5. 将首次构建全绿时间（TTFGB）作为 UX 指标进行测量。目标：p50 低于 10 分钟。

## Key Terms

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Deterministic substrate | “Recipe 引擎” | OpenRewrite / libcst：带安全性保证的声明式 AST 重写 |
| Codemod | “修改代码的程序” | 机械性改变源代码的重写规则 |
| Build drift | “工具版本偏差” | Maven / Gradle / uv 在大版本间微妙的行为差异 |
| Failure class | “分类桶” | 仓库未迁移的标注原因：依赖、语法、测试、构建工具、预算 |
| Coverage delta | “覆盖率保持” | 从基础分支到迁移分支的测试覆盖率 % 变化 |
| Agent turn | “工具调用轮次” | 智能体循环中的一次“规划 → 执行 → 观察”周期 |
| Budget exhaustion | “触及上限” | 仓库在达到 30 分钟 / $8 / 20 轮限制前仍未通过 |

## Further Reading

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 2026 年的权威基准
- [Moderne.io OpenRewrite platform](https://www.moderne.io) — 确定性基础层参考
- [OpenRewrite documentation](https://docs.openrewrite.org) — recipe 编写指南
- [Grit.io](https://www.grit.io) — 另一种 codemod DSL
- [OpenAI sandboxed migration cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK 参考
- [Google App Engine Py2 to Py3 migrator](https://cloud.google.com/appengine) — 另一种迁移基准
- [libcst](https://github.com/Instagram/LibCST) — Python 确定性基础层
- [Daytona sandboxes](https://daytona.io) — 按分支的沙箱参考
