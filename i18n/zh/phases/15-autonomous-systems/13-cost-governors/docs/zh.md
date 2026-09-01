# Action Budgets, Iteration Caps, and Cost Governors

> 一个中型电商智能体团队启用了"订单追踪"技能后，月度 LLM 成本从 $1,200 飙升至 $4,800。这不是定价 bug，而是一个找到了新循环并持续在其中消耗资源的智能体。Microsoft 的 Agent Governance Toolkit（2026 年 4 月 2 日）将针对此类问题的防御手段规范化为：每次请求的 `max_tokens`、每个任务的 token 与美元预算、每日/每月上限、迭代上限、分层模型路由、prompt 缓存、上下文窗口压缩、昂贵操作的 HITL 检查点、预算超限时触发 kill switch。Anthropic 的 Claude Code Agent SDK 以不同名称提供了相同的原语。金融速度限制（例如：10 分钟内支出超过 $50 时切断访问）比月度上限更快地捕获循环问题。

**类型：** Learn
**语言：** Python（stdlib，分层 cost-governor 模拟器）
**前置条件：** Phase 15 · 10（权限模式）、Phase 15 · 12（持久化执行）
**时间：** 约 60 分钟

## The Problem

自主智能体每一步都要花费真金白银。聊天机器人输出错误结果只是一次糟糕的回复；智能体陷入错误循环则是一张账单。业界对这种故障模式的术语是"钱包耗尽"（Denial of Wallet）——智能体不断推理、不断调用工具、不断产生费用，而没有任何机制能阻止它，因为根本就没有设计这样的机制。

解决方案不是一个数字，而是一组在不同时间尺度和粒度上的限制栈：每次请求、每个任务、每小时、每天、每月。设计良好的限制栈能在几分钟内捕获失控循环，几小时内捕获缓慢泄漏，一天内捕获糟糕的发布。同样的限制栈能在智能体进行长周期自主运行时始终保持预算可控。

这是一堂工程课：数学很简单，真正的难点在于团队的纪律性。下面的限制项列表均命名于 Microsoft Agent Governance Toolkit 或 Anthropic Claude Code Agent SDK 文档中。

## The Concept

### 成本治理限制栈

1. **`max_tokens` 每次请求。** 简单直接。防止任何单次调用输出无界补全。
2. **每个任务的 token 预算。** 在整个运行期间不超过 N 个 token。达到上限后硬停止。
3. **每个任务的美元预算。** 与 token 预算同理，但以货币计。即 Claude Code 中的 `max_budget_usd`。
4. **每次工具调用上限。** 最多 N 次 `WebFetch` 调用、N 次 `shell_exec` 调用等。
5. **迭代上限（`max_turns`）。** 智能体总循环迭代次数；防止无限推理循环。
6. **每分钟/每小时/每天/每月上限。** 滚动窗口。在不同时间尺度上捕获泄漏。
7. **金融速度限制。** 例如："若 10 分钟内支出超过 $50，则切断访问。" 在月度上限触发前捕获基于循环的消耗。
8. **分层模型路由。** 默认使用较小模型；仅当分类器判定任务值得使用时才升级到较大模型。
9. **Prompt 缓存。** 系统 prompt 和稳定上下文存储在提供商缓存中；重新发送的 token 成本近乎为零。
10. **上下文窗口压缩。** 通过压缩/摘要将活跃上下文保持在阈值以下；直接降低 token 成本。
11. **昂贵操作的 HITL 检查点。** 对于已知昂贵的操作（长时间工具调用、大文件下载、高成本模型升级）之前，需要人工确认。
12. **预算超限时触发 kill switch。** 任何上限触发时会话中止。超限会被记录；需要独立的重启用路径。

### 为什么需要栈而不只是一个上限

单一月度上限只能在钱包被耗尽之后才捕获失控智能体。单一每次请求上限在会话层面则毫无作用。不同的故障模式需要不同的时间尺度：

- **失控循环**（智能体卡在 5 秒重试中）：由速度限制捕获。
- **缓慢泄漏**（智能体每任务执行约 2 倍预期工作量）：由每日上限捕获。
- **糟糕的发布**（新版本使用 5 倍 token）：由周/月度上限捕获。
- **正常需求激增**（真实需求，非 bug）：由小时/天级上限配合清晰日志捕获。

### Harness 预算表面

Claude Code Agent SDK 暴露了（公开文档）：

- `max_turns` — 迭代上限。
- `max_budget_usd` — 美元上限；超限时会话中止。
- `allowed_tools` / `disallowed_tools` — 工具白名单与黑名单。
- 工具使用前钩子，用于自定义成本核算。

结合权限模式阶梯（Lesson 10）。没有 `max_budget_usd` 的 `autoMode` 会话是无治理的自主性。Anthropic 明确指出 Auto Mode 需要预算控制；分类器与成本是正交的概念。

### EU AI Act、OWASP Agentic Top 10

Microsoft Agent Governance Toolkit 覆盖了 OWASP Agentic Top 10 和 EU AI Act 第 14 条（人类监督）的要求。在欧盟生产环境中，日志记录和上限执行不是可选的。

### 观察到的 $1,200 → $4,800 案例

Microsoft 文档中的真实案例：一个电商智能体在新增工具后月度成本翻了四倍。该工具允许智能体在每个会话中轮询订单状态。没有循环检测。没有每个工具的上下限。没有周环比增长告警。修复方案是增加每个工具的上下限加上每日增长告警。这是一个模板：每个新工具都是一个潜在的新循环；每个新工具都需要自己的上限和告警。

```figure
cost-governor-stack
```

## Use It

`code/main.py` 模拟了有无分层 cost-governor 限制栈的智能体运行。模拟智能体在若干回合后会漂移进轮询循环；分层限制栈会在速度窗口内捕获它，而单一月度上限要数天之后才会触发。

## Ship It

`outputs/skill-agent-budget-audit.md` 审计拟部署智能体的 cost-governor 限制栈并标记缺失的层级。

## Exercises

1. 运行 `code/main.py`。确认在轮询循环轨迹上速度限制在迭代上限之前触发。然后禁用速度限制，测量在迭代上限捕获之前智能体"花费"了多少。

2. 为一个浏览器智能体（Lesson 11）设计一套每个工具的上下限。哪个工具需要最严格的限制？哪个工具可以无上限运行而无风险？

3. 阅读 Microsoft Agent Governance Toolkit 文档。列出该工具命名中的所有上限类型。将每种类型映射到一个故障模式（失控循环、缓慢泄漏、糟糕发布、需求激增）。

4. 为一个实际任务（例如"对仓库中的 50 个问题进行分类"）估算通宵无人值守运行的成本。将 `max_budget_usd` 设为点估计的 2 倍。论证为何是 2 倍。

5. Claude Code 的 `max_budget_usd` 在会话累计成本超限时触发。设计一个你会在外部强制执行的可补充速度限制。什么会触发切断，重新启用是什么样的？

## Key Terms

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| Denial of Wallet | "失控账单" | 智能体循环产生支出且无上限能阻止它 |
| max_tokens | "每次请求上限" | 单次补全大小的天花板 |
| max_turns | "迭代上限" | 会话中智能体循环迭代的天花板 |
| max_budget_usd | "美元 kill switch" | 会话成本上限；超限时中止 |
| Velocity limit | "速率上限" | 短窗口内的支出限制（如 $50 / 10 分钟） |
| Tiered routing | "先小模型" | 便宜模型默认；仅当分类器判定合理时才升级 |
| Prompt caching | "缓存的系统 prompt" | 提供商侧缓存使重新发送的 token 成本降至近乎为零 |
| HITL checkpoint | "人工审批门" | 昂贵操作前需要人工确认 |

## Further Reading

- [Anthropic Claude Code Agent SDK — agent loop and budgets](https://code.claude.com/docs/en/agent-sdk/agent-loop) — `max_turns`、`max_budget_usd`、工具白名单。
- [Microsoft Agent Framework — human-in-the-loop and governance](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — cost-governor 检查点。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 提供商侧成本控制。
- [Anthropic — Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) — 缓存机制。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 长周期智能体的成本画像。
