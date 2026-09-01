```markdown
# 人环（Human-in-the-Loop）：提议-承诺模式

> 2026 年对人环（HITL）的共识是具体的。它不是"代理询问，用户点击批准"。它是提议-承诺模式：提议的操作被持久化到耐久存储中，附带幂等键；向审核者呈现意图、数据谱系、权限范围、影响半径和回滚计划；只有在获得正面确认后才会提交；执行后验证以确认副作用实际发生。LangGraph 的 `interrupt()`、Microsoft Agent Framework 的 `RequestInfoEvent` 和 Cloudflare 的 `waitForApproval()` 都实现了相同的形状。典型的失效模式是橡皮图章式批准：未经审查就点击了"批准"。文档记录的反制措施是带有明确清单的挑战-响应机制。

**类型：** 学习
**语言：** Python（标准库，带幂等性的提议-承诺状态机）
**前置知识：** 第 15 阶段 · 12（耐久执行）、第 15 阶段 · 14（安全触发器）
**时间：** 约 60 分钟

## 问题所在

代理执行了一个操作。用户必须决定：批准还是不批准。如果决策是即时的，那很可能不是真正的审查。如果决策是结构化的，虽然慢但值得信赖。工程问题是如何让结构化审查成为阻力最小的路径。

2023 年的人环模式是一个同步提示："代理想发送邮件给 X，内容为 Y——批准？"用户点击"批准"。每个人都觉得系统是安全的。实际上，这种表面现象被大量使用了：用户快速批准，批准难以预测，当代理出错时，审计轨迹显示用户无法回忆的漫长批准历史。

2026 年的模式——提议-承诺——将人环置于耐久基础之上，附带结构化元数据，并要求正面承诺。每个托管代理 SDK 都提供了版本：LangGraph `interrupt()`、Microsoft Agent Framework `RequestInfoEvent`、Cloudflare `waitForApproval()`。API 名称不同，但形状一致。

## 概念

### 提议-承诺状态机

1. **提议。** 代理生成一个提议的操作。持久化到耐久存储（PostgreSQL、Redis、Durable Object）。包括：
   - 意图（代理为什么要这样做）
   - 数据谱系（什么来源导致了这个提议）
   - 权限范围（涉及哪些作用域 / 文件 / 端点）
   - 影响半径（最坏情况是什么）
   - 回滚计划（如果提交，如何撤销）
   - 幂等键（每个提议唯一；重新提交返回相同记录）
2. **呈现。** 审核者看到包含所有元数据的提议。审核者是人（不是代理自我审查）。
3. **提交。** 正面确认。操作执行。
4. **验证。** 执行后，重新读取副作用并确认。如果验证步骤失败，系统处于已知的不良状态，告警机制启动。

### 幂等键

没有幂等键，瞬态故障后的重试可能导致已批准的操作被执行两次。具体示例：用户批准"从 A 转账 $100 到 B"。网络抖动。工作流重试。用户已批准一次，但转账执行了两次。幂等键将批准与单个、唯一的副作用绑定；第二次执行成为空操作。

这与 Stripe 和 AWS API 使用的幂等性模式相同。在 Microsoft Agent Framework 文档中明确指出将其用于代理批准。

### 耐久性：为什么批准能超越进程

批准等待室是代理不拥有的状态部分。工作流暂停（第 12 课）。当批准到达时，工作流从该点恢复。这就是为什么 LangGraph 将 `interrupt()` 与 PostgreSQL 检查点配对，而不是仅使用内存状态——两天后的批准仍能找到完整的工作流。

### 橡皮图章式批准与挑战-响应反制措施

人环的默认 UI（"批准" / "拒绝"按钮）产生快速批准，没有真正的审查。文档记录的反制措施：一个挑战-响应清单，要求在启用"批准"按钮之前对特定问题给出正面回答。具体形状：

- "你理解这个操作涉及什么资源吗？[ ]"
- "你已验证影响半径可接受吗？[ ]"
- "如果失败你有回滚计划吗？[ ]"

不是为官僚主义而官僚主义——这是一个强制函数。无法勾选这些框的审核者要么要求澄清（升级），要么拒绝（安全默认值）。Anthropic 的代理安全研究明确引用清单驱动的 HITL 作为橡皮图章式批准模式的反制措施。

### 什么算作有影响力的操作

不是每个操作都需要提议-承诺模式。2026 年的指导：

- **有影响力的操作**（总是需要 HITL）：不可逆写入、金融交易、出站通信、生产数据库变更、破坏性文件系统操作。
- **可逆操作**（有时需要 HITL）：本地文件编辑、 staging 环境变更、带明确回滚的可逆写入。
- **读取和检查**（从不 HITL）：读取文件、列出资源、调用只读 API。

### 事后验证

"提交已运行"不同于"副作用已发生"。网络分区和竞态条件可能导致认为成功的工作流，而后端未持久化。验证步骤在提交后重新读取目标资源以确认。这与带 `RETURNING` 子句的数据库事务或在 `PutObject` 后调用 AWS `GetObject` 的模式相同。

### EU AI Act 第 14 条

第 14 条要求欧盟的高风险 AI 系统实施有效的人工监督。"有效"不是装饰性的。监管语言明确排除橡皮图章模式。带挑战-响应的提议-承诺是 Microsoft Agent 治理工具包合规文档中能够经受第 14 条审查的形状。

```figure
mx-propose-then-commit
```

## 使用它

`code/main.py` 在标准库 Python 中实现提议-承诺状态机。耐久存储是 JSON 文件。幂等键是 (thread_id, action_signature) 的哈希。驱动程序模拟三种情况：干净的批准流程、瞬态故障后的重试（不应重复执行），以及橡皮图章式默认值与挑战-响应流程。

## 交付它

`outputs/skill-hitl-design.md` 审查提议的 HITL 工作流是否符合提议-承诺形状，并标记缺失的元数据、幂等性、验证或挑战-响应层。

## 练习

1. 运行 `code/main.py`。确认已批准提议的重试使用耐久记录且不重新执行。现在修改幂等键以包含时间戳，并展示重试导致重复执行。

2. 扩展提议记录，添加 `rollback` 字段。模拟执行验证步骤失败的情况。展示回滚自动触发。

3. 阅读 Microsoft Agent Framework 的 `RequestInfoEvent` 文档。识别 API 包含的一个元数据字段，而玩具引擎缺失的。添加它并解释它防范什么风险。

4. 为特定操作设计一个挑战-响应清单（例如，"发布到公共 Twitter 账户"）。审核者必须回答哪三个问题？为什么是这三个？

5. 选择一个同步"批准？"提示就足够的场景（不需要耐久存储）。解释原因，并说明你接受的 Risks Class。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|---|---|---|
| 提议-承诺 | "两阶段批准" | 持久化提议 + 正面提交 + 验证 |
| 幂等键 | "重试安全令牌" | 每个提议唯一；第二次执行成为空操作 |
| 数据谱系 | "来自哪里" | 导致提议的具体源内容 |
| 影响半径 | "最坏情况" | 如果操作出错的影响范围 |
| 橡皮图章 | "快速批准" | 未经真正审查就点击"批准" |
| 挑战-响应 | "强制清单" | 审核者必须对特定问题正面确认 |
| RequestInfoEvent | "MS 代理框架原语" | 带结构化元数据的耐久 HITL 请求 |
| `interrupt()` / `waitForApproval()` | "框架原语" | LangGraph / Cloudflare 中相同形状的实现 |

## 延伸阅读

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`、耐久批准。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` 和 Durable Objects。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — HITL 作为长周期风险的补救措施。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — 高风险系统的监管基线。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 围绕监督的宪法框架。
```
