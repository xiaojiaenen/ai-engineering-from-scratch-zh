# 可执行约束：Agent 指令

> 以散文形式编写的指令是愿望。以约束形式编写的指令是测试。工作bench把每条规则变成 agent 可在运行时检查、审阅者可在事后验证的东西。

**类型：** 构建
**语言：** Python (stdlib)
**前置条件：** 第14阶段 · 32（最小工作bench）
**耗时：** ~50 分钟

## 学习目标

- 将路由说明文本与操作规则分离。
- 将启动规则、禁止操作、完成定义、不确定性处理和审批边界表达为机器可检查的约束。
- 实现一个规则检查器，对运行结果评分。
- 让规则集便于 diff 对比，使审阅者能看清变更内容。

## 问题所在

典型的 `AGENTS.md` 读起来像是入职文档。它告诉 agent "要谨慎"、"要彻底测试"、"不确定时要问"。三天后，agent 提交了一个没有测试的变更，写入了被禁止的目录，而且从未询问——因为它根本不知道边界在哪里。

指令在可操作时强大，在仅具愿景时软弱。解决方案是把规则写成工作bench可解释、审阅者可评分的形式。

## 概念

规则放在 `docs/agent-rules.md` 中，与简短的根路由分离。每条规则有名称、类别和检查函数。

```mermaid
flowchart LR
  Router[AGENTS.md] --> Rules[docs/agent-rules.md]
  Rules --> Checker[rule_checker.py]
  Checker --> Report[rule_report.json]
  Report --> Reviewer[审阅者]
```

### 涵盖大多数规则的五个类别

| 类别 | 规则回答的问题 | 示例 |
|------|----------------|------|
| 启动 | 工作开始之前必须满足什么条件？ | "状态文件存在且是最新的" |
| 禁止 | 什么绝对不允许发生？ | "不得编辑 `scripts/release.sh`" |
| 完成定义 | 什么能证明任务已完成？ | "pytest 退出码为 0 且验收行通过" |
| 不确定性 | agent 不确定时该怎么做？ | "打开问题备注，而不是猜测" |
| 审批 | 什么需要人工审批？ | "任何新依赖、任何生产环境写入" |

一条不属于这五类的规则，通常应当拆成两条。强制拆分。

### 规则是机器可读的

每条规则都有 slug、类别、一行描述和一个 `check` 字段，该字段引用 `rule_checker.py` 中的一个函数。添加规则即添加检查函数；检查器随工作bench一起成长。

### 规则是 diff 友好的

每条规则独占一个标题，放在同一个 markdown 文件中。重命名在 diff 中清晰可见。新规则置于类别顶部。过时的规则直接删除，而不是注释掉——因为工作bench是真相来源，而非团队上一季度的聊天日志。

### 规则与框架护栏的区别

框架护栏（OpenAI Agents SDK 护栏、LangGraph 中断）在运行时级别强制执行规则。本课程中的规则集是那些护栏所实现的人可读、可审查的契约。两者缺一不可：运行时在单个 turn 期间捕获违规，规则集证明运行时正在做正确的事。

### 渐进式披露：一张地图，而非百科全书

`AGENTS.md` 不断膨胀的原因在于，每次事件都会添加一条规则，却从不删除旧规则。一年后，文件达到两千行，agent 读完第一屏就耗尽了注意力预算，只按指令的一小部分行事。巨大的指令文件失败的原因与四十页入职文档失败的原因相同：读者只扫一眼，永远不会回头再看真正重要的部分。

解决方案不是缩短文件，而是分层。根路由保持足够小，每次会话都能读完，且只包含指针。深度内容由主题文件承载，仅在任务涉及对应主题时才加载。给 agent 一张地图，而不是整个百科全书，让它自己走到需要的页面。

```
AGENTS.md                  # 路由器，< 50 行：这个仓库是什么、去哪里查找、5 条硬性规则
docs/
  agent-rules.md           # 完整规则集（本课内容）
  architecture.md          # 任务触及模块边界时加载
  testing.md               # 任务编写或运行测试时加载
  deploy.md                # 仅在发布工作时加载，受审批规则门禁控制
feature_list.json          # 待办列表（第14阶段 · 36）
```

| 层级 | 存放位置 | 读取时机 | 大小预算 |
|------|----------|----------|----------|
| 路由 | `AGENTS.md` | 每次会话，始终 | 少于约 50 行 |
| 规则 | `docs/agent-rules.md` | 每次会话，启动时 | 每个类别一屏 |
| 主题文档 | `docs/<topic>.md` | 仅当任务触及该主题时 | 按需深入 |

两条测试保证分层不被破坏。可达性测试：agent 从路由出发最多两步应能到达任意规则，因此路由必须以路径链接每个主题文档，而非用散文描述。新鲜度测试：路由足够短，审阅者会在每次 PR 时重新阅读，这是阻止它无声膨胀回原来百科全书形态的唯一手段。一个无法解析的指针比缺失的规则更严重，因此路由中损坏的链接本身就是启动检查的违规项。

```figure
wb-rule-checkoff
```

## 构建它

`code/main.py` 提供：

- `agent-rules.md` 解析器，将规则加载到数据类中。
- `rule_checker.py` 样式检查函数，每个 `check` 引用对应一个。
- 一个演示 agent 运行，故意违反两条规则，以及一个能捕获它们的通过检查。

运行：

```
python3 code/main.py
```

输出：已解析规则集、运行追踪、每条规则的通过/失败状态，以及保存到脚本旁边的 `rule_report.json`。

## 生产环境中的三种模式

区分规则集能维持一个季度还是在一周内衰败的，是三种模式。

**写入时标记严重程度。** 每条规则携带 `severity`：`block`、`warn` 或 `info`。检查器报告全部三种；运行时仅在 `block` 时拒绝。大多数团队在早期高估严重程度，然后在截止压力下悄悄弱化；在写入时标记强制 upfront 校准。结合验证门禁（第14阶段 · 38），该门禁会将任何对 `block` 规则的覆盖签名记录到 `overrides.jsonl` 审计日志中。

**规则过期作为强制机制。** 每条规则携带 `expires_at` 日期（默认为创建后 90 天）。当一条未过期的规则连续 60 天零违规时，检查器发出警告；下一次季度评审决定是保留、降为 `info` 还是删除。Cloudflare 的生产 AI 代码审查数据（2026年4月，30天内 5,169 个仓库共 131,246 次审查运行）显示，具有明确过期时间的规则集每个仓库保持在 30 条以下；没有过期时间的规则集膨胀到 80 条以上，且大多数从未触发。

**Markdown 作为源文件，JSON 作为缓存。** `agent-rules.md` 是 authored 文件；`agent-rules.lock.json` 是检查器在热路径中读取的缓存。lock 文件由 pre-commit hook 再生。markdown diff 可审查；JSON 解析不在每次 turn 中出现。与 `package.json` / `package-lock.json` 及 `Cargo.toml` / `Cargo.lock` 相同的模式。

## 使用它

在生产环境中：

- Claude Code、Codex、Cursor 在会话开始时读取规则，并在拒绝操作时引用它们。检查器在 CI 中重新运行以捕获无声漂移。
- OpenAI Agents SDK 护栏将相同检查注册为输入和输出护栏。markdown 是文档表面；SDK 是运行时表面。
- LangGraph 中断在运行中的节点违反规则时触发。中断处理器读取规则、询问人工并恢复执行。

规则集在这三种平台之间可移植，因为它只是 markdown 加函数名。

## 交付它

`outputs/skill-rule-set-builder.md` 对项目所有者进行访谈，将其现有的散文指令分类到五个类别中，并生成版本化的 `agent-rules.md` 及检查器存根。

## 练习

1. 如果你的产品确实需要，添加第六个类别。论证它为何不会坍缩进五个类别之一。
2. 扩展检查器，使规则可携带严重程度（`block`、`warn`、`info`），报告相应汇总。
3. 将检查器接入 CI：若最新 agent 运行中 block 级规则失败，则构建失败。
4. 为每条规则添加"过期"字段。90 天内无检查失败，该规则进入评审。
5. 找一个真实的 `AGENTS.md` 并将其重写为五类别规则。其中有多少行是可操作的？多少是愿景性的？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 操作性规则 | "一条真正的指令" | 工作bench可在运行时检查的规则 |
| 愿景性规则 | "要谨慎" | 没有检查的规则；要么删除，要么升级 |
| 完成定义 | "验收" | 客观的、基于文件的证据证明任务完成 |
| block 级严重性 | "硬性规则" | 违规会中止运行；没有操作员干预无法屏蔽 |
| 规则过期 | "过期规则清理" | N 天内无失败的规则，进入退役评审 |

## 延伸阅读

- [OpenAI Agents SDK 护栏](https://openai.github.io/openai-agents-python/guardrails/)
- [LangGraph 中断](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/)
- [Anthropic，《构建有效 Agent》](https://www.anthropic.com/research/building-effective-agents)
- [Rick Hightower，《Agent RuleZ：一个确定性策略引擎》](https://medium.com/@richardhightower/agent-rulez-a-deterministic-policy-engine-for-ai-coding-agents-9489e0561edf) — 生产环境中的 block/warn/info 严重性
- [Cloudflare，《大规模编排 AI 代码审查》](https://blog.cloudflare.com/ai-code-review/) — 13.1 万次审查运行，规则组合经验
- [microservices.io，《GenAI 开发平台——第一部分：护栏》](https://microservices.io/post/architecture/2026/03/09/genai-development-platform-part-1-development-guardrails.html) — 规则与 CI 之间的纵深防御
- [《类型化合规：确定性护栏》(arXiv 2604.01483)](https://arxiv.org/pdf/2604.01483) — Lean 4 作为规则即检查的上限
- [logi-cmd/agent-guardrails](https://github.com/logi-cmd/agent-guardrails) — 合并门禁实现：作用域、变异测试、违规预算
- 第14阶段 · 32 — 本规则集嵌入的最小工作bench
- 第14阶段 · 38 — 消费规则报告的验证门禁
- 第14阶段 · 39 — 对规则合规性评分的审阅者 agent
