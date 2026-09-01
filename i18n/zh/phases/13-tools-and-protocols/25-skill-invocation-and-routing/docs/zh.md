# Skill 调用与路由

> 调用是授权决策与相关性决策的组合。好的描述帮助模型做出选择；好的策略决定该选择是否被允许。

**类型：** Build
**语言：** Python (stdlib)
**前置条件：** 第13阶段 · 24（技能发现与渐进式披露）
**时间：** ~105 分钟

## 学习目标

- 区分显式用户调用、隐式模型调用、应用调用和技能间调用。
- 将人类可见性和模型可执行性建模为独立策略维度。
- 编写带有正向触发器和边界案例的路由描述。
- 在追踪和测试中分离可执行性、选择、激活、参数绑定和执行。
- 适应特定运行时调用字段，但不将其呈现为可移植的 frontmatter。

## 问题

你安装了一个 `database-migration` 技能。用户可以通过名称运行它，但模型也能看到它的描述，并在有人问一般数据库问题时选择它。技能随后为只需要解释的任务提议了 schema 变更。

你添加 `user-invocable: false`，期望阻止用户手动运行它。在另一个运行时中，该字段被忽略。你添加 `disable-model-invocation: true`，期望技能完全消失。在那个理解它的运行时中，用户仍然可以显式调用它。

字段名本身没有问题。问题在于模型的理解有误。"用户能看到它"、"模型能选择它"、"应用能预加载它"和"其中的工具能执行"是独立的事实。一个名为 `invocable` 的单一布尔值无法表达这些区别。

路由还有第二个失败模式。如果描述模糊，多个技能都变得合理；如果描述堆砌关键词，不相关的任务也会触发它们。目录是一个概率接口：既要紧凑到能装下，又要具体到能路由。

## 概念

### 五种通道可以启动生命周期

| 行为者 | 调用形状 | 典型用途 | 主要风险 |
|---|---|---|---|
| 人类用户 | 在 UI 或提示中命名技能 | 有意识的工作流选择 | 用户期望宿主未授予的可用性或权限 |
| 模型或自主智能体 | 从任务上下文中选择目录条目 | 自动专家程序 | 误报路由 |
| 应用 | 通过运行时代码激活或预加载技能 | 固定产品工作流 | 与单一宿主隐藏耦合 |
| 另一个技能或子智能体 | 作为工作流依赖请求精确技能 | 组合 | 循环、依赖缺失或上下文泄漏 |
| 评估工具 | 在固定场景下激活精确技能 | 可重复测量 | 评估技能时意外绕过正在研究的生成策略 |

可移植的 Agent Skills 规范定义了包的结构。它没有标准化单一的全局斜杠命令 UI、隐式路由标志、应用 API 或子智能体生命周期。

### 五个调用阶段

```figure
skill-invocation-stages
```

精确使用以下术语：

- **Eligible（可执行）** 表示策略允许此行为者请求该技能。
- **Selected（已选择）** 表示用户命名了它或路由器判断它相关。
- **Activated（已激活）** 表示其指令进入了工作上下文。
- **Executing（执行中）** 表示智能体在这些指令下开始模型或工具工作。
- **Completed（已完成）** 表示输出通过了独立成功检查。

仅记录 `skill_used=true` 的追踪会掩盖失败发生的边界。

### 人类调用和模型调用构成 2x2 矩阵

| 人类可调用 | 模型可调用 | 模式 | 合适示例 |
|:---:|:---:|---|---|
| 是 | 是 | 共享 | 代码解释、测试规划、文档审查 |
| 是 | 否 | 人类专属 | 发布准备、计费导出、破坏性清理计划 |
| 否 | 是 | 模型专属 | 内部风格指南、领域参考、自动支持程序 |
| 否 | 否 | 禁用或应用专属 | 分阶段部署、废弃包、编程预加载 |

矩阵是一种策略模型，不是标准 YAML。

某个当前宿主使用 `disable-model-invocation: true` 对应人类专属行，使用 `user-invocable: false` 对应模型专属行。默认值为两者均可。另一个宿主使用 `agents/openai.yaml` 中的 `allow_implicit_invocation: false` 来保留显式调用同时禁用隐式选择。这些都是运行时适配器。未知宿主可能会忽略它们。

这个令人困惑的细节很重要：`user-invocable: false` 并不意味着"模型不能使用它"。它在定义它的宿主中移除了直接用户调用。`disable-model-invocation: true` 并不意味着"技能被禁用"。它移除模型发起的选择，但保留显式用户访问。

### 显式调用是身份优先

显式调用直接提供身份标识：

```text
/release-readiness v2.4.0
```

或：

```text
release-readiness check v2.4.0 without publishing
```

当前 Codex 界面记录了 `/skills` 用于选择，在请求中直接使用技能名称用于显式调用。Claude Code 记录了 `/skill-name` 和宿主特定的参数展开。确切的语法、菜单可见性、引号规则和变量展开属于宿主。

显式请求仍然通过策略检查。命名一个技能不应绕过缺失的权限、工作区约束、审批门控或运行时隔离。

### 隐式调用是描述优先

对于隐式路由，模型最初看到的是目录元数据而非完整内容。因此描述是技能的路由接口。

弱描述：

```yaml
description: Helps with releases.
```

过宽描述：

```yaml
description: Use for release, version, package, build, deploy, publish, tag, changelog, GitHub, CI, or software tasks.
```

有界描述：

```yaml
description: Inspect an already prepared release candidate and produce a readiness report. Use when the user asks whether a version, tag, package, or image is ready to publish; do not use for ordinary build failures or feature development.
```

有界版本包含：

1. **能力：** 检查已准备的候选版本。
2. **输出：** 就绪报告。
3. **正向边界：** 询问发布物是否就绪。
4. **负向边界：** 普通构建和开发不在范围内。

负向边界在两个相近技能共享词汇时很有用。它们不能替代边界案例评估。

### 路由是带弃权选项的分类

对于技能 `s` 和请求 `x`，想象一个路由器评分：

```text
score(s, x) = capability_match + trigger_match + context_match - exclusion_match - ambiguity_penalty
```

确切的评分可能是 LLM 决策而非算术。工程原则仍然成立：选择应超过阈值并击败竞争性技能。当证据不足时，弃权。

```figure
skill-routing-abstention
```

对于高影响技能，即使描述很强，隐式路由可能也不合适。当假阳性成本超过自动选择的便利性时，使用人类专属策略。

### 可执行性必须优先于排序

不要对所有发现的技能评分，选择最强匹配，然后事后检查该技能的政策。一个被阻止的顶级匹配会错误地阻止可执行的次级候选者被考虑。

对隐式路由使用以下顺序：

1. 按请求行为者和活动宿主适配器过滤发现的技能。
2. 仅对可执行的候选者评分。
3. 如果最强可执行匹配通过阈值和歧义规则，则选择它。
4. 当无可执行候选者或无可执行评分足够强时弃权。

假设 `incident-triage` 得分 `0.80` 但其宿主扩展禁用模型调用。`incident-review` 得分 `0.55` 并允许模型调用。路由器应将 `incident-review` 作为最佳可执行候选进行评估。不应选择 `incident-triage`，拒绝它然后停止。

这种顺序还确保策略变更不会改变相关性的含义。可执行性定义选择集。相关性对该集排序。

### 路由评估需要边界案例

正向案例证明召回率：

```json
{"prompt":"Is version 2.4.0 ready to publish?","expected":"release-readiness"}
```

清晰负向案例证明基本精确度：

```json
{"prompt":"Explain rotary position embeddings.","expected":null}
```

边界案例暴露边界质量：

```json
{"prompt":"Why did today's package build fail?","expected":"build-diagnostics"}
```

边界案例与发布技能共享 `package` 和 `build`，但属于其他地方。仅由明显正向和不相关负向组成的路由集会高估质量。

### 参数有三种表示

调用参数跨越多个边界：

```figure
skill-argument-boundaries
```

在每个边界，保留意图但不将文本视为代码。

- 宿主解析器决定命令语法和引号。
- 技能根据宿主规则接收绑定文本或变量。
- 指令验证必需值和默认值。
- 工具调用将值转换为类型化 schema 并重新验证。

不要将原始参数插值到 shell 命令中。优先使用带参数向量的脚本或类型化 MCP 工具。

### 应用调用是显式编排

产品可以激活技能，因为其工作流已经知道任务类型。例如，pull-request 审查服务可以在用户按下 Review 后预加载 `pull-request-risk-review`。

这消除了路由不确定性，但创建了对运行时 API 的依赖。将此适配器保持在可移植内容之外：

```figure
skill-host-adapter
```

技能在由不同合规客户端打开时应保持可理解。

### 技能间调用是类工具边

假设 `release-readiness` 在依赖文件更改时请求 `security-change-review`。

调用者应提供：

- 目标技能身份；
- 有界任务和工件路径；
- 预期响应契约；
- 调用原因；
- 不可用时的回退；
- 最大深度或循环规则。

```json
{
  "target_skill": "security-change-review",
  "task": "Review dependency changes in the candidate diff",
  "inputs": ["artifacts/release.diff"],
  "expected": "risk-report.json",
  "max_depth": 2
}
```

第二个技能不是盲目粘贴到第一个中。宿主决定如何激活它，以及是否共享上下文、在 fork 中运行或通过工具结果返回。

### 上下文生命周期是宿主特定的

激活后，技能内容可能保留在对话中、在压缩期间被摘要、或在委托上下文中运行。工具允许可能只持续一个回合，而指令持续更久。子智能体可能收到技能而没有父代的整个历史。

不要编写依赖隐形生命周期假设的技能。将持久输出放入文件或类型化状态，确保可重入安全，并说明中断后必须重新加载的内容。

```markdown
On resume, read `artifacts/release-readiness.json` if it exists.
Revalidate the candidate commit before continuing.
Do not repeat an external write whose idempotency key is already recorded.
```

## 构建

`code/main.py` 实现将策略和路由作为独立适配器。

模型包括：

- `Actor` 用于人类、模型、自主智能体、应用、技能和评估工具调用者；
- `SkillMetadata` 用于路由身份；
- `InvocationPolicy` 用于人类/模型矩阵；
- `InvocationRequest` 和 `InvocationDecision` 用于可追踪的输入和输出；
- `CorePolicyAdapter` 用于无宿主扩展的可移植行为；
- `ExtensionPolicyAdapter` 用于识别的运行时字段；
- `build_invocation_matrix(policy)` 用于 2x2 视图；
- `route_request(skills, request, adapter)` 用于相关性排序前的可执行性过滤、选择和拒绝。

运行它：

```bash
cd phases/13-tools-and-protocols/25-skill-invocation-and-routing
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

演示打印一个矩阵和对显式人类、隐式模型、自主智能体、应用、技能组合和评估工具通道的决策。其扩展适配器结果显示一个被阻止的顶级词汇匹配在可执行替代方案排序前被移除。它还包括精确名称白名单。不需要模型 API。确定性路由器存在是为了使策略边界可检查，而非声称词汇匹配能重现生产模型路由。

### 为什么核心适配器和扩展适配器分开

如果一个解析器为每个观察到的 frontmatter 字段分配含义，它会悄悄地将运行时约定提升为假标准。分开的适配器迫使调用者命名哪些宿主语义处于活动状态。

`CorePolicyAdapter` 仅使用应用提供的策略。`ExtensionPolicyAdapter` 识别一组明确的宿主字段并记录哪些字段改变了决策。

## 使用

在发布技能前编写调用契约：

```yaml
actors:
  human: allow
  model: deny
  application: allow
  skill: deny
explicit_name: release-readiness
arguments:
  candidate: required
  publish: fixed_false
ambiguity: ask_user
missing_dependency: stop
context:
  durable_state: artifacts/release-readiness.json
  max_composition_depth: 2
```

此契约是适配器和测试的设计文档。除非标准明确采用它，否则它不是可移植的 `SKILL.md` frontmatter。

## 交付

本课程产出 `skill-invocation-router` 包。它包括一个调用模型参考、一个示例宿主策略和一个非执行 CLI，用于评估一个人类、模型、自主智能体、应用、技能组合或评估工具请求，并返回包含通道、适配器、评分和理由的 JSON 决策。

单次请求 CLI 是策略探针，不是完整的触发评估。使用第27课中标记的正向案例和边界案例设计来计算混淆计数、精确度、召回率和重复运行稳定性。

## 练习

1. 创建人类/模型矩阵的所有四行，并为每行编写一个合法用例。
2. 向 `CorePolicyAdapter` 添加应用专属激活。证明人类和模型调用者仍然被拒绝。
3. 为部署技能编写十个边界案例。每个提示必须与技能共享词汇但属于不同工作流。
4. 在顶部两个路由分数之间添加歧义裕度。当裕度太小时返回 `ask`。
5. 向技能间请求添加最大组合深度并检测两技能循环。
6. 让同一标记集通过核心和扩展适配器。解释每一个改变的决策。

## 关键术语

| 术语 | 人们说什么 | 实际含义 |
|---|---|---|
| Explicit invocation | "斜杠命令" | 行为者直接提供技能身份，受策略约束 |
| Implicit invocation | "模型选择" | 路由器基于任务上下文从可执行目录元数据中选择 |
| User-invocable | "人类可以使用它" | 宿主特定的菜单或直接调用属性，不是核心字段 |
| Model-invocable | "智能体可以使用它" | 在宿主策略下的隐式模型选择可执行性 |
| Invocation adapter | "Frontmatter 解析器" | 将宿主的字段和 API 映射到声明的策略模型的代码 |
| Near miss | "困难负例" | 非触发请求，类似于技能的预期输入 |
| Abstention | "未选择技能" | 当证据缺失或模糊时有意作出的路由结果 |

## 进一步阅读

- [Optimizing skill descriptions](https://agentskills.io/skill-creation/optimizing-descriptions) 用于正向触发器、具体性和评估。
- [Evaluating skills](https://agentskills.io/skill-creation/evaluating-skills) 用于触发器和输出评估设计。
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills) 用于当前 Codex 显式和隐式调用控制。
- [Claude Code skills](https://code.claude.com/docs/en/skills) 用于一个宿主的 `user-invocable`、`disable-model-invocation`、参数和委托上下文。
