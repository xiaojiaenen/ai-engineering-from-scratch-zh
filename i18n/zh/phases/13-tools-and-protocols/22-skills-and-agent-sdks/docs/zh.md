# Agent Skills：便携式契约与运行时边界

> 技能不是一个带更好文件名的长提示词。它是一个可发现的指令、资源和可执行助手的集合包，通过运行时契约进入智能体的上下文。

**类型：** 构建
**语言：** Python (stdlib)
**前置条件：** 第 13 阶段 · 01（工具接口），第 13 阶段 · 05（工具模式设计）
**时间：** 约 90 分钟

## 学习目标

- 定义智能体技能，同时将其与提示词、仓库指令、工具、钩子、子代理或插件区分开来。
- 阅读便携式 `SKILL.md` 契约，并将其与运行时特定的扩展分离。
- 解释发现、选择、激活、资源加载、工具使用和验证作为不同的生命周期阶段。
- 在运行时将技能包放入智能体目录之前对其进行验证。
- 为具体任务在选择技能、MCP 工具、钩子、子代理或普通代码之间做出判断。

## 十分钟首次成功

在执行详细解释之前完成此操作。你将创建一个小型技能，将完整的评审器 bundle 安装到真实的智能体主机中，调用它，验证结果，然后移除它。这将通过一个可观察的结果来证明生命周期。

### 真实主机实验室的预飞检查

真实主机检查点需要 Node.js、`npx`、Python 3、一个选定的支持技能的主机，以及对你在安装器中选择的项目或用户范围的写入权限。首先验证本地命令：

```bash
node --version
npx --version
python3 --version
```

在安装前决定将使用哪个主机和范围。如果任何要求不可用，请在线访问本课程或在下面继续手动包练习。那个备用方案教授契约，但不证明主机发现、调用、bundle 脚本执行或卸载行为。将这些观察标记为待定。

### 1. 从空工作目录开始

从你保存学习工作的任意父目录运行以下命令：

```bash
mkdir -p agent-skills-first-run
cd agent-skills-first-run
TARGET_ROOT="$(pwd -P)"
printf 'TARGET_ROOT=%s\n' "$TARGET_ROOT"
ls -A
```

最后一个命令应不输出任何内容。如果它输出了文件，请选择另一个空目录，以便评审有清晰的边界。

为你的第一个技能创建一个目录：

```bash
mkdir -p my-first-skill
```

使用以下内容创建 `my-first-skill/SKILL.md`：

```markdown
---
name: my-first-skill
description: 当用户要求记录技术决策时，将粗略的会议笔记转换为紧凑的决策记录。
---

# 决策记录

提取决策、上下文、备选方案、负责人和下次审核日期。
如果笔记不包含决策，则询问一个澄清问题，而不是捏造一个。
```

验证你已在预期目录中创建了文件：

```bash
test -f my-first-skill/SKILL.md
```

无输出且退出代码为 0 表示文件存在。

### 2. 安装完整的评审器 bundle

留在 `agent-skills-first-run` 并运行：

```bash
npx skills add rohitg00/ai-engineering-from-scratch --skill skill-contract-reviewer --full-depth
```

选择你正在使用的智能体主机和范围。安装器应列出 `skill-contract-reviewer` 及其写入的目标位置。`--full-depth` 是必需的，因为本课程技能是一个带有引用、脚本和资产的嵌套 bundle。

将 `SKILL_ROOT` 设置为安装器报告的绝对目录。它必须是包含已安装的 `SKILL.md` 的目录，而不是课程源目录，也不是当前工作区：

```bash
# 用安装器打印的目标路径替换占位符。
SKILL_ROOT="$(cd "/absolute/path/to/skill-contract-reviewer" && pwd -P)"
test -f "$SKILL_ROOT/SKILL.md"
printf 'SKILL_ROOT=%s\n' "$SKILL_ROOT"
```

如果代理会话已经打开，请启动新会话或使用该主机的技能重新扫描命令。不要假设每个主机都会热重载其目录。

### 3. 显式调用它

在已安装的主机上，以 `agent-skills-first-run` 作为工作目录，使用该主机支持的语法：

| 主机 | 显式调用 |
|---|---|
| Codex | `skill-contract-reviewer`，或从 `/skills` 中选择，然后提供评审请求 |
| Claude Code | `/skill-contract-reviewer` 后跟评审请求 |
| 便携回退 | `Use skill-contract-reviewer to review the target package.` |

在请求中使用为 `SKILL_ROOT` 和 `TARGET_ROOT` 打印的绝对值。要求主机在执行前扩展它们并显示完全解析的命令，而不是依赖于进程工作目录的命令：

```text
Use skill-contract-reviewer to review <TARGET_ROOT>/my-first-skill. The installed bundle root is <SKILL_ROOT>. Run python3 <SKILL_ROOT>/scripts/check_skill.py <TARGET_ROOT>/my-first-skill. Before running it, show the fully resolved argv. Return the validation report, selected primitives, and one sentence for each selection. Include the resolved script path, resolved target path, cwd, argv, and exit code as execution evidence.
```

解析后的命令应具有以下形状，不含任何占位符：

```bash
python3 "/absolute/install/path/skill-contract-reviewer/scripts/check_skill.py" \
  "/absolute/workspace/path/agent-skills-first-run/my-first-skill"
```

成功的结果具有三个属性：

1. 主机通过名称找到 `skill-contract-reviewer`。
2. 评审器读取包契约并运行其捆绑的验证器。
3. 响应包含验证报告，对示例没有结构性错误，加上经过论证的原语选择。

执行证据还必须命名脚本路径、目标路径、cwd、精确参数向量和退出代码。一份没有这些字段的流畅报告并不能证明已安装的伴随脚本已运行。

如果主机报告该技能不可用，请验证安装目标，重新扫描或重新启动一次，然后重试显式请求。不要重写技能描述来掩盖安装失败。

### 4. 探测隐式选择

启动一个新的代理回合，在不提及技能的情况下输入相同的任务：

```text
Review <TARGET_ROOT>/my-first-skill as a reusable agent package and tell me whether its package contract is valid.
```

如果主机暴露了所选技能，记录它是否选择了 `skill-contract-reviewer`。如果主机未暴露路由，将隐式选择标记为未验证。显式调用是便携回退。

### 5. 清理

仅移除已安装的评审器 bundle：

```bash
npx skills remove skill-contract-reviewer
```

选择安装期间使用的主机和范围。重新扫描或新会话后，对 `skill-contract-reviewer` 的显式请求应报告其不可用。为后续课程保留 `my-first-skill`，或在完成课程轨道后移除实验室目录。

## 问题所在

假设你的团队有一个可靠的发布工作流。它会查找已合并的更改、检查迁移说明、更新 changelog、运行打包命令，并生成评审清单。

将此工作流放在一个提示词中易于粘贴但难以操作。该提示词没有稳定的身份、没有发现规则、没有资源边界、没有可测试的包形状，也回答不了基本问题：谁能调用它？模型何时应选择它？它可以运行哪些脚本？哪些文件可信？当上下文被压缩时会留下什么？

相反的错误是将每个可复用的指令都视为技能。仓库约定、确定性自动化、外部工具、事件钩子和委托代理解决不同的问题。将所有这些打包到 `SKILL.md` 会产生一个看起来可移植但实际上依赖某个主机未记录行为的目录。

第一个工程任务是分类。在决定如何打包工件之前，先确定工件是什么。

## 概念阐述

### 技能编码过程性知识

智能体技能是一个目录，其入口点是 `SKILL.md`。入口文件包含 YAML frontmatter 后跟 Markdown 指令。目录还可以包含引用、脚本和资源。

```figure
skill-package-anatomy
```

目录，而不仅仅是 Markdown 文件，是可部署单元。缺少引用的已复制 `SKILL.md` 是一个损坏的包，即使其 frontmatter 可以解析。

### 相邻的抽象概念

| 工件 | 主要职责 | 加载或运行时机 | 不应冒充什么 |
|---|---|---|---|
| 提示词 | 塑造单次模型交互 | 由应用程序或用户包含 | 带有资源的版本化包 |
| 仓库指令 | 解释一个代码库的既定规则 | 编码运行时进入该范围 | 可复用的任务工作流 |
| 智能体技能 | 提供可复用的过程性知识 | 显式或隐式激活 | 硬性授权边界 |
| MCP 工具 | 暴露类型化的远程能力 | 模型或应用程序调用它时 | 详细的操作规程 |
| 钩子 | 在事件上运行确定性逻辑 | 声明的事件发生时 | 概率性模型路由 |
| 子代理 | 使用单独的上下文和状态委托工作 | 编排器创建或调用它时 | 静态指令 bundle |
| 插件 | 分发更大的运行时扩展 | 主机安装或启用它时 | 便携式技能契约本身 |
| 学习的技能库 | 存储通过经验发现的行为 | 策略检索先前程序或轨迹时 | 基于标准的 `SKILL.md` 包 |

一个发布技能可以告诉智能体如何检查发布。一个 MCP 服务器可以暴露发布注册表。一个钩子可以禁止直接推送。一个子代理可以独立审计候选者。这些组件可以组合，因为它们保持不同的职责。

### 单词 "skill" 命名了两个不同的概念

研究系统有时将学习的程序、成功的轨迹或环境特定策略片段称为技能。智能体可以在探索期间创建这些工件，按任务相似性检索它们，执行它们，并从反馈中修订库。第 14 阶段 · 10 构建这种终身学习库。

本迷你课程中的智能体技能则不同。它是一个作者包，具有声明的文件系统契约、目录元数据、渐进式披露、运行时介导的调用和主机控制工具。它可以被智能体生成或改进，但该格式不需要学习。

| 维度 | 智能体技能包 | 学习的技能库 |
|---|---|---|
| 基本单元 | `SKILL.md` 目录 | 程序、策略、轨迹或记忆记录 |
| 创建 | 作者编写、生成或策展 | 通常从环境经验中发现 |
| 选择 | 目录描述加运行时策略 | 按任务状态检索或策略 |
| 执行 | 模型遵循指令并调用主机工具 | 环境运行存储的行为或代码工件 |
| 可移植性 | 包契约可以跨兼容主机 | 通常绑定到一个环境和工作空间 |
| 评估 | 路由、工件、安全性和主机兼容性 | 奖励、成功率、迁移和库增长 |

两种概念都封装了可复用的能力。它们不应仅仅因为共享一个名称就共享实现声明。

### 便携核心

Agent Skills 规范需要两个 frontmatter 字段：

```yaml
---
name: release-readiness
description: 当用户询问版本是否准备好发布时，检查发布候选者。
---
```

`name` 是稳定标识符。它必须满足规范中的命名规则并与父目录匹配。`description` 既是文档又是路由元数据。它应该说明技能做什么以及何时适用。

便携的可选字段是：

| 字段 | 目的 | 可移植性说明 |
|---|---|---|
| `license` | 声明包的条款 | 核心规范 |
| `compatibility` | 声明环境要求 | 核心规范 |
| `metadata` | 携带字符串值的扩展数据 | 核心规范 |
| `allowed-tools` | 建议预批准的工具 | 实验性；主机支持各不相同 |

Markdown 正文包含操作指令。它应该定义工作流、决策点、失败行为和指向支持资源的直接路径。

```markdown
# 发布就绪状态

此工作流用于发布候选者，而非普通开发构建。

1. 读取 `references/release-policy.md`。
2. 运行 `python3 scripts/inspect_release.py --format json`。
3. 如果报告包含阻塞性失败，则停止。
4. 从 `assets/release-checklist.md` 生成清单。
5. 在任何发布或标签操作之前征求批准。
```

### 运行时扩展是第二层

某些主机接受额外的 frontmatter 或伴随配置。这些字段可能有用，但它们不是自动可移植的。

| 行为 | 示例主机扩展 | 可移植核心？ |
|---|---|:---:|
| 隐藏技能免受模型路由，同时保持直接用户调用 | `disable-model-invocation` | 否 |
| 从用户命令菜单隐藏技能，同时允许模型路由 | `user-invocable` | 否 |
| 在命令菜单中显示参数帮助 | `argument-hint` | 否 |
| 在委托上下文中运行技能 | `context`、`agent` | 否 |
| 固定模型或推理设置 | `model`、`effort` | 否 |
| 注册生命周期自动化 | `hooks` | 否 |
| 禁用 Codex 中的隐式调用 | `agents/openai.yaml` 策略 | 否 |

将每个扩展视为适配器。在没有它的情况下保持核心工作流有效，记录回退方案，并测试消费它的主机。运行时可能会忽略未知字段、拒绝它或保留它但不实现该行为。

### Frontmatter 是可执行的元数据

元数据在读取技能正文之前改变系统行为。

- 格式错误的 `name` 会导致发现失败。
- 模糊的 `description` 会路由错误的请求。
- 仅人类可用的标志会从模型的目录中移除技能。
- 工具许可会改变主机是否询问权限。
- 上下文设置会将执行移到单独的智能体会话中。

像配置代码一样审查 frontmatter。验证它、版本化它，并将其行为包含在评估中。

### 技能生命周期

```figure
skill-runtime-lifecycle
```

每条箭头都是一个具有自己故障模式的边界。

1. **发现** 在配置的位置查找可能的包。
2. **验证** 在目录发布之前拒绝格式错误或危险的包。
3. **编目** 暴露紧凑的 `name` 和 `description`，而非完整包。
4. **选择** 决定技能是否相关。
5. **激活** 将正文加载到模型可见的上下文中。
6. **披露** 仅在分支需要时读取引用或资产。
7. **执行** 在主机的权限和隔离规则下使用主机工具。
8. **验证** 独立于模型的声明检查生成的工件。

压缩这些阶段会导致糟糕的心理模型。已发现的技能不是活动的。活动的技能不是被授权做它描述的一切的。被允许的工具调用并不能证明结果是正确的。

### 技能和工具是正交的

MCP 回答"这个应用程序可以调用哪些能力，它们的模式是什么？" 技能回答"模型应该如何处理这类任务？"

```figure
skill-tool-orthogonality
```

技能可以命名一个工具，但主机拥有实际的能力注册表。如果工具不存在，技能应该说明回退方案或清晰失败。它绝不应暗示命名一个能力就创建了它。

### 技能和仓库指令是不同的范围

仓库指令描述你已处于的环境：命令、约定、生成的文件和边界。技能为可能跨多个仓库发生的任务提供可复用的过程。

当两者都适用时，活跃的用户请求和仓库规则约束技能。通用的重构技能不得覆盖禁止编辑生成文件的仓库规则。

### 技能不相互导入

一个技能可以指示模型调用另一个技能，但这不是语言级别的导入。第二个技能仍然要经过运行时发现、资格检查、激活、权限和上下文处理。

将跨技能依赖关系写为可观察的工作流边：

```markdown
生成候选 changelog 后，调用 `release-risk-review` 技能。
传递候选路径并要求阻塞或非阻塞裁决。
如果该技能不可用，则停止并报告缺少的依赖。
```

这使得依赖关系可测试，并给主机一个执行策略的机会。

## 构建它

`code/main.py` 实现了一个小型面向标准的验证器和工件选择器。它保持仅 stdlib，以便每条规则都可见。

验证器暴露：

- `parse_frontmatter(text)` 用于将元数据与正文分离。
- `validate_skill_text(text, directory_name, allowed_runtime_extensions=())` 用于检查必填字段、命名、未知扩展、正文存在性和便携限制。
- `ValidationIssue` 和 `SkillReport` 用于返回结构化证据而非一个不透明的布尔值。
- `FrontmatterSyntaxError` 用于无法安全解释的输入。

选择器暴露 `TaskShape` 和 `select_primitives(task)`。它将任务需求映射到普通代码、仓库指令、技能、钩子、子代理或 MCP 工具。

运行实验室：

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/22-skills-and-agent-sdks
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

此命令块需要本地克隆，必须从该克隆内的任何位置开始，以便 `git rev-parse --show-toplevel` 可以解析仓库根目录。

演示打印一个有效便携技能、一个主机扩展技能、一个无效包和几个任务形状决策的 JSON。检查问题代码。包验证器应解释如何修复工件，而不是替作者猜测。

### 验证顺序很重要

在更深的规则之前验证便宜的结构性事实：

```figure
skill-validation-order
```

此顺序防止次要错误掩盖第一个损坏的不变量。

## 使用它

在编写技能之前，填写此决策卡：

| 问题 | 如果是 | 可能的原语 |
|---|---|---|
| 这是否需要跨多个步骤的可复用模型判断？ | 过程稳定但决策各异 | 技能 |
| 这是否必须在每次事件触发时发生？ | 缺少一次执行是不可接受的 | 钩子或应用程序代码 |
| 模型是否需要具有类型化输入的外部能力？ | 操作存在于模型上下文之外 | 工具或 MCP 服务器 |
| 这项工作是否需要隔离的上下文、状态或所有权？ | 单独的 worker 返回有界结果 | 子代理 |
| 这是否是针对一个仓库的指南？ | 它描述本地命令和约束 | 仓库指令 |
| 一次交互是否足够？ | 不需要包生命周期 | 提示词 |

许多生产工作流使用多行。此卡片防止一个工件假装提供所有属性。

## 交付它

本课程在 `outputs/` 下生成 `skill-contract-reviewer` bundle。它包含：

- 一个可移植的 `SKILL.md` 用于评审提议的技能包；
- 便携契约和原语选择的参考清单；
- 一个确定性验证脚本；
- 覆盖提示词、技能、工具、钩子、普通代码和子代理的任务形状测试用例。

安装完整 bundle，而不仅仅是其入口文件：

```bash
cd "$(git rev-parse --show-toplevel)"
python3 scripts/install_skills.py /tmp/aiefs-skills --phase 13 --type skill
```

课程安装器报告每个复制的第 13 阶段技能并写入
`/tmp/aiefs-skills/manifest.json`。这个干净的目的地检查包形状；上面的首次成功循环检查真实主机中的发现和调用。

以下课程深化每个生命周期阶段。第 24 课构建发现和渐进式披露。第 25 课构建调用策略和路由。第 26 课将权限与沙箱化分离。第 27 课将整个包转化为评估的发布工件。

## 练习

1. 使用 `TaskShape` 分类你自己团队的五个工作流。为你选择多个原语的每种情况辩护。
2. 添加边界测试，证明 500 个字符的 `compatibility` 值通过，而 501 个字符的值失败为规范错误。
3. 在允许列表中添加工时扩展。编写一个测试，证明同一文件仍可区别于仅便携的技能。
4. 将一个 400 行的提示词拆分为 `SKILL.md`、一个引用、一个脚本契约和一个输出模板。保持每个文件对一种类型的信息负责。
5. 为引用了不可用 MCP 工具的技能设计一个失败响应。不要静默替换权限更广的工具。
6. 评审现有技能，并将每个句子标记为路由、过程、策略、引用指针或输出契约。移动任何不属于该类别的内容。

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|---|---|---|
| 智能体技能 | "保存的提示词" | 过程性指令和可选资源的可发现目录 |
| 便携核心 | "每个运行时共享的字段" | 由 Agent Skills 规范定义的契约 |
| 运行时扩展 | "额外的 frontmatter" | 需要兼容适配器的特定主机配置 |
| 激活 | "技能运行了" | 技能正文进入了模型可见的上下文；执行可能稍后发生 |
| 技能依赖 | "导入另一个技能" | 具有可用性和策略检查的运行时介导调用边 |
| 工具契约 | "函数模式" | 输入、输出、权限、副作用、错误和能力证据 |

## 延伸阅读

- [Agent Skills specification](https://agentskills.io/specification) 了解便携目录和 frontmatter 契约。
- [Agent Skills best practices](https://agentskills.io/skill-creation/best-practices) 了解范围、指令和资源组织。
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills) 了解当前 Codex 发现和调用行为。
- [Claude Code skills](https://code.claude.com/docs/en/skills) 了解一个运行时的调用、参数、工具和委托上下文扩展。
