# 为什么需要多智能体？

> 一个智能体撞墙了。明智的做法不是造一个更大的智能体——而是用更多智能体。

**类型：** 学习
**语言：** TypeScript
**前置知识：** 阶段 14（智能体工程）
**时间：** 约 60 分钟

## 学习目标

- 识别单智能体的天花板（上下文溢出、混合专家能力、串行瓶颈），并解释何时应该拆分为多个智能体
- 比较编排模式（流水线、并行扇出、主管、分层），并根据任务结构选择正确的模式
- 设计具有清晰角色边界、共享状态和通信契约的多智能体系统
- 分析多智能体复杂性（延迟、成本、调试难度）与单智能体简单性之间的权衡

## 问题所在

你在阶段 14 构建了一个单智能体。它运行良好——能读取文件、执行命令、调用 API，并对结果进行推理。然后你让它处理一个真实的代码库：200 个文件、三种编程语言、依赖基础设施的测试，以及需要在写代码前研究外部 API 的需求。

这个智能体崩溃了。不是因为 LLM 不够聪明，而是因为任务超出了单个智能体循环能处理的范围。上下文窗口被文件内容填满，智能体忘记了 40 次工具调用之前读过的内容。它试图同时扮演研究者、程序员和审查者的角色，结果三样都做得不好。

这就是单智能体的天花板。每当任务需要以下任何一种情况时，你都会遇到这个瓶颈：

- **上下文超过窗口容量** — 读取 50 个文件就会超过 20 万 token
- **不同阶段需要不同的专业知识** — 研究需要的提示方式与代码生成不同
- **可以并行完成的工作** — 既然可以同时读取三个文件，为什么要串行读取？

## 概念

### 单智能体天花板

单智能体意味着一个循环、一个上下文窗口、一个系统提示。想象一下：

```
┌─────────────────────────────────────────┐
│            单智能体                      │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         上下文窗口                │  │
│  │                                   │  │
│  │  研究笔记                           │  │
│  │  + 代码文件                        │  │
│  │  + 测试结果                         │  │
│  │  + 审查反馈                         │  │
│  │  + API 文档                        │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ 已满 ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  一个系统提示试图覆盖                     │
│  研究 + 编码 + 审查 + 测试               │
│                                         │
│  结果：每样都做得 mediocre（平庸）        │
└─────────────────────────────────────────┘
```

三个问题会出现：

1. **上下文饱和** — 工具结果不断堆积。到第 30 步时，智能体已经消耗了 15 万 token 的文件内容、命令输出和先前推理。第 5 步的关键细节已经丢失。

2. **角色混淆** — 一个写着"你是研究者、程序员、审查者和测试员"的系统提示，会产生一个半研究、半编码、永远不完成审查的智能体。

3. **串行瓶颈** — 智能体先读文件 A，再读文件 B，再读文件 C。三次串行的 LLM 调用。三次串行的工具执行。没有并行。

### 多智能体解决方案

拆分工作。让每个智能体只做一件事，拥有一个上下文窗口，并针对该工作调优一个系统提示：

```
┌──────────────────────────────────────────────────────────┐
│                    编排器                                │
│                                                          │
│  "为用户管理构建 REST API"                                 │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │研究者    │ │ 程序员   │ │ 审查者   │ │ 测试员   │  │
│   │          │ │          │ │          │ │          │  │
│   │ 读取     │ │ 编写     │ │ 检查     │ │ 运行     │  │
│   │ 文档、   │ │ 基于     │ │ 代码质量， │ │ 测试并   │  │
│   │ 查找     │ │ 研究     │ │ 找出     │ │ 报告结果  │  │
│   │ 模式     │ │ 规范写代码│ │ 缺陷     │ │          │  │
│   │          │ │          │ │          │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     合并结果                              │
└──────────────────────────────────────────────────────────┘
```

每个智能体都有：
- 聚焦的系统提示（"你是一个代码审查者。你的唯一任务是找出 bug。"）
- 独立的上下文窗口（不会被其他智能体的工作污染）
- 清晰的输入/输出契约（接收研究笔记，输出代码）

### 实际系统中的多智能体实践

**Claude Code 子智能体** — 当 Claude Code 使用 `Task` 派生子智能体时，它会创建一个具有限定任务的子智能体。父智能体保持上下文清洁，子智能体执行聚焦工作并返回摘要。

**Devin** — 运行一个规划智能体、一个编码智能体和一个浏览器智能体。规划器将工作分解为步骤，编码器编写代码，浏览器研究文档。每个都有独立的上下文。

**多智能体编程团队（SWE-bench）** — SWE-bench 上表现最好的系统使用一个读取代码库的研究者、一个设计修复方案的规划者，以及一个实现它的编码器。单智能体系统的得分较低。

**ChatGPT Deep Research** — 并行派生多个搜索智能体，每个探索不同的角度，然后综合结果。

### 谱系

多智能体不是一个二元选择。它是一个谱系：

```
简单 ──────────────────────────────────────────── 复杂

单智能体      子智能体        流水线        团队         群体
              (Subagents)                                  (Swarm)

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │共享   │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ 状态   │
             └───┘          └───┘───┘  │ 消息   │    └───────┘
                                       │ 总线   │
  1 个循环     父+子任务      按阶段     │       │    N 个对等体，
  1 个上下文                              └───────┘    涌现式
                                       显式角色       行为
```

**单智能体** — 一个循环，一个提示。适合简单任务。

**子智能体** — 父智能体为聚焦的子任务派生子智能体。父智能体维护计划，子智能体汇报结果。这就是 Claude Code 的做法。

**流水线** — 智能体按顺序运行。智能体 A 的输出成为智能体 B 的输入。适合分阶段工作流：研究 -> 编码 -> 审查 -> 测试。

**团队** — 智能体并行运行，通过共享消息总线通信。每个都有角色，编排器负责协调。适合需要不同技能同时工作的场景。

**群体** — 许多相同或近似的智能体共享状态。没有固定的编排器。智能体从队列中拾取任务。适合高吞吐量的并行任务。

### 四种多智能体模式

#### 模式 1：流水线

```
输入 ──▶ 智能体 A ──▶ 智能体 B ──▶ 智能体 C ──▶ 输出
         (研究)       (编码)       (审查)
```

每个智能体转换数据并传递给下一个。易于推理。一个阶段失败会阻塞后续阶段。

#### 模式 2：扇出/扇入

```
                ┌──▶ 智能体 A ──┐
                │              │
输入 ──▶ 分割 ├──▶ 智能体 B ──├──▶ 合并 ──▶ 输出
                │              │
                └──▶ 智能体 C ──┘
```

将工作分配到并行智能体，然后合并结果。适合可以分解为独立子任务的任务。

#### 模式 3：编排器-工作者

```
                    ┌──────────┐
                    │  编排器  │
                    └──┬───┬───┘
                  任务 │   │ 任务
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ 工作者 A  │   │ 工作者 B  │
           └──────────┘   └──────────┘
```

智能的编排器决定做什么，委托给工作者，并综合结果。编排器本身是一个具有派生工作者工具的智能体。

#### 模式 4：对等群体

```
         ┌───┐ ◄──── 消息 ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      消息  │    ┌───────────┐     │ 消息
           └───▶│  共享     │◄────┘
                │  状态/队列 │
           ┌───▶│           │◄────┐
           │    └───────────┘     │
      消息  │                      │ 消息
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── 消息 ────▶ │ D │
         └───┘                  └───┘
```

没有中央编排器。智能体点对点通信。决策从交互中涌现。更难调试，但可以扩展到大量智能体。

### 何时不该使用多智能体

多智能体增加了复杂性。智能体之间的每条消息都可能是潜在的故障点。调试从"阅读一段对话"变成"追踪五个智能体之间的消息。"

**保持单智能体当：**
- 任务在一个上下文窗口内（工作数据少于约 10 万 token）
- 不需要为不同阶段使用不同的系统提示
- 串行执行足够快
- 任务足够简单，拆分它带来的开销大于价值

**复杂性成本：**
- 每个智能体边界都是有损压缩步骤：智能体 A 的完整上下文被压缩成发给智能体 B 的消息
- 协调逻辑（谁做什么、何时做、以什么顺序）本身就是一个 bug 来源
- 延迟增加：N 个智能体意味着至少 N 次串行 LLM 调用，如果需要来回通信则更多
- 成本倍增：每个智能体独立消耗 token

经验法则：如果任务少于 20 次工具调用且能在 10 万 token 内完成，保持单智能体。

```figure
swarm-messages
```

## 动手构建

### 步骤 1：过载的单智能体

这是一个试图做所有事情的单智能体。它有一个巨大的系统提示和一个包含研究、代码和审查的上下文窗口：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  // 系统提示：要求智能体同时做研究、编码、审查和测试
  const systemPrompt = `你是一个全栈开发者。你必须：
1. 研究需求
2. 编写代码
3. 审查代码中的 bug
4. 编写测试
在一个对话中完成所有这些工作。`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  // 研究阶段
  const research = await fakeLLMCall(systemPrompt, `研究：${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  // 编码阶段
  const code = await fakeLLMCall(
    systemPrompt,
    `基于以下研究：\n${contextWindow.join("\n")}\n\n现在为以下任务编写代码：${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  // 审查阶段
  const review = await fakeLLMCall(
    systemPrompt,
    `基于所有先前上下文：\n${contextWindow.join("\n")}\n\n审查代码。`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

这种方法的缺点：
- 上下文窗口随着每个阶段增长。到审查步骤时，它包含研究笔记 AND 代码 AND 先前推理。
- 系统提示是通用的。无法针对每个阶段进行调优。
- 没有并行执行。

### 步骤 2：专业智能体

现在拆分它。每个智能体只做一件事：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      // 使用专业化的系统提示运行智能体
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

// 创建研究者智能体
const researcher = createSpecialist(
  "researcher",
  "你是一个技术研究员。阅读文档、查找模式并总结发现。只输出实现所需的事实。"
);

// 创建编码器智能体
const coder = createSpecialist(
  "coder",
  "你是一名高级 TypeScript 开发者。根据需求和研究笔记编写干净、经过测试的代码。只做这一件事。"
);

// 创建审查者智能体
const reviewer = createSpecialist(
  "reviewer",
  "你是一个代码审查者。找出 bug、安全问题和逻辑错误。要具体。引用行号。"
);
```

每个专家都有聚焦的提示。每个都获得干净的上下文窗口，只包含它需要的输入。

### 步骤 3：通过消息协调

用显式消息传递将专家连接起来：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  // 研究者工作
  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  // 组装编码器的输入
  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[来自 ${m.from}]: ${m.content}`)
    .join("\n");

  // 编码器工作
  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  // 组装审查者的输入
  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[来自 ${m.from}]: ${m.content}`)
    .join("\n");

  // 审查者工作
  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个智能体只接收发给它的消息。没有上下文污染。研究者的 5 万 token 文档读取永远不会进入审查者的上下文。

### 步骤 4：比较

```typescript
async function compare() {
  // 测试任务：为 Express.js API 构建速率限制中间件
  const task = "为 Express.js API 构建一个速率限制中间件";

  console.log("=== 单智能体 ===");
  const single = await singleAgentApproach(task);
  console.log(`Token 使用：${single.tokensUsed}`);
  console.log(`工具调用：${single.toolCalls}`);

  console.log("\n=== 多智能体 ===");
  const multi = await multiAgentApproach(task);
  console.log(`Token 使用：${multi.tokensUsed}`);
  console.log(`工具调用：${multi.toolCalls}`);
}
```

多智能体版本使用更多的总 token（三个智能体，三次独立的 LLM 调用），但每个智能体的上下文保持清洁。每个阶段的质量提高，因为系统提示是专业化的。

## 使用

本课产出一个可重用的提示，用于决定何时使用多智能体。参见 `outputs/prompt-multi-agent-decision.md`。

## 练习

1. 添加第四个专家：一个"测试员"智能体，它接收来自编码器的代码和来自审查者的审查反馈，然后编写测试
2. 修改流水线，使审查者可以将反馈发送回编码器进行修订循环（最多 2 轮）
3. 将顺序流水线转换为扇出：并行运行研究者和一个"需求分析器"智能体，然后在传递给编码器之前合并它们的输出

## 关键术语

| 术语 | 人们说什么 | 实际含义 |
|------|------------|----------|
| Swarm（群体） | "AI 智能体的蜂群思维" | 一组具有共享状态且没有固定领导者的对等智能体。行为从局部交互中涌现。 |
| Orchestrator（编排器） | "主管智能体" | 工具包括派生和管理其他智能体的智能体。它规划和委托，但不一定执行实际工作。 |
| Coordinator（协调器） | "交通警" | 一个非智能体组件（通常是代码，而非 LLM），根据规则在智能体之间路由消息。 |
| Consensus（共识） | "智能体达成一致" | 一种协议，多个智能体必须达成一致才能继续推进。用于需要解决冲突输出的场景。 |
| Emergent behavior（涌现行为） | "智能体自己想出来的" | 从智能体交互中产生的系统级模式，但不是显式编程的结果。可能有用也可能有害。 |
| Fan-out / fan-in（扇出/扇入） | "智能体的 MapReduce" | 将任务分配到并行智能体（扇出），然后组合它们的結果（扇入）。 |
| Message passing（消息传递） | "智能体相互交谈" | 智能体之间的通信机制：从一个智能体发送到另一个智能体的结构化数据，替代共享上下文窗口。 |

## 延伸阅读

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - 多智能体模式综述
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - 微软的多智能体对话框架
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) - Claude Code 如何使用 Task 委托工作
- [CrewAI documentation](https://docs.crewai.com/) - 基于角色的多智能体框架
