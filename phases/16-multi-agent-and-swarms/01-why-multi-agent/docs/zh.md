# 为何需要多智能体？

> 一个智能体遇到瓶颈时，明智之举不是造更大的智能体，而是造更多智能体。

**类型：** 学习模块
**编程语言：** TypeScript
**前置要求：** 第14阶段（智能体工程）
**预计时间：** 约60分钟

## 学习目标

- 识别单智能体的瓶颈（上下文溢出、专业知识混合、顺序执行瓶颈）并解释何时应该拆分为多个智能体
- 比较编排模式（流水线、并行扇出、监督者、分层）并为给定任务结构选择合适模式
- 设计具有清晰角色边界、共享状态和通信契约的多智能体系统
- 分析多智能体复杂度（延迟、成本、调试难度）与单智能体简洁性之间的权衡

## 问题所在

你在第14阶段构建了单个智能体。它运行良好，能读取文件、执行命令、调用API并推理结果。当你指向一个真实代码库：200个文件、三种编程语言、依赖基础设施的测试，以及需要先研究外部API再编写代码的需求时。

智能体卡住了。并非因为LLM不够智能，而是因为任务超出了单智能体循环的处理能力。上下文窗口被文件内容填满。智能体会忘记40次工具调用前读取的内容。它试图同时扮演研究者、编码者和审查者，结果三件事都做不好。

这就是单智能体的瓶颈。每当任务需要以下条件时就会遇到：

- **单窗口无法容纳的上下文** —— 阅读50个文件会突破20万token限制
- **不同阶段需要不同专业知识** —— 研究需要的提示词与代码生成完全不同
- **可并行执行的工作** —— 为什么顺序读取三个文件，而不同时读取呢？

## 核心概念

### 单智能体的瓶颈

单智能体意味着一个循环、一个上下文窗口、一个系统提示。请想象以下场景：

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

三个问题会暴露：

1. **上下文饱和** —— 工具结果不断堆积。到第30轮时，智能体已消耗15万token的文件内容、命令输出和先前推理。第5轮的关键细节被遗忘。

2. **角色混乱** —— 系统提示若写着“你是研究者、编码者、审查者和测试者”，产生的智能体会半心半意地研究、半吊子地编码，且永远完不成审查。

3. **顺序瓶颈** —— 智能体先读文件A，再读文件B，最后读文件C。三次串行LLM调用，三次串行工具执行，毫无并行性。

### 多智能体解决方案

将工作拆分。为每个智能体分配单一任务、独立上下文窗口和针对性优化的系统提示：

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

每个智能体具备：
- 专注的系统提示（“你是代码审查员，唯一任务是发现缺陷”）
- 自身的上下文窗口（不被其他智能体的工作污染）
- 明确的输入/输出契约（接收研究笔记，输出代码）

### 实际系统案例

**Claude Code子智能体** —— 当Claude Code通过`Task`生成子智能体时，会创建具有限定任务的子代理。父智能体保持上下文清晰，子智能体执行专注工作并返回摘要。

**Devin** —— 运行规划智能体、编码智能体和浏览器智能体。规划智能体分解任务步骤，编码智能体编写代码，浏览器智能体研究文档，各自拥有独立上下文。

**多智能体编码团队（SWE-bench）** —— SWE-bench上表现最佳的系统采用研究员读取代码库、规划师设计修复方案、编码员实现代码的分工模式，其得分高于单智能体系统。

**ChatGPT深度研究** —— 并行生成多个搜索智能体，各自探索不同角度，最后综合结果。

### 光谱体系

多智能体并非非黑即白，而是一个光谱：

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**单智能体** —— 单一循环、单一提示，适用于简单任务。

**子智能体** —— 父智能体生成专注子任务的子代理，父智能体维持计划，子代理反馈结果。Claude Code即采用此模式。

**流水线** —— 智能体顺序运行，A的输出成为B的输入，适用于分阶段工作流：研究→编码→审查→测试。

**团队** —— 智能体通过共享消息总线并行运行，各自拥有角色，编排器协调，适用于需要同时调动不同技能的场景。

**集群** —— 大量相同或近似的智能体共享状态，无固定编排器，智能体从队列中领取任务，适用于高吞吐量并行任务。

### 四大多智能体模式

#### 模式1：流水线

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

每个智能体转换数据并向前传递。易于理解，但任一阶段故障会阻塞后续环节。

#### 模式2：扇出/扇入

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

将任务分配给并行智能体，再合并结果。适用于可分解为独立子任务的工作。

#### 模式3：编排器-工作者

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

智能编排器决策行动、分配任务给工作者、综合结果。编排器自身是具备生成工作者工具的智能体。

#### 模式4：对等集群

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

无中央编排器，智能体点对点通信，决策从交互中涌现。调试难度较高，但易于扩展。

### 何时不该使用多智能体

多智能体会增加复杂性。智能体间的每条消息都是潜在故障点。调试范围从“阅读单个对话”扩大为“追踪五个智能体间的消息流”。

**保持单智能体当：**
- 任务适合单个上下文窗口（工作数据少于10万token）
- 无需为不同阶段设置不同系统提示
- 顺序执行足够快速
- 任务简单，拆分增加的开销大于收益

**复杂性成本：**
- 每个智能体边界都是有损压缩环节：A智能体的完整上下文会被压缩为给B智能体的消息
- 协调逻辑（谁做什么、何时做、按什么顺序）本身成为漏洞源
- 延迟增加：N个智能体至少需要N次串行LLM调用，若需来回沟通则更多
- 成本倍增：每个智能体独立消耗token

经验法则：若任务需要少于20次工具调用且适合10万token，请保持单智能体。

## 实践构建

### 步骤1：超载的单智能体

以下是一个试图处理所有工作的单智能体。它拥有庞大的系统提示和同时承载研究、代码、审查的单一上下文窗口：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
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

此方案的问题：
- 上下文窗口随阶段增长。到审查环节时，已包含研究笔记、代码和先前推理
- 系统提示过于通用，无法为各阶段优化
- 无并行执行

### 步骤2：专业智能体

现在拆分。每个智能体专注单一任务：

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
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

每个专家拥有专注的提示。每个获得仅包含所需输入的纯净上下文窗口。

### 步骤3：通过消息协调

用显式消息传递连接各专家：

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

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

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

每个智能体仅接收地址给它的消息，无上下文污染。研究员的5万token文档阅读数据永远不会进入审查员的上下文。

### 步骤4：对比

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

多智能体版本使用更多总token（三个智能体、三次独立LLM调用），但每个智能体的上下文保持清洁。由于系统提示专业化，各阶段质量得到提升。

## 实际应用

本课程产出了一个可复用提示模板，用于决定何时采用多智能体。参见`outputs/prompt-multi-agent-decision.md`。

## 练习

1. 增加第四个专家：“测试员”智能体，接收编码员的代码和审查员的反馈，然后编写测试
2. 修改流水线，使审查员能向编码员发送反馈形成修订循环（最多2轮）
3. 将顺序流水线改为扇出模式：并行运行研究员和“需求分析”智能体，合并输出后再传递给编码员

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 集群 | “AI智能体蜂群” | 一组对等智能体，共享状态且无固定领导者，行为通过局部交互涌现 |
| 编排器 | “老板智能体” | 其工具包含生成和管理其他智能体的智能体。负责规划和委派，可能不执行实际工作 |
| 协调器 | “交通警察” | 非智能体组件（通常仅是代码而非LLM），根据规则在智能体间路由消息 |
| 共识 | “智能体达成一致” | 多个智能体必须达成协议才能继续的协议，用于需要解决输出冲突的情况 |
| 涌现行为 | “智能体自行解决” | 由智能体交互产生但未显式编程的系统级模式，可能有益或有害 |
| 扇出/扇入 | “智能体的映射-归约” | 将任务拆分给并行智能体（扇出），再合并结果（扇入） |
| 消息传递 | “智能体相互通信” | 智能体间的通信机制：从一个智能体发送到另一个的结构化数据，取代共享上下文窗口 |

## 扩展阅读

- [新兴AI智能体架构全景](https://arxiv.org/abs/2409.02977) - 多智能体模式综述
- [AutoGen：赋能下一代LLM应用](https://arxiv.org/abs/2308.08155) - 微软多智能体对话框架
- [Claude Code子智能体文档](https://docs.anthropic.com/en/docs/claude-code) - Claude Code如何通过Task委派工作
- [CrewAI文档](https://docs.crewai.com/) - 基于角色的多智能体框架