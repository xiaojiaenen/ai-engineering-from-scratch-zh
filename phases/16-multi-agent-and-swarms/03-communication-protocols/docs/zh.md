# 通信协议

> 不能说同一种语言的智能体不是一个团队。他们是对着虚空喊叫的陌生人。

**类型：** 构建
**语言：** TypeScript
**前提条件：** 第14阶段（智能体工程），课程16.01（为何需要多智能体）
**时长：** ~120分钟

## 学习目标

- 实现MCP工具发现和调用，使智能体能够使用外部服务器暴露的工具
- 构建A2A智能体卡片和任务端点，允许一个智能体通过HTTP将工作委派给另一个
- 比较MCP（工具访问）、A2A（智能体到智能体）、ACP（企业审计）和ANP（去中心化信任），并解释哪种协议解决哪种问题
- 在单一系统中整合多种协议，使智能体能够通过MCP发现工具并通过A2A委派任务

## 问题

你将系统拆分为多个智能体。一个研究员、一个编码员、一个审查员。他们各自擅长自己的工作。但现在你需要他们真正互相沟通。

你的第一次尝试显而易见：传递字符串。研究员返回一大段文本，编码员尽可能解析它。这能行得通，直到编码员误解了研究摘要，或者两个智能体互相等待导致死锁，或者你需要不同团队构建的智能体协作。突然间，"只管传递字符串"的做法就崩溃了。

这就是通信协议问题。如果没有智能体如何交换信息的共享契约，多智能体系统就会脆弱、不可审计，并且无法扩展到超出你亲自编写的少数智能体。

AI生态系统提出了四种协议，每种都解决问题的不同部分：

- **MCP** 用于工具访问
- **A2A** 用于智能体间协作
- **ACP** 用于企业可审计性
- **ANP** 用于去中心化身份和信任

本课程将深入探讨。你将阅读每个规范的真实线格式，构建可工作的实现，并将所有四种协议连接到一个统一的系统中。

## 概念

### 协议全景

将这四种协议想象成层次结构，每层解决不同的问题：

```mermaid
block-beta
  columns 1
  block:ANP["ANP — How do agents trust strangers?\nDecentralized identity (DID), E2EE, meta-protocol"]
  end
  block:A2A["A2A — How do agents collaborate on goals?\nAgent Cards, task lifecycle, streaming, negotiation"]
  end
  block:ACP["ACP — How do agents talk in auditable systems?\nRuns, trajectory metadata, session continuity"]
  end
  block:MCP["MCP — How does an agent use a tool?\nTool discovery, execution, context sharing"]
  end

  style ANP fill:#f3e8ff,stroke:#7c3aed
  style A2A fill:#dbeafe,stroke:#2563eb
  style ACP fill:#fef3c7,stroke:#d97706
  style MCP fill:#d1fae5,stroke:#059669
```

它们不是竞争对手。它们在不同层次解决不同的问题。

### MCP（回顾）

MCP在第13阶段有深入介绍。快速回顾：MCP标准化了LLM如何连接到外部工具和数据源。它是一种**客户端-服务器**协议，智能体（客户端）发现并调用服务器暴露的工具。

```mermaid
sequenceDiagram
    participant Agent as Agent (client)
    participant MCP1 as MCP Server<br/>(database, API, files)

    Agent->>MCP1: list tools
    MCP1-->>Agent: tool definitions
    Agent->>MCP1: call tool X
    MCP1-->>Agent: result
```

MCP是**智能体到工具**的通信。它不帮助智能体之间互相交谈。

### A2A（智能体间协议）

**创建者：** Google（现归属Linux基金会，作为`lf.a2a.v1`）
**规范版本：** 1.0.0
**问题：** 自主智能体如何协作、协商并向彼此委派任务？

A2A是**对等智能体协作**的协议。MCP将智能体连接到工具，A2A将智能体连接到其他智能体。每个智能体在一个已知URL发布一个**智能体卡片**，其他智能体发现它、与它协商并向它委派任务。

#### A2A如何工作

```mermaid
sequenceDiagram
    participant Client as Client Agent
    participant Remote as Remote Agent

    Client->>Remote: GET /.well-known/agent-card.json
    Remote-->>Client: Agent Card (skills, modes, security)

    Client->>Remote: POST /message:send
    Remote-->>Client: Task (submitted/working)

    alt Polling
        Client->>Remote: GET /tasks/{id}
        Remote-->>Client: Task status + artifacts
    else Streaming
        Client->>Remote: POST /message:stream
        Remote-->>Client: SSE: statusUpdate
        Remote-->>Client: SSE: artifactUpdate
        Remote-->>Client: SSE: completed
    end
```

#### 真实的智能体卡片

以下是现实世界中A2A智能体卡片的样子。在`GET /.well-known/agent-card.json`处提供：

```json
{
  "name": "Research Agent",
  "description": "Searches documentation and summarizes findings",
  "version": "1.0.0",
  "supportedInterfaces": [
    {
      "url": "https://research-agent.example.com/a2a/v1",
      "protocolBinding": "JSONRPC",
      "protocolVersion": "1.0"
    },
    {
      "url": "https://research-agent.example.com/a2a/rest",
      "protocolBinding": "HTTP+JSON",
      "protocolVersion": "1.0"
    }
  ],
  "provider": {
    "organization": "Your Company",
    "url": "https://example.com"
  },
  "capabilities": {
    "streaming": true,
    "pushNotifications": false
  },
  "defaultInputModes": ["text/plain", "application/json"],
  "defaultOutputModes": ["text/plain", "application/json"],
  "skills": [
    {
      "id": "web-research",
      "name": "Web Research",
      "description": "Searches the web and synthesizes findings",
      "tags": ["research", "search", "summarization"],
      "examples": ["Research the latest changes in React 19"]
    },
    {
      "id": "doc-analysis",
      "name": "Documentation Analysis",
      "description": "Reads and analyzes technical documentation",
      "tags": ["docs", "analysis"],
      "inputModes": ["text/plain", "application/pdf"],
      "outputModes": ["application/json"]
    }
  ],
  "securitySchemes": {
    "bearer": {
      "httpAuthSecurityScheme": {
        "scheme": "Bearer",
        "bearerFormat": "JWT"
      }
    }
  },
  "security": [{ "bearer": [] }]
}
```

需要注意的关键点：
- **技能**是智能体能做什么。每个技能都有一个ID、标签和支持的输入/输出MIME类型。客户端智能体以此判断这个远程智能体是否能处理其请求。
- **supportedInterfaces**列出了多个协议绑定。一个智能体可以同时支持JSON-RPC、REST和gRPC。
- **安全性**内置于卡片中。客户端在发出第一个请求之前就知道需要什么认证。

#### 任务生命周期

任务是A2A工作的核心单元。它们经历定义的状态：

```mermaid
stateDiagram-v2
    [*] --> submitted
    submitted --> working
    working --> input_required: needs more info
    input_required --> working: client sends data
    working --> completed: success
    working --> failed: error
    working --> canceled: client cancels
    submitted --> rejected: agent declines

    completed --> [*]
    failed --> [*]
    canceled --> [*]
    rejected --> [*]

    note right of completed: Terminal states are immutable.\nFollow-ups create new tasks\nwithin the same contextId.
```

全部8个状态（规范还定义了`UNSPECIFIED`作为哨兵，此处省略）：

| 状态 | 终态？ | 含义 |
|---|---|---|
| `TASK_STATE_SUBMITTED` | 否 | 已确认，尚未开始处理 |
| `TASK_STATE_WORKING` | 否 | 正在处理中 |
| `TASK_STATE_INPUT_REQUIRED` | 否 | 智能体需要客户端提供更多信息 |
| `TASK_STATE_AUTH_REQUIRED` | 否 | 需要认证 |
| `TASK_STATE_COMPLETED` | 是 | 成功完成 |
| `TASK_STATE_FAILED` | 是 | 带错误完成 |
| `TASK_STATE_CANCELED` | 是 | 完成前被取消 |
| `TASK_STATE_REJECTED` | 是 | 智能体拒绝了任务 |

一旦任务达到终态，它就是不可变的。不再有后续消息。后续操作在同一个`contextId`内创建一个新任务。

#### 线格式

A2A使用JSON-RPC 2.0。以下是真实消息交换的样子：

**客户端发送任务：**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "SendMessage",
  "params": {
    "message": {
      "messageId": "msg-001",
      "role": "ROLE_USER",
      "parts": [{ "text": "Research React 19 compiler features" }]
    },
    "configuration": {
      "acceptedOutputModes": ["text/plain", "application/json"],
      "historyLength": 10
    }
  }
}
```

**智能体响应任务：**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "task": {
      "id": "task-abc-123",
      "contextId": "ctx-xyz-789",
      "status": {
        "state": "TASK_STATE_COMPLETED",
        "timestamp": "2026-03-27T10:30:00Z"
      },
      "artifacts": [
        {
          "artifactId": "art-001",
          "name": "research-results",
          "parts": [{
            "data": {
              "findings": [
                "React 19 compiler auto-memoizes components",
                "No more manual useMemo/useCallback needed",
                "Compiler runs at build time, not runtime"
              ]
            },
            "mediaType": "application/json"
          }]
        }
      ]
    }
  }
}
```

**通过SSE流式传输：**
```text
POST /message:stream HTTP/1.1
Content-Type: application/json
A2A-Version: 1.0

data: {"task":{"id":"task-123","status":{"state":"TASK_STATE_WORKING"}}}

data: {"statusUpdate":{"taskId":"task-123","status":{"state":"TASK_STATE_WORKING","message":{"role":"ROLE_AGENT","parts":[{"text":"Searching documentation..."}]}}}}

data: {"artifactUpdate":{"taskId":"task-123","artifact":{"artifactId":"art-1","parts":[{"text":"partial findings..."}]},"append":true,"lastChunk":false}}

data: {"statusUpdate":{"taskId":"task-123","status":{"state":"TASK_STATE_COMPLETED"}}}
```

### ACP（智能体通信协议）

**创建者：** IBM / BeeAI
**规范版本：** 0.2.0 (OpenAPI 3.1.1)
**状态：** 正在合并到Linux基金会的A2A中
**问题：** 智能体如何在具有完整可审计性、会话持续性和轨迹跟踪的情况下通信？

ACP是**企业协议**。与许多总结声称的不同，ACP **不**使用JSON-LD。它是一个通过OpenAPI定义的简单REST/JSON API。其特殊之处在于**TrajectoryMetadata**：每个智能体响应都可以携带产生该响应的推理步骤和工具调用的详细日志。

```mermaid
sequenceDiagram
    participant Client
    participant ACP as ACP Agent
    participant Audit as Audit Log

    Client->>ACP: POST /runs (mode: sync)
    ACP->>ACP: Process request...
    ACP->>Audit: Log trajectory:<br/>reasoning + tool calls
    ACP-->>Client: Response + TrajectoryMetadata
    Note over Audit: Every step recorded:<br/>tool_name, tool_input,<br/>tool_output, reasoning
```

#### ACP中的智能体发现

ACP定义了四种发现方法：

```mermaid
graph LR
    A[Agent Discovery] --> B["Runtime<br/>GET /agents"]
    A --> C["Open<br/>.well-known/agent.yml"]
    A --> D["Registry<br/>Centralized catalog"]
    A --> E["Embedded<br/>Container labels"]

    style B fill:#dbeafe,stroke:#2563eb
    style C fill:#d1fae5,stroke:#059669
    style D fill:#fef3c7,stroke:#d97706
    style E fill:#f3e8ff,stroke:#7c3aed
```

**AgentManifest** 比A2A的智能体卡片更简单：

```json
{
  "name": "summarizer",
  "description": "Summarizes documents with source citations",
  "input_content_types": ["text/plain", "application/pdf"],
  "output_content_types": ["text/plain", "application/json"],
  "metadata": {
    "tags": ["summarization", "RAG"],
    "framework": "BeeAI",
    "capabilities": [
      {
        "name": "Document Summarization",
        "description": "Condenses long documents into key points"
      }
    ],
    "recommended_models": ["llama3.3:70b-instruct-fp16"],
    "license": "Apache-2.0",
    "programming_language": "Python"
  }
}
```

#### 运行生命周期

ACP使用"运行"而不是"任务"。运行是智能体执行，有三种模式：

| 模式 | 行为 |
|---|---|
| `sync` | 阻塞式。响应包含完整结果。 |
| `async` | 立即返回202。轮询`GET /runs/{id}`获取状态。 |
| `stream` | SSE流。智能体工作时触发事件。 |

```mermaid
stateDiagram-v2
    [*] --> created
    created --> in_progress
    in_progress --> completed: success
    in_progress --> failed: error
    in_progress --> awaiting: needs input
    awaiting --> in_progress: client resumes
    in_progress --> cancelling: cancel request
    cancelling --> cancelled

    completed --> [*]
    failed --> [*]
    cancelled --> [*]
```

#### TrajectoryMetadata（审计轨迹）

这是ACP的关键差异化因素。每条消息部分都可以包含元数据，准确显示智能体做了什么：

```json
{
  "role": "agent/researcher",
  "parts": [
    {
      "content_type": "text/plain",
      "content": "The weather in San Francisco is 72F and sunny.",
      "metadata": {
        "kind": "trajectory",
        "message": "I need to check the weather for this location",
        "tool_name": "weather_api",
        "tool_input": { "location": "San Francisco, CA" },
        "tool_output": { "temperature": 72, "condition": "sunny" }
      }
    }
  ]
}
```

对于受监管行业，这是黄金标准。每个答案都伴随着可证明的推理链：调用了哪些工具，使用了什么输入，收到了什么输出。不再是黑箱。

ACP还支持**CitationMetadata**用于来源归属：

```json
{
  "kind": "citation",
  "start_index": 0,
  "end_index": 47,
  "url": "https://weather.gov/sf",
  "title": "NWS San Francisco Forecast"
}
```

### ANP（智能体网络协议）

**创建者：** 开源社区（由高维章创立）
**仓库：** [github.com/agent-network-protocol/AgentNetworkProtocol](https://github.com/agent-network-protocol/AgentNetworkProtocol)
**问题：** 不同组织的智能体如何在没有中心权威的情况下相互信任？

ANP是**去中心化身份协议**。它使用W3C去中心化标识符（DID）和端到端加密构建信任。与通过已知端点发现智能体的A2A不同，ANP让智能体通过密码学方式证明其身份。

ANP有三层：

```mermaid
graph TB
    subgraph Layer3["Layer 3: Application Protocol"]
        AD[Agent Description Documents]
        DISC[Discovery endpoints]
    end
    subgraph Layer2["Layer 2: Meta-Protocol"]
        NEG[AI-powered protocol negotiation]
        CODE[Dynamic code generation]
    end
    subgraph Layer1["Layer 1: Identity & Secure Communication"]
        DID["did:wba (W3C DID)"]
        HPKE[HPKE E2EE - RFC 9180]
        SIG[Signature verification]
    end

    Layer3 --> Layer2
    Layer2 --> Layer1

    style Layer1 fill:#d1fae5,stroke:#059669
    style Layer2 fill:#dbeafe,stroke:#2563eb
    style Layer3 fill:#f3e8ff,stroke:#7c3aed
```

#### DID文档（真实结构）

ANP使用一种称为`did:wba`（基于网络的智能体）的自定义DID方法。DID `did:wba:example.com:user:alice` 解析为 `https://example.com/user/alice/did.json`：

```json
{
  "@context": [
    "https://www.w3.org/ns/did/v1",
    "https://w3id.org/security/suites/jws-2020/v1",
    "https://w3id.org/security/suites/secp256k1-2019/v1"
  ],
  "id": "did:wba:example.com:user:alice",
  "verificationMethod": [
    {
      "id": "did:wba:example.com:user:alice#key-1",
      "type": "EcdsaSecp256k1VerificationKey2019",
      "controller": "did:wba:example.com:user:alice",
      "publicKeyJwk": {
        "crv": "secp256k1",
        "x": "NtngWpJUr-rlNNbs0u-Aa8e16OwSJu6UiFf0Rdo1oJ4",
        "y": "qN1jKupJlFsPFc1UkWinqljv4YE0mq_Ickwnjgasvmo",
        "kty": "EC"
      }
    },
    {
      "id": "did:wba:example.com:user:alice#key-x25519-1",
      "type": "X25519KeyAgreementKey2019",
      "controller": "did:wba:example.com:user:alice",
      "publicKeyMultibase": "z9hFgmPVfmBZwRvFEyniQDBkz9LmV7gDEqytWyGZLmDXE"
    }
  ],
  "authentication": [
    "did:wba:example.com:user:alice#key-1"
  ],
  "keyAgreement": [
    "did:wba:example.com:user:alice#key-x25519-1"
  ],
  "humanAuthorization": [
    "did:wba:example.com:user:alice#key-1"
  ],
  "service": [
    {
      "id": "did:wba:example.com:user:alice#agent-description",
      "type": "AgentDescription",
      "serviceEndpoint": "https://example.com/agents/alice/ad.json"
    }
  ]
}
```

需要注意的关键点：
- **密钥分离**被强制执行。签名密钥（secp256k1）与加密密钥（X25519）是分开的。
- **`humanAuthorization`** 是ANP独有的。这些密钥在使用前需要显式的人类批准（生物识别、密码、HSM）。高风险操作如资金转移会经过此路径。
- **`keyAgreement`** 密钥用于HPKE端到端加密（RFC 9180）。
- **service** 部分链接到智能体描述文档。

#### ANP中的信任如何运作

ANP **不**使用信任网或背书图。信任是双边的，并在每次交互时验证：

```mermaid
sequenceDiagram
    participant A as Agent A
    participant Domain as Agent A's Domain
    participant B as Agent B

    A->>B: HTTP request + DID + signature
    B->>Domain: Fetch DID document (HTTPS)
    Domain-->>B: DID document + public key
    B->>B: Verify signature with public key
    B-->>A: Issue access token
    A->>B: Subsequent requests use token
    Note over A,B: Trust = TLS domain verification<br/>+ DID signature verification<br/>+ Principle of least trust
```

信任来自三个来源：
1. **域名级TLS** 验证DID文档主机
2. **DID密码学签名** 验证智能体身份
3. **最小信任原则** 仅授予最低权限

没有基于Gossip的信任传播或PageRank评分。你直接通过其DID验证每个智能体。

#### 元协议协商

这是ANP最新颖的功能。当来自不同生态系统的两个智能体相遇时，它们不需要预先商定的数据格式。它们用自然语言协商：

```json
{
  "action": "protocolNegotiation",
  "sequenceId": 0,
  "candidateProtocols": "I can communicate using:\n1. JSON-RPC with hotel booking schema\n2. REST with OpenAPI 3.1 spec\n3. Natural language over HTTP",
  "modificationSummary": "Initial proposal",
  "status": "negotiating"
}
```

```mermaid
sequenceDiagram
    participant A as Agent A
    participant B as Agent B

    A->>B: protocolNegotiation (candidateProtocols)
    B->>A: protocolNegotiation (counter-proposal)
    A->>B: protocolNegotiation (accepted)
    Note over A,B: Agents dynamically generate code<br/>to handle the agreed format.<br/>Max 10 rounds, then timeout.
```

智能体来回协商（最多10轮），直到就格式达成一致，然后动态生成代码来处理它。状态值：`negotiating`、`rejected`、`accepted`、`timeout`。

这意味着两个从未见过面的智能体可以自己弄清楚如何通信，而无需任何人预先定义共享模式。

### 对比（修正版）

| | MCP | A2A | ACP | ANP |
|---|---|---|---|---|
| **创建者** | Anthropic | Google / Linux基金会 | IBM / BeeAI | 社区 |
| **规范格式** | JSON-RPC | JSON-RPC / REST / gRPC | OpenAPI 3.1 (REST) | JSON-RPC |
| **主要用途** | 智能体到工具 | 智能体到智能体 | 智能体到智能体 | 智能体到智能体 |
| **发现** | 工具列表 | `/.well-known/agent-card.json` | `GET /agents`、`/.well-known/agent.yml` | `/.well-known/agent-descriptions`、DID服务端点 |
| **身份** | 隐式（本地） | 安全方案（OAuth、mTLS） | 服务器级 | W3C DID (`did:wba`) 配合端到端加密 |
| **审计轨迹** | 不适用 | 基础（任务历史） | TrajectoryMetadata（工具调用、推理） | 未正式指定 |
| **状态机** | 不适用 | 9种任务状态 | 7种运行状态 | 不适用 |
| **流式传输** | 不适用 | SSE | SSE | 传输无关 |
| **独特功能** | 工具模式 | 智能体卡片 + 技能 | 轨迹审计 | 元协议协商 |
| **最适合** | 工具与数据 | 动态协作 | 受监管行业 | 跨组织信任 |
| **状态** | 稳定 | 稳定 (v1.0) | 合并到A2A | 积极开发中 |

### 它们如何协同工作

这些协议不是互斥的。一个现实的企业系统会使用多种：

```mermaid
graph TB
    subgraph org["Your Organization"]
        RA[Research Agent] <-->|A2A| CA[Coding Agent]
        RA -->|MCP| SS[Search Server]
        CA -->|MCP| GS[GitHub Server]
        AUDIT["All agent responses carry<br/>ACP TrajectoryMetadata"]
    end

    subgraph ext["External (DID verified via ANP)"]
        EA[External Agent]
        PA[Partner Agent]
    end

    RA <-->|ANP + A2A| EA
    CA <-->|ANP + A2A| PA

    style org fill:#f8fafc,stroke:#334155
    style ext fill:#fef2f2,stroke:#991b1b
    style AUDIT fill:#fef3c7,stroke:#d97706
```

- **MCP** 将每个智能体连接到其工具
- **A2A** 处理智能体之间的协作（内部和外部）
- **ACP** 用轨迹元数据包装响应以实现可审计性
- **ANP** 为你不控制的智能体提供身份验证

## 动手构建

### 步骤1：核心消息类型

每个多智能体系统都从消息格式开始。我们定义映射到真实协议使用的类型：

```typescript
import crypto from "node:crypto";

type MessageRole = "user" | "agent";

type MessagePart =
  | { kind: "text"; text: string }
  | { kind: "data"; data: unknown; mediaType: string }
  | { kind: "file"; name: string; url: string; mediaType: string };

type TrajectoryEntry = {
  reasoning: string;
  toolName?: string;
  toolInput?: unknown;
  toolOutput?: unknown;
  timestamp: number;
};

type AgentMessage = {
  id: string;
  role: MessageRole;
  parts: MessagePart[];
  trajectory?: TrajectoryEntry[];
  replyTo?: string;
  timestamp: number;
};

function createMessage(
  role: MessageRole,
  parts: MessagePart[],
  replyTo?: string
): AgentMessage {
  return {
    id: crypto.randomUUID(),
    role,
    parts,
    replyTo,
    timestamp: Date.now(),
  };
}

function textMessage(role: MessageRole, text: string): AgentMessage {
  return createMessage(role, [{ kind: "text", text }]);
}
```

注意：`MessagePart` 是多模态的（文本、结构化数据、文件），就像真实的A2A和ACP规范一样。`TrajectoryEntry` 捕获推理链，与ACP的TrajectoryMetadata匹配。

### 步骤2：A2A智能体卡片和注册表

构建符合真实A2A规范的智能体发现：

```typescript
type Skill = {
  id: string;
  name: string;
  description: string;
  tags: string[];
  inputModes: string[];
  outputModes: string[];
};

type AgentCard = {
  name: string;
  description: string;
  version: string;
  url: string;
  capabilities: {
    streaming: boolean;
    pushNotifications: boolean;
  };
  defaultInputModes: string[];
  defaultOutputModes: string[];
  skills: Skill[];
};

class AgentRegistry {
  private cards: Map<string, AgentCard> = new Map();

  register(card: AgentCard) {
    this.cards.set(card.name, card);
  }

  discoverBySkillTag(tag: string): AgentCard[] {
    return [...this.cards.values()].filter((card) =>
      card.skills.some((skill) => skill.tags.includes(tag))
    );
  }

  discoverByInputMode(mimeType: string): AgentCard[] {
    return [...this.cards.values()].filter(
      (card) =>
        card.defaultInputModes.includes(mimeType) ||
        card.skills.some((skill) => skill.inputModes.includes(mimeType))
    );
  }

  resolve(name: string): AgentCard | undefined {
    return this.cards.get(name);
  }

  listAll(): AgentCard[] {
    return [...this.cards.values()];
  }
}
```

这比简单的名称到能力映射丰富得多。你可以通过技能标签、输入MIME类型或名称发现智能体，就像真实的A2A规范支持的那样。

### 步骤3：A2A任务生命周期

构建完整的任务状态机：

```typescript
type TaskState =
  | "submitted"
  | "working"
  | "input-required"
  | "auth-required"
  | "completed"
  | "failed"
  | "canceled"
  | "rejected";

const TERMINAL_STATES: TaskState[] = [
  "completed",
  "failed",
  "canceled",
  "rejected",
];

type TaskStatus = {
  state: TaskState;
  message?: AgentMessage;
  timestamp: number;
};

type Artifact = {
  id: string;
  name: string;
  parts: MessagePart[];
};

type Task = {
  id: string;
  contextId: string;
  status: TaskStatus;
  artifacts: Artifact[];
  history: AgentMessage[];
};

type TaskEvent =
  | { kind: "statusUpdate"; taskId: string; status: TaskStatus }
  | {
      kind: "artifactUpdate";
      taskId: string;
      artifact: Artifact;
      append: boolean;
      lastChunk: boolean;
    };

type TaskHandler = (
  task: Task,
  message: AgentMessage
) => AsyncGenerator<TaskEvent>;

class TaskManager {
  private tasks: Map<string, Task> = new Map();
  private handlers: Map<string, TaskHandler> = new Map();
  private listeners: Map<string, ((event: TaskEvent) => void)[]> = new Map();

  registerHandler(agentName: string, handler: TaskHandler) {
    this.handlers.set(agentName, handler);
  }

  subscribe(taskId: string, listener: (event: TaskEvent) => void) {
    const existing = this.listeners.get(taskId) ?? [];
    existing.push(listener);
    this.listeners.set(taskId, existing);
  }

  async sendMessage(
    agentName: string,
    message: AgentMessage,
    contextId?: string
  ): Promise<Task> {
    const handler = this.handlers.get(agentName);
    if (!handler) {
      const task = this.createTask(contextId);
      task.status = {
        state: "rejected",
        timestamp: Date.now(),
        message: textMessage("agent", `No handler for ${agentName}`),
      };
      return task;
    }

    const task = this.createTask(contextId);
    task.history.push(message);
    task.status = { state: "submitted", timestamp: Date.now() };

    this.processTask(task, handler, message).catch((err) => {
      task.status = {
        state: "failed",
        timestamp: Date.now(),
        message: textMessage("agent", String(err)),
      };
    });
    return task;
  }

  getTask(taskId: string): Task | undefined {
    return this.tasks.get(taskId);
  }

  cancelTask(taskId: string): boolean {
    const task = this.tasks.get(taskId);
    if (!task || TERMINAL_STATES.includes(task.status.state)) return false;
    task.status = { state: "canceled", timestamp: Date.now() };
    this.emit(taskId, {
      kind: "statusUpdate",
      taskId,
      status: task.status,
    });
    return true;
  }

  private createTask(contextId?: string): Task {
    const task: Task = {
      id: crypto.randomUUID(),
      contextId: contextId ?? crypto.randomUUID(),
      status: { state: "submitted", timestamp: Date.now() },
      artifacts: [],
      history: [],
    };
    this.tasks.set(task.id, task);
    return task;
  }

  private async processTask(
    task: Task,
    handler: TaskHandler,
    message: AgentMessage
  ) {
    task.status = { state: "working", timestamp: Date.now() };
    this.emit(task.id, {
      kind: "statusUpdate",
      taskId: task.id,
      status: task.status,
    });

    try {
      for await (const event of handler(task, message)) {
        if (TERMINAL_STATES.includes(task.status.state)) break;

        if (event.kind === "statusUpdate") {
          task.status = event.status;
        }
        if (event.kind === "artifactUpdate") {
          const existing = task.artifacts.find(
            (a) => a.id === event.artifact.id
          );
          if (existing && event.append) {
            existing.parts.push(...event.artifact.parts);
          } else {
            task.artifacts.push(event.artifact);
          }
        }
        this.emit(task.id, event);
      }
    } catch (err) {
      task.status = {
        state: "failed",
        timestamp: Date.now(),
        message: textMessage("agent", String(err)),
      };
      this.emit(task.id, {
        kind: "statusUpdate",
        taskId: task.id,
        status: task.status,
      });
    }
  }

  private emit(taskId: string, event: TaskEvent) {
    for (const listener of this.listeners.get(taskId) ?? []) {
      listener(event);
    }
  }
}
```

这实现了真实的A2A任务生命周期：已提交、处理中、需要输入、终态。处理器是异步生成器，产生事件（状态更新和工件块），匹配SSE流模型。

### 步骤4：ACP风格的审计轨迹

用轨迹跟踪包装通信：

```typescript
type AuditEntry = {
  runId: string;
  agentName: string;
  input: AgentMessage[];
  output: AgentMessage[];
  trajectory: TrajectoryEntry[];
  status: "created" | "in-progress" | "completed" | "failed" | "awaiting";
  startedAt: number;
  completedAt?: number;
  sessionId?: string;
};

class AuditableRunner {
  private log: AuditEntry[] = [];
  private handlers: Map<
    string,
    (input: AgentMessage[]) => Promise<{
      output: AgentMessage[];
      trajectory: TrajectoryEntry[];
    }>
  > = new Map();

  registerAgent(
    name: string,
    handler: (input: AgentMessage[]) => Promise<{
      output: AgentMessage[];
      trajectory: TrajectoryEntry[];
    }>
  ) {
    this.handlers.set(name, handler);
  }

  async run(
    agentName: string,
    input: AgentMessage[],
    sessionId?: string
  ): Promise<AuditEntry> {
    const entry: AuditEntry = {
      runId: crypto.randomUUID(),
      agentName,
      input: structuredClone(input),
      output: [],
      trajectory: [],
      status: "created",
      startedAt: Date.now(),
      sessionId,
    };
    this.log.push(entry);

    const handler = this.handlers.get(agentName);
    if (!handler) {
      entry.status = "failed";
      return entry;
    }

    entry.status = "in-progress";
    try {
      const result = await handler(input);
      entry.output = structuredClone(result.output);
      entry.trajectory = structuredClone(result.trajectory);
      entry.status = "completed";
      entry.completedAt = Date.now();
    } catch (err) {
      entry.status = "failed";
      entry.trajectory.push({
        reasoning: `Error: ${String(err)}`,
        timestamp: Date.now(),
      });
      entry.completedAt = Date.now();
    }
    return entry;
  }

  getFullAuditLog(): AuditEntry[] {
    return structuredClone(this.log);
  }

  getAuditLogForAgent(agentName: string): AuditEntry[] {
    return structuredClone(
      this.log.filter((e) => e.agentName === agentName)
    );
  }

  getAuditLogForSession(sessionId: string): AuditEntry[] {
    return structuredClone(
      this.log.filter((e) => e.sessionId === sessionId)
    );
  }

  getTrajectoryForRun(runId: string): TrajectoryEntry[] {
    const entry = this.log.find((e) => e.runId === runId);
    return entry ? structuredClone(entry.trajectory) : [];
  }
}
```

每次智能体执行都产生一个完整的审计条目：输入了什么，输出了什么，以及期间工具调用和推理步骤的完整轨迹。你可以按智能体、会话或单个运行进行查询。

### 步骤5：ANP风格的身份验证

构建基于DID的身份和验证：

```typescript
type VerificationMethod = {
  id: string;
  type: string;
  controller: string;
  publicKeyDer: string;
};

type DIDDocument = {
  id: string;
  verificationMethod: VerificationMethod[];
  authentication: string[];
  keyAgreement: string[];
  humanAuthorization: string[];
  service: { id: string; type: string; serviceEndpoint: string }[];
};

type AgentIdentity = {
  did: string;
  document: DIDDocument;
  privateKey: crypto.KeyObject;
  publicKey: crypto.KeyObject;
};

class IdentityRegistry {
  private documents: Map<string, DIDDocument> = new Map();

  publish(doc: DIDDocument) {
    this.documents.set(doc.id, doc);
  }

  resolve(did: string): DIDDocument | undefined {
    return this.documents.get(did);
  }

  verify(did: string, signature: string, payload: string): boolean {
    const doc = this.documents.get(did);
    if (!doc) return false;

    const authKeyIds = doc.authentication;
    const authKeys = doc.verificationMethod.filter((vm) =>
      authKeyIds.includes(vm.id)
    );

    for (const key of authKeys) {
      const publicKey = crypto.createPublicKey({
        key: Buffer.from(key.publicKeyDer, "base64"),
        format: "der",
        type: "spki",
      });
      const isValid = crypto.verify(
        null,
        Buffer.from(payload),
        publicKey,
        Buffer.from(signature, "hex")
      );
      if (isValid) return true;
    }
    return false;
  }

  requiresHumanAuth(did: string, operationKeyId: string): boolean {
    const doc = this.documents.get(did);
    if (!doc) return false;
    return doc.humanAuthorization.includes(operationKeyId);
  }
}

function createIdentity(domain: string, agentName: string): AgentIdentity {
  const did = `did:wba:${domain}:agent:${agentName}`;
  const { publicKey, privateKey } = crypto.generateKeyPairSync("ed25519");

  const publicKeyDer = publicKey
    .export({ format: "der", type: "spki" })
    .toString("base64");

  const keyId = `${did}#key-1`;
  const encKeyId = `${did}#key-x25519-1`;

  const document: DIDDocument = {
    id: did,
    verificationMethod: [
      {
        id: keyId,
        type: "Ed25519VerificationKey2020",
        controller: did,
        publicKeyDer,
      },
      {
        id: encKeyId,
        type: "X25519KeyAgreementKey2019",
        controller: did,
        publicKeyDer,
      },
    ],
    authentication: [keyId],
    keyAgreement: [encKeyId],
    humanAuthorization: [],
    service: [
      {
        id: `${did}#agent-description`,
        type: "AgentDescription",
        serviceEndpoint: `https://${domain}/agents/${agentName}/ad.json`,
      },
    ],
  };

  return { did, document, privateKey, publicKey };
}

function signPayload(identity: AgentIdentity, payload: string): string {
  return crypto
    .sign(null, Buffer.from(payload), identity.privateKey)
    .toString("hex");
}
```

这反映了真实的ANP身份模型：智能体拥有DID文档，其中包含单独的身份验证、密钥协商和人类授权密钥。`IdentityRegistry` 模拟了DID解析（在生产中，这将是到智能体域的HTTP请求）。

### 步骤6：协议网关

将所有四种协议连接成一个统一的系统：

```mermaid
graph LR
    REQ[Incoming Request] --> ANP_V{ANP: Verify DID}
    ANP_V -->|Valid| A2A_D{A2A: Discover Agent}
    ANP_V -->|Invalid| REJECT[Reject]
    A2A_D -->|Found| ACP_A[ACP: Audit Run]
    A2A_D -->|Not Found| REJECT
    ACP_A --> A2A_T[A2A: Create Task]
    A2A_T --> RESULT[Task + Audit Entry]

    style ANP_V fill:#d1fae5,stroke:#059669
    style A2A_D fill:#dbeafe,stroke:#2563eb
    style ACP_A fill:#fef3c7,stroke:#d97706
    style A2A_T fill:#dbeafe,stroke:#2563eb
```

```typescript
class ProtocolGateway {
  private registry: AgentRegistry;
  private taskManager: TaskManager;
  private auditRunner: AuditableRunner;
  private identityRegistry: IdentityRegistry;

  constructor(
    registry: AgentRegistry,
    taskManager: TaskManager,
    auditRunner: AuditableRunner,
    identityRegistry: IdentityRegistry
  ) {
    this.registry = registry;
    this.taskManager = taskManager;
    this.auditRunner = auditRunner;
    this.identityRegistry = identityRegistry;
  }

  async delegateTask(
    fromDid: string,
    signature: string,
    targetAgent: string,
    message: AgentMessage,
    sessionId?: string
  ): Promise<{ task: Task; audit: AuditEntry } | { error: string }> {
    if (!this.identityRegistry.verify(fromDid, signature, message.id)) {
      return { error: "Identity verification failed" };
    }

    const card = this.registry.resolve(targetAgent);
    if (!card) {
      return { error: `Agent ${targetAgent} not found in registry` };
    }

    const audit = await this.auditRunner.run(
      targetAgent,
      [message],
      sessionId
    );
    const task = await this.taskManager.sendMessage(targetAgent, message);

    return { task, audit };
  }

  discoverAndDelegate(
    fromDid: string,
    signature: string,
    skillTag: string,
    message: AgentMessage
  ): Promise<{ task: Task; audit: AuditEntry } | { error: string }> {
    const candidates = this.registry.discoverBySkillTag(skillTag);
    if (candidates.length === 0) {
      return Promise.resolve({
        error: `No agents found with skill tag: ${skillTag}`,
      });
    }
    return this.delegateTask(
      fromDid,
      signature,
      candidates[0].name,
      message
    );
  }
}
```

网关在一次调用中做四件事：
1. **ANP**：通过DID签名验证调用者身份
2. **A2A**：发现目标智能体并检查能力
3. **ACP**：用轨迹包装执行进行审计
4. **A2A**：创建具有完整生命周期跟踪的任务

### 步骤7：全部连接起来

```typescript
async function protocolDemo() {
  const registry = new AgentRegistry();
  registry.register({
    name: "researcher",
    description: "Searches and summarizes findings",
    version: "1.0.0",
    url: "https://researcher.local/a2a/v1",
    capabilities: { streaming: true, pushNotifications: false },
    defaultInputModes: ["text/plain"],
    defaultOutputModes: ["text/plain", "application/json"],
    skills: [
      {
        id: "web-research",
        name: "Web Research",
        description: "Searches the web",
        tags: ["research", "search", "summarization"],
        inputModes: ["text/plain"],
        outputModes: ["application/json"],
      },
    ],
  });
  registry.register({
    name: "coder",
    description: "Writes code from specs",
    version: "1.0.0",
    url: "https://coder.local/a2a/v1",
    capabilities: { streaming: false, pushNotifications: false },
    defaultInputModes: ["text/plain", "application/json"],
    defaultOutputModes: ["text/plain"],
    skills: [
      {
        id: "code-gen",
        name: "Code Generation",
        description: "Generates code",
        tags: ["coding", "generation"],
        inputModes: ["text/plain", "application/json"],
        outputModes: ["text/plain"],
      },
    ],
  });

  const taskManager = new TaskManager();
  const auditRunner = new AuditableRunner();

  const researchTrajectory: TrajectoryEntry[] = [];

  taskManager.registerHandler(
    "researcher",
    async function* (task, message) {
      yield {
        kind: "statusUpdate" as const,
        taskId: task.id,
        status: { state: "working" as const, timestamp: Date.now() },
      };

      researchTrajectory.push({
        reasoning: "Searching for React 19 documentation",
        toolName: "web_search",
        toolInput: { query: "React 19 compiler features" },
        toolOutput: {
          results: ["react.dev/blog/react-19", "github.com/react/react"],
        },
        timestamp: Date.now(),
      });

      researchTrajectory.push({
        reasoning: "Extracting key findings from search results",
        toolName: "doc_analysis",
        toolInput: { url: "react.dev/blog/react-19" },
        toolOutput: {
          summary:
            "React 19 compiler auto-memoizes, no manual useMemo needed",
        },
        timestamp: Date.now(),
      });

      yield {
        kind: "artifactUpdate" as const,
        taskId: task.id,
        artifact: {
          id: crypto.randomUUID(),
          name: "research-results",
          parts: [
            {
              kind: "data" as const,
              data: {
                findings: [
                  "React 19 compiler auto-memoizes components",
                  "No more manual useMemo/useCallback needed",
                  "Compiler runs at build time, not runtime",
                ],
                sources: ["react.dev/blog/react-19"],
              },
              mediaType: "application/json",
            },
          ],
        },
        append: false,
        lastChunk: true,
      };

      yield {
        kind: "statusUpdate" as const,
        taskId: task.id,
        status: { state: "completed" as const, timestamp: Date.now() },
      };
    }
  );

  auditRunner.registerAgent("researcher", async () => ({
    output: [
      textMessage("agent", "React 19 compiler auto-memoizes components"),
    ],
    trajectory: researchTrajectory,
  }));

  const identityRegistry = new IdentityRegistry();

  const coderIdentity = createIdentity("coder.local", "coder");
  const researcherIdentity = createIdentity("researcher.local", "researcher");

  identityRegistry.publish(coderIdentity.document);
  identityRegistry.publish(researcherIdentity.document);

  const gateway = new ProtocolGateway(
    registry,
    taskManager,
    auditRunner,
    identityRegistry
  );

  console.log("=== Protocol Demo ===\n");

  console.log("1. Agent Discovery (A2A)");
  const researchAgents = registry.discoverBySkillTag("research");
  console.log(
    `   Found ${researchAgents.length} agent(s):`,
    researchAgents.map((a) => a.name)
  );

  console.log("\n2. Identity Verification (ANP)");
  const message = textMessage("user", "Research React 19 compiler features");
  const signature = signPayload(coderIdentity, message.id);
  const verified = identityRegistry.verify(
    coderIdentity.did,
    signature,
    message.id
  );
  console.log(`   Coder DID: ${coderIdentity.did}`);
  console.log(`   Signature verified: ${verified}`);

  console.log("\n3. Task Delegation (A2A + ACP + ANP)");
  const result = await gateway.delegateTask(
    coderIdentity.did,
    signature,
    "researcher",
    message,
    "session-001"
  );

  if ("error" in result) {
    console.log(`   Error: ${result.error}`);
    return;
  }

  console.log(`   Task ID: ${result.task.id}`);
  console.log(`   Task state: ${result.task.status.state}`);
  console.log(`   Artifacts: ${result.task.artifacts.length}`);

  console.log("\n4. Audit Trail (ACP)");
  console.log(`   Run ID: ${result.audit.runId}`);
  console.log(`   Status: ${result.audit.status}`);
  console.log(`   Trajectory steps: ${result.audit.trajectory.length}`);
  for (const step of result.audit.trajectory) {
    console.log(`     - ${step.reasoning}`);
    if (step.toolName) {
      console.log(`       Tool: ${step.toolName}`);
    }
  }

  console.log("\n5. Full Audit Log");
  const fullLog = auditRunner.getFullAuditLog();
  console.log(`   Total runs: ${fullLog.length}`);
  for (const entry of fullLog) {
    const duration = entry.completedAt
      ? `${entry.completedAt - entry.startedAt}ms`
      : "in-progress";
    console.log(`   ${entry.agentName}: ${entry.status} (${duration})`);
  }
}

protocolDemo().catch((err) => {
  console.error("Protocol demo failed:", err);
  process.exitCode = 1;
});
```

## 可能出错的地方

协议解决了快乐路径。以下是生产中可能出错的地方：

**模式漂移。** 智能体A发布智能体卡片，宣传`application/json`输出。但JSON模式在不同版本间发生变化。智能体B解析旧格式，得到垃圾数据。修复：对你的技能和输出模式进行版本控制。A2A规范支持`version`就是为此。

**状态机违规。** 智能体处理器产生一个`completed`事件，然后试图产生更多工件。任务是不可变的。你的代码会静默丢弃更新或抛出异常。修复：在产生前检查终态。上面的`TaskManager`通过终态后的`break`强制执行此操作。

**信任解析失败。** 智能体A试图验证智能体B的DID，但智能体B的域名宕机。无法获取DID文档。你是选择失败开放（接受未验证的智能体）还是失败关闭（拒绝一切）？ANP推荐使用最小信任原则的失败关闭。

**轨迹膨胀。** ACP轨迹记录功能强大但代价高昂。一个每次运行进行200次工具调用的复杂智能体会产生庞大的审计条目。修复：以可配置的详细程度记录轨迹。为合规性记录工具名称和IO，对非受监管工作负载跳过推理步骤。

**发现风暴。** 50个智能体在启动时同时查询`GET /agents`。修复：缓存带TTL的智能体卡片，错开发现间隔，或使用基于推送的注册而不是轮询。

## 应用

### 真实实现

**A2A** 最成熟。Google的[官方规范](https://github.com/google/A2A)在Linux基金会下开源。有Python和TypeScript的SDK。如果你的智能体需要动态发现和协作，从这里开始。

**ACP** 正在合并到A2A中。IBM的[BeeAI项目](https://github.com/i-am-bee/acp)创建了ACP作为REST优先的替代方案，但轨迹元数据概念正被吸收到A2A生态系统中。即使你使用A2A作为传输层，也可以使用ACP模式（轨迹记录、运行生命周期）。

**ANP** 最具实验性。[社区仓库](https://github.com/agent-network-protocol/AgentNetworkProtocol)有一个Python SDK（AgentConnect）。元协议协商概念非常新颖。值得在跨组织智能体部署中关注。

**MCP** 已经在第13阶段涵盖。如果你希望智能体使用工具，MCP是标准。

### 选择正确的协议

```mermaid
graph TD
    START{Do agents need<br/>to use tools?}
    START -->|Yes| MCP_R[Use MCP]
    START -->|No| TALK{Do agents need to<br/>talk to each other?}
    TALK -->|No| NONE[You don't need<br/>a protocol]
    TALK -->|Yes| AUDIT{Need audit trails<br/>for compliance?}
    AUDIT -->|Yes| ACP_R[A2A + ACP<br/>trajectory patterns]
    AUDIT -->|No| ORG{All agents<br/>within your org?}
    ORG -->|Yes| A2A_R[A2A<br/>Agent Cards + Tasks]
    ORG -->|No| INFRA{Shared<br/>infrastructure?}
    INFRA -->|Yes| BROKER[A2A + message broker]
    INFRA -->|No| ANP_R[ANP + A2A<br/>DID verification]

    style MCP_R fill:#d1fae5,stroke:#059669
    style A2A_R fill:#dbeafe,stroke:#2563eb
    style ACP_R fill:#fef3c7,stroke:#d97706
    style ANP_R fill:#f3e8ff,stroke:#7c3aed
    style BROKER fill:#e0e7ff,stroke:#4338ca
```

## 交付

本课程产出：
- `code/main.ts` -- 所有四种协议模式的完整实现
- `outputs/prompt-protocol-selector.md` -- 一个帮助你为系统选择协议的提示

## 练习

1. **多跳任务委派。** 扩展`TaskManager`，使智能体处理器可以将子任务委派给其他智能体。研究员接收一个任务，将"搜索"和"总结"子任务委派给两个专家智能体，等待两者完成，然后将结果合并到自己的工件中。

2. **流式审计轨迹。** 修改`AuditableRunner`以支持流模式。不等待完整结果，而是随着轨迹条目添加实时产生`AuditEntry`更新。使用异步生成器产生审计快照。

3. **DID轮换。** 向`IdentityRegistry`添加密钥轮换。智能体应该能够发布具有更新密钥的新DID文档，同时维护`previousDid`引用。在宽限期内，验证者应接受当前和先前密钥的签名。

4. **协议协商。** 实现ANP的元协议概念。两个智能体交换`protocolNegotiation`消息，携带候选格式（例如，"我能说JSON-RPC"与"我更喜欢REST"）。最多3轮后，它们就格式达成一致或超时。商定的格式决定它们使用哪个`TaskManager`或`AuditableRunner`。

5. **限速发现。** 添加一个`RateLimitedRegistry`包装器，缓存带可配置TTL的智能体卡片查找，并限制每个智能体每秒的发现查询。模拟100个智能体在启动时发现彼此的风暴，并测量差异。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| MCP | "AI工具的协议" | 一个客户端-服务器协议，用于智能体发现和使用工具。智能体到工具，不是智能体到智能体。 |
| A2A | "Google的智能体协议" | Linux基金会下的对等协议，用于智能体协作。通过智能体卡片发现，9状态任务生命周期，通过SSE流式传输。支持JSON-RPC、REST和gRPC绑定。 |
| ACP | "企业智能体消息" | IBM/BeeAI的REST API，用于智能体运行，带有TrajectoryMetadata：每个响应携带完整的推理和工具调用链。正在合并到A2A。 |
| ANP | "去中心化智能体身份" | 一个社区协议，使用`did:wba`（DID）实现密码学身份，使用HPKE实现端到端加密，并使用AI驱动的元协议协商处理从未见过面的智能体。 |
| 智能体卡片 | "智能体的名片" | 位于`/.well-known/agent-card.json`的JSON文档，描述技能、支持的MIME类型、安全方案和协议绑定。 |
| DID | "去中心化ID" | W3C标准，用于在智能体自己的域名上托管的可密码学验证身份。ANP使用`did:wba`方法。 |
| TrajectoryMetadata | "审计收据" | ACP的机制，用于将推理步骤、工具调用及其输入/输出附加到每个智能体响应。 |
| 元协议 | "智能体协商如何交谈" | ANP的方法，智能体使用自然语言动态商定数据格式，然后生成代码来处理它们。 |
| 任务 | "工作单元" | A2A的有状态对象，跟踪从提交到完成的工作。达到终态后不可变。 |

## 延伸阅读

- [Google A2A规范](https://github.com/google/A2A) -- 官方规范和SDK (v1.0.0, Linux基金会)
- [IBM/BeeAI ACP规范](https://github.com/i-am-bee/acp) -- OpenAPI 3.1规范，用于智能体运行和轨迹元数据
- [智能体网络协议](https://github.com/agent-network-protocol/AgentNetworkProtocol) -- 基于DID的身份、端到端加密、元协议协商
- [模型上下文协议文档](https://modelcontextprotocol.io/) -- Anthropic的MCP规范（在第13阶段涵盖）
- [W3C去中心化标识符](https://www.w3.org/TR/did-core/) -- 支撑ANP的身份标准
- [RFC 9180 (HPKE)](https://www.rfc-editor.org/rfc/rfc9180) -- ANP用于端到端加密的加密方案
- [FIPA智能体通信语言](http://www.fipa.org/specs/fipa00061/SC00061G.html) -- 现代智能体协议的学术前身