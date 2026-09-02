# 第13阶段：工具与协议

> AI 与现实世界之间的接口。

本阶段从函数调用和工具模式进入互操作性协议、Agent Skills、安全性和生产治理领域。数字顺序便于浏览，但以下专注路线是更可靠的学习顺序。

## 在 GitHub 上开始本阶段

**前置要求：** 第11阶段 LLM completion APIs。如需使用 MCP 或 Agent Skills，请采用下方的专注路线，而非假设按数字顺序学习课程。

**第一阶段完整课程：** [工具接口](01-the-tool-interface/)

从仓库根目录运行此命令：

```bash
python3 phases/13-tools-and-protocols/01-the-tool-interface/code/main.py
```

保留命令、退出码、describe-decide-execute-observe 追踪、被拒绝输入的示例证据，以及一句解释回合限制的话。

**下一步：** 继续进入 [函数调用深入探究](02-function-calling-deep-dive/)，或选择下方 Model Context Protocol (MCP) 或 Agent Skills 路线。

浏览 [第13阶段完整课程列表](../../README.md#phase-13) 或 [跨阶段路线图](../../ROADMAP.md)。

## Model Context Protocol (MCP) 路线

专注的 MCP 路线共 17 个课程，约 23 小时 15 分钟。它遵循 MCP `2026-07-28`，从一个自描述的 JSON-RPC 请求到一个可操作的合规性关口。

| 阶段 | 课程 | 验证内容 | 时长 |
|---|---|---|---:|
| 核心 | [06](06-mcp-fundamentals/)、[07](07-building-an-mcp-server/)、[08](08-building-an-mcp-client/)、[09](09-mcp-transports/)、[10](10-mcp-resources-and-prompts/) | 信封、发现、客户端和服务端行为、传输协议、资源和提示词。 | 5 小时 50 分钟 |
| 双向 | [11](11-mcp-sampling/)、[12](12-mcp-roots-and-elicitation/)、[13](13-mcp-async-tasks/)、[14](14-mcp-apps/) | MRTR 输入、显式范围、持久化任务和无需服务端发起请求的应用边界。 | 5 小时 |
| 安全 | [15](15-mcp-security-tool-poisoning/)、[16](16-mcp-security-oauth-2-1/)、[18](18-mcp-auth-production/)、[17](17-mcp-gateways-and-registries/) | 投毒防御、授权、生产令牌、网关路由和注册表准入。 | 5 小时 15 分钟 |
| 高级 | [28](28-mcp-tool-contracts-and-content/)、[29](29-mcp-reliability-cancellation-and-flow-control/)、[30](30-mcp-registry-supply-chain-and-drift/)、[31](31-mcp-conformance-versioning-and-operations/) | 契约保真度、取消竞态、供应链漂移和发布证据。 | 7 小时 10 分钟 |

确切顺序为 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16, 18, 17, 28,
29, 30, 31。该顺序定义在
[`learning-paths/model-context-protocol.json`](../../learning-paths/model-context-protocol.json)。
助教每次调用教授一个课程，并记录每次检查点要求的请求、响应、命令、工作目录、退出码以及脱敏后的边界证据。

从你的主机支持的调用开始：

| 主机 | 调用方式 |
|---|---|
| Codex | `learn-mcp`，或从 `/skills` 中选择 |
| Claude Code | `/learn-mcp` |
| 其他兼容主机 | `Use learn-mcp to start or resume the Model Context Protocol (MCP) path.` |

### 你的最初十分钟

从仓库根目录运行第 06 课的无状态转录：

```bash
python3 phases/13-tools-and-protocols/06-mcp-fundamentals/code/main.py
```

在输出中找到四个要素：重复的请求元数据、完整的
`server/discover` 结果、对不支持版本的错误 `-32022`，以及一个不会创建或终止 MCP 协议会话的传输关闭。
这份转录本身就是第一个检查点，而不仅仅是一个演示。

如果仓库或 Python 3 不可用，请阅读 [第 06 课](06-mcp-fundamentals/)
并手动追踪一个请求和响应过程。将该检查点标记为概念性完成，并保留运行时、传输协议、授权和部署证据处于待定状态。

在任何非回环绑定、共享入口、托管端点或注册表发布之前，先完成第 15 课的可执行安全检查点。审查外部目标和所请求的权限，然后明确确认部署操作。完成的教程并不授予部署权限。

旧的 `initialize`、`Mcp-Session-Id`、独立 SSE `GET`、会话 `DELETE` 和由服务端发起的请求流程仅出现在明确的兼容性说明中。现代请求会在
`params._meta` 中声明协议版本和客户端能力，使用 `server/discover`，并携带足够的信息以便独立进行验证、授权、路由和重试。

[第 23 课](23-capstone-tool-ecosystem/) 是唯一的可选 MCP 路线毕业设计。在开始之前，先完成 17 个必修课程以及 [第 19 课](19-a2a-protocol/) 和 [第 20 课](20-opentelemetry-genai/)。

## Agent Skills 快速路径

专注路线共五个课程，约 9 小时 30 分钟：

| 步骤 | 课程 | 成果 | 时长 |
|---:|---|---|---:|
| 1 | [22: 可移植契约与运行时边界](22-skills-and-agent-sdks/) | 创建、安装、调用、验证和移除一个完整的技能包。 | 90 分钟 |
| 2 | [24: 发现与渐进式披露](24-skill-discovery-and-progressive-disclosure/) | 追踪发现、编目、激活和资源加载。 | 105 分钟 |
| 3 | [25: 调用与路由](25-skill-invocation-and-routing/) | 控制显式、隐式、人工、模型和弃权路径。 | 105 分钟 |
| 4 | [26: 权限、沙箱与信任](26-skill-permissions-sandboxes-and-trust/) | 分离指令、权限、隔离和验证。 | 120 分钟 |
| 5 | [27: 评测、打包与可移植性](27-skill-evals-packaging-and-portability/) | 构建发布关口并在真实主机中验证行为。 | 150 分钟 |

从你的主机支持的调用开始：

| 主机 | 调用方式 |
|---|---|
| Codex | `learn-agent-skills`，或从 `/skills` 中选择 |
| Claude Code | `/learn-agent-skills` |
| 其他兼容主机 | `Use learn-agent-skills to start or resume the Agent Skills Engineering path.` |

助教会创建或恢复 `AGENT-SKILLS-LEARNING.md`，每次调用教授一个课程，并记录每次检查点所需的证据。该路线定义在
[`learning-paths/agent-skills.json`](../../learning-paths/agent-skills.json)。

如果你更倾向于先阅读，可以从 [第 22 课](22-skills-and-agent-sdks/) 开始。
它的第一个实验能在大约十分钟内将一个技能包接入真实主机。

### 前置快速通道

- 进行真实实验需要 `node`、`npx`、`python3`、一个已选定的支持技能的主机，以及对所选项目或用户技能范围的写权限。在安装前，用 `node --version`、`npx --version` 和
  `python3 --version` 验证这三个命令。
- 如果前置检查不可用，请使用网站或手动阅读每个 `docs/en.md`。你可以完成概念性工作，但请将发现、调用、脚本、更新和卸载证据标记为待定。
- 如果工具契约对你来说是新概念，请略读 [第 01 课](01-the-tool-interface/) 和 [第 05 课](05-tool-schema-design/)。
- 在第 26 课之前，确认你能解释工具投毒和不受信任的指令。[第 15 课](15-mcp-security-tool-poisoning/) 是可选的复习材料，而非本路线的第六个必修课程。
- [第 23 课](23-capstone-tool-ecosystem/) 是一个可选的系统毕业设计，不是第 22 课之后的下一个 Agent Skills 课程。完成第 06 至 20 课后再进行。

## 完整阶段

查看 [ROADMAP.md](../../ROADMAP.md) 获取完整课程计划。
