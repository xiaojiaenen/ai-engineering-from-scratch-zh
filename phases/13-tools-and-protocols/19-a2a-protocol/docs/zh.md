# A2A — 智能体间协议

> MCP 实现智能体与工具的交互。A2A（智能体间协议）则面向智能体与智能体的协作——这是一个开放协议，让基于不同框架构建的不透明智能体能够协同工作。该协议由谷歌于2025年4月发布，同年6月捐赠给Linux基金会，并于2026年4月推出v1.0版本，获得超过150家支持者（包括AWS、Cisco、微软、Salesforce、SAP和ServiceNow）。它吸收了IBM的ACP协议，并新增了AP2支付扩展。本课将详细讲解智能体卡片、任务生命周期以及两种传输绑定方式。

**类型:** 构建类
**语言:** Python（标准库、智能体卡片+任务工具）
**先决条件:** 第13阶段第06课（MCP基础）、第13阶段第08课（MCP客户端）
**时长:** 约75分钟

## 学习目标

- 区分智能体与工具（MCP）和智能体与智能体（A2A）的应用场景。
- 在 `/.well-known/agent.json` 发布包含技能和端点元数据的智能体卡片。
- 掌握任务生命周期（提交→处理中→需要输入→完成/失败/取消/拒绝）。
- 使用包含部件（文本、文件、数据）的消息和工件作为输出。

## 问题场景

客服智能体需要将报告撰写任务委托给专业的写作智能体。在A2A出现之前的解决方案包括：

- 自定义REST API。可行但每组配对都是独立实现。
- 共享代码库。要求两个智能体运行相同框架。
- MCP。不适用：MCP用于调用工具，而非两个智能体在保持各自内部推理不透明的情况下协作。

A2A填补了这一空白。它将交互建模为一个智能体向另一个智能体发送任务，包含生命周期、消息和工件。被调用智能体的内部状态保持不透明——调用方只能看到任务状态转换和最终输出。

A2A是"让跨框架智能体互相对话"的协议。它不取代MCP，两者互补。

## 核心概念

### 智能体卡片

每个A2A兼容的智能体都在 `/.well-known/agent.json` 发布卡片：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

通过URL发现机制：获取卡片，了解A2A端点URL，枚举技能列表。

### 签名智能体卡片（AP2）

AP2扩展（2025年9月）为智能体卡片添加了加密签名。发布方用JWT签名自己的卡片；消费方进行验证。可防止冒充。

### 任务生命周期

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

客户端通过 `tasks/send` 发起请求。被调用智能体在状态间转换；客户端通过SSE订阅或轮询来获取状态更新。

### 消息与部件

消息包含一个或多个部件：

- `text` — 纯文本内容。
- `file` — 带MIME类型的base64二进制数据。
- `data` — 带类型的JSON负载（被调用智能体的结构化输入）。

示例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### 工件

输出是工件而非原始字符串。工件是带名称和类型的输出：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

工件可以分块流式传输。调用方负责聚合。

### 两种传输绑定

1. **JSON-RPC over HTTP。** `/a2a` 端点，POST用于请求，可选SSE用于流式传输。默认绑定。
2. **gRPC。** 适用于gRPC原生的企业环境。

两种绑定承载相同的逻辑消息结构。

### 不透明性保留

关键设计原则：被调用智能体的内部状态不透明。调用方只能看到任务状态和工件。被调用智能体的思维链、工具调用、子智能体委托——全部不可见。这与MCP中工具调用透明的方式不同。

原理：A2A使竞争方能够在不泄露内部实现的情况下协作。A2A可以实现"调用此客服智能体"而无需调用方了解该智能体如何实现服务。

### 发展历程

- **2025-04-09。** 谷歌宣布A2A。
- **2025-06-23。** 捐赠给Linux基金会。
- **2025-08。** 吸收IBM的ACP。
- **2025-09。** AP2扩展（智能体支付协议）发布。
- **2026-04。** v1.0版本发布，获得150+支持组织。

### 与MCP的关系

| 维度 | MCP | A2A |
|-----------|-----|-----|
| 应用场景 | 智能体与工具 | 智能体与智能体 |
| 透明度 | 工具调用透明 | 内部推理不透明 |
| 典型调用方 | 智能体运行时 | 另一个智能体 |
| 状态 | 工具调用结果 | 具有生命周期的任务 |
| 授权 | OAuth 2.1（第13阶段第16课） | JWT签名的智能体卡片（AP2） |
| 传输 | 标准输入输出/流式HTTP | JSON-RPC over HTTP/gRPC |

当需要调用特定工具时使用MCP。当需要将整个任务委托给另一个智能体时使用A2A。许多生产系统同时使用两者：智能体用MCP构建工具层，用A2A构建协作层。

## 实践应用

`code/main.py` 实现了最小化A2A框架：研究智能体发布其卡片，写作智能体接收包含PDF和文本指令部件的 `tasks/send`，经历处理中→需要输入→处理中→完成的状态转换，并返回文本工件。全部使用标准库；采用内存传输以聚焦消息结构。

关键观察点：

- 智能体卡片的JSON结构。
- 任务ID分配和状态转换。
- 包含混合类型部件的消息。
- 任务中途的"需要输入"分支。
- 完成时的工件返回。

## 产出物

本课将生成 `outputs/skill-a2a-agent-spec.md`。对于应该可被其他智能体调用的新智能体，该技能将生成智能体卡片JSON、技能模式和端点蓝图。

## 练习

1. 运行 `code/main.py`。追踪完整的任务生命周期，包括被调用智能体要求澄清时的"需要输入"暂停。

2. 添加签名智能体卡片。使用HMAC对卡片的规范JSON进行签名。编写验证器并确认对已篡改卡片的验证失败。

3. 实现任务流式传输：写作智能体通过SSE发送三个增量工件块，调用方负责聚合。

4. 设计一个包装MCP服务器的A2A智能体。将每个MCP工具映射为A2A技能。注意权衡——损失了哪些不透明性？

5. 阅读A2A v1.0公告，找出截至2026年4月尚未被任何框架实现的一个特性。（提示：与多跳任务委托相关。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| A2A | "智能体间协议" | 用于不透明智能体协作的开放协议 |
| 智能体卡片 | "`.well-known/agent.json`" | 描述智能体技能和端点的发布元数据 |
| 技能 | "可调用单元" | 智能体支持的命名操作（类似MCP工具） |
| 任务 | "委托单元" | 具有生命周期和最终工件的工作项 |
| 消息 | "任务输入" | 承载部件（文本、文件、数据） |
| 部件 | "类型化块" | `text` / `file` / `data` 消息元素 |
| 工件 | "任务输出" | 完成时返回的带名称、类型的输出 |
| AP2 | "智能体支付协议" | 用于信任和支付的签名智能体卡片扩展 |
| 不透明性 | "黑盒协作" | 被调用智能体的内部对调用方隐藏 |
| 需要输入 | "任务暂停" | 智能体需要更多信息时的生命周期状态 |

## 扩展阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A权威规范
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — 参考实现和SDK
- [Linux基金会 — A2A发布新闻稿](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025年6月治理权转移
- [Google Cloud — A2A协议升级](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — 路线图和合作伙伴进展
- [Google Dev — A2A 1.0里程碑](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0发布说明和向后兼容指导