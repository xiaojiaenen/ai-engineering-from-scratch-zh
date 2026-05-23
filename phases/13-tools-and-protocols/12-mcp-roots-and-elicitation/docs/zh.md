# 根与引导——作用域限定与中途用户输入

> 硬编码路径在用户打开不同项目时会失效。预填充的工具参数在用户未提供足够信息时会失效。根将服务器作用域限制在用户控制的一组URI内；引导则在工具调用中途暂停，通过表单或URL向用户请求结构化输入。两种客户端原语，解决常见的MCP故障模式。SEP-1036（URL模式引导，2025-11-25）在2026年上半年之前为实验性功能——使用前请检查SDK版本。

**类型：** 构建  
**语言：** Python（标准库，根与引导演示）  
**前提：** 第13阶段 · 07（MCP服务器）  
**时间：** 约45分钟

## 学习目标

- 声明`roots`并响应`notifications/roots/list_changed`。
- 将服务器文件操作限制在声明的根集合内的URI。
- 使用`elicitation/create`在工具调用中途向用户询问确认或结构化输入。
- 选择表单模式或URL模式引导（后者为实验性，存在漂移风险说明）。

## 问题所在

笔记MCP服务器在生产环境中遇到的两个具体故障。

**路径假设失效。** 服务器是针对`~/notes`编写的。在不同机器上、笔记位于`~/Documents/Notes`的用户，会遇到工具调用静默失败（未找到文件）或更糟的情况——写入错误位置。

**用户本应知道的缺失参数。** 用户要求“删除旧的TPS报告笔记”。模型调用`notes_delete(title: "TPS report")`，但存在来自2023年、2024年和2025年的三个匹配笔记。工具无法猜测。以“模糊”为由失败令人恼火；对所有三个执行操作则是灾难性的。

根解决了第一个问题：客户端在`initialize`声明服务器可以访问的URI集合。引导解决了第二个问题：服务器暂停工具调用并发送`elicitation/create`，要求用户选择其中一个。

## 概念

### 根

客户端在`initialize`声明根列表：

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

服务器随后可以调用`roots/list`：

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

服务器**必须**将根视为边界：任何超出根集合的文件读写操作都会被拒绝。这并非由客户端强制执行（服务器仍是用户信任的代码），但符合规范的服务器会遵守它。

当用户添加或删除根时，客户端发送`notifications/roots/list_changed`。服务器重新调用`roots/list`并更新其边界。

### 为何根是客户端原语

根由客户端声明，因为它们代表用户的同意模型。用户告诉Claude Desktop“允许此笔记服务器访问这两个目录”。服务器无法扩大该范围。

### 引导：表单模式默认

`elicitation/create`接受一个表单模式和自然语言提示：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

客户端渲染表单，收集用户答案，返回：

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

三种可能的操作：`accept`（用户填写完成）、`decline`（用户关闭了它）、`cancel`（用户中止了整个工具调用）。

表单模式是扁平的——v1版本不支持嵌套对象。SDK通常会拒绝比单层更复杂的结构。

### 引导：URL模式（SEP-1036，实验性）

2025-11-25新增。服务器不发送模式，而是发送一个URL：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

客户端在浏览器中打开URL，等待完成，当用户返回时返回。适用于OAuth流程、支付授权和文档签署等表单无法满足的场景。

漂移风险说明：SEP-1036响应结构仍在确定中；一些SDK返回回调URL，另一些返回完成令牌。在生产环境中使用URL模式前，请阅读您的SDK发布说明。

### 何时引导是合适工具

- 在破坏性操作前进行用户确认（破坏性提示 + 引导）。
- 消歧义（从N个匹配项中选择一个）。
- 首次运行设置（API密钥、目录、偏好设置）。
- OAuth风格的流程（URL模式）。

### 何时引导不合适

- 填充工具的必需参数，而这些参数本应由模型通过对话询问。使用正常的重新提示，而非引导对话框。
- 高频率调用。引导会中断对话；不要在循环内触发它。
- 任何服务器可以事后验证的事情。验证后返回错误，让模型在文本中询问用户。

### 人机协同桥梁

引导与采样结合，实现了MCP的“人机协同”模型。服务器的代理循环可以暂停以等待用户输入（引导）或模型推理（采样）。第13阶段 · 11涵盖了采样；本课程涵盖引导。将它们结合起来，以实现完整的中途循环控制。

## 应用它

`code/main.py`扩展笔记服务器，包含：

- `roots/list`响应，服务器在收到根列表更改通知后重新查询。
- 一个`notes_delete`工具，当有多个笔记匹配时使用`elicitation/create`进行消歧。
- 一个`notes_setup`工具，使用URL模式引导打开首次运行配置页面（模拟）。
- 边界检查，拒绝在声明的根之外的URI上执行操作。

演示运行三个场景：顺利路径（一个匹配）、消歧（三个匹配，触发引导）、超出根写入（被拒绝）。

## 交付它

本课程产出`outputs/skill-elicitation-form-designer.md`。给定一个可能需要用户确认或消歧的工具，该技能设计引导表单模式和消息模板。

## 练习

1.  运行`code/main.py`。触发消歧路径；确认模拟的用户答案是否被路由回工具。
2.  添加一个新工具`notes_archive`，该工具每次都需要引导确认（破坏性提示）。检查用户体验：这与模型在文本中重新询问相比如何？
3.  为首次运行的OAuth流程实现URL模式引导。注意漂移风险，并添加SDK版本防护。
4.  扩展`roots/list`处理：当收到通知时，服务器应原子性地重新读取和重新扫描现在可能超出作用域的已打开文件句柄。
5.  阅读GitHub上的SEP-1036问题讨论线程。找出一个影响服务器应如何处理URL模式回调的未决问题。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 根 | “同意边界” | 客户端允许服务器访问的URI |
| `roots/list` | “服务器请求作用域” | 客户端返回当前根集合 |
| `notifications/roots/list_changed` | “用户更改了作用域” | 客户端发出信号，表明根集合已发生变化 |
| 引导 | “中途询问用户” | 服务器发起的结构化用户输入请求 |
| `elicitation/create` | “该方法” | 用于引导请求的JSON-RPC方法 |
| 表单模式 | “基于模式的表单” | 在客户端UI中渲染为表单的扁平JSON模式 |
| URL模式 | “浏览器重定向” | SEP-1036实验性；打开URL并等待 |
| `accept` / `decline` / `cancel` | “用户响应结果” | 服务器处理的三个分支 |
| 消歧 | “选择一个” | 当工具有N个候选时常见的引导用例 |
| 扁平表单 | “仅顶层属性” | 引导模式不能嵌套 |

## 延伸阅读

-   [MCP — 客户端根规范](https://modelcontextprotocol.io/specification/draft/client/roots) — 权威根参考
-   [MCP — 客户端引导规范](https://modelcontextprotocol.io/specification/draft/client/elicitation) — 权威引导参考
-   [Cisco — MCP引导、结构化内容、OAuth增强的新特性](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — 2025-11-25新增内容演练
-   [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — URL模式引导提案（实验性，存在漂移风险）
-   [The New Stack — 引导如何将人机协同引入AI工具](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — 用户体验演练