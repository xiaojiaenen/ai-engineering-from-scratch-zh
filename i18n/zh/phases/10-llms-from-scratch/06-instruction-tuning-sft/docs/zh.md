# 指令微调（SFT）

> 基础模型负责预测下一个 token。仅此而已。它不会遵循指令、回答问题或拒绝有害请求。SFT 是连接 token 预测器和有用助手之间的桥梁。你曾与之对话过的每个模型——Claude、GPT、Llama Chat——都经历了这一步骤。

**类型：** Build
**语言：** Python（带 numpy）
**前置条件：** 第10阶段，课程04（预训练 Mini GPT）
**时间：** 约90分钟

## 学习目标

- 实现监督微调（SFT），将基础语言模型转变为可遵循指令的助手
- 使用包含系统、用户和助手角色的对话模板格式化训练数据，并对非助手 token 屏蔽损失
- 解释为什么需要 SFT：基础模型会继续文本而非回答问题
- 通过在保留指令集上比较基础模型与微调后模型的响应来评估 SFT 质量

## 问题所在

你在课程04中训练了一个模型。它能在给定序列时预测下一个 token。喂入 "The transformer architecture"，它可能会继续生成 "has revolutionized natural language processing."。对于下一个 token 预测器来说，这已经很令人印象深刻了。

现在试试这个：喂入 "What is the capital of France?"。基础模型不会回答 "Paris"。它会继续遵循模式。它可能会产生 "What is the capital of Germany? What is the capital of Spain?"，因为它从包含问题列表的文档中学到了这种模式。或者它可能会产生 "is a question that many people ask"，因为那是合理的下一个 token 续写。模型没有 *回答* 的概念。它只知道 *继续*。

这正是 GPT-3（基础模型，2020年6月发布）和 ChatGPT（指令微调，2022年11月发布）之间的差距。相同的架构。相同的预训练。区别在于那 2万到10万条精心设计的（指令，响应）对，教会了模型遵循对话模式。

Stanford Alpaca 证明了你不需百万级的示例。2023年3月，他们在仅 5.2万条由 GPT-3.5 生成的指令-响应对上微调了 Llama 7B。总成本：600美元。结果是一个能遵循指令、回答问题、进行对话的聊天机器人。不如 ChatGPT，但考虑到600美元和几小时的训练时间，已经令人惊讶地接近。

Meta 的 Llama 2 Chat 在其初始 SFT 阶段仅使用了约 2.7万条高质量示例。关键洞察：质量比数量更重要。2.7万条由专业标注员撰写的示例胜过从互联网上爬取的100万条嘈杂示例。

## 概念解析

### SFT 实际做了什么

监督微调延续了预训练中的相同训练循环——前向传播、计算损失、反向传播、更新权重——但使用的是不同类型的**数据**。不是原始文本，而是在结构化对话上进行训练：

```json
{
  "system": "You are a helpful assistant.",
  "user": "What is the capital of France?",
  "assistant": "The capital of France is Paris."
}
```

模型已经知道巴黎是法国首都。它在预训练期间从维基百科、教科书和网页中学到了这一点。SFT 不教模型新的事实。它教给模型一种新的*行为*：当看到问题时，产生回答。当看到指令时，产生补全。当看到有害请求时，产生拒绝。

可以这样理解。预训练给予模型知识。SFT 给予模型礼仪。

### 数据格式

业界主要采用三种格式。每种格式以不同的分隔符编码相同的信息——谁说了什么。

**Alpaca 格式**（斯坦福，2023年3月）：

```json
{
  "instruction": "Summarize the following article in 3 sentences.",
  "input": "The European Central Bank raised interest rates...",
  "output": "The ECB increased rates by 25 basis points..."
}
```

简单且广泛使用。`input` 字段是可选的——许多指令不需要额外上下文。斯坦福发布了 5.2万条这种格式的示例，由 GPT-3.5 生成，花费600美元。这开启了开源指令微调运动。

**ShareGPT 格式**（社区，2023年）：

```json
{
  "conversations": [
    {"from": "system", "value": "You are a helpful assistant."},
    {"from": "human", "value": "What causes tides?"},
    {"from": "gpt", "value": "Tides are caused by the gravitational pull of the Moon..."},
    {"from": "human", "value": "How often do they occur?"},
    {"from": "gpt", "value": "Most coastal areas experience two high tides and two low tides per day..."}
  ]
}
```

支持多轮对话。"from" 字段按惯例使用 "human" 和 "gpt"，与实际模型无关。Vicuna 是在从用户共享的 ChatGPT 对话中抓取的 7万条 ShareGPT 对话上训练的。

**ChatML 格式**（OpenAI，许多开源模型使用）：

```
