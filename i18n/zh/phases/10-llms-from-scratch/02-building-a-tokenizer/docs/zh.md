# 从零构建分词器

> 第 01 课给了你一个玩具。这节课给你一把武器。

**类型：** 动手实践
**语言：** Python
**前置知识：** 第 10 阶段，第 01 课（分词器：BPE、WordPiece、SentencePiece）
**预计时长：** 约 90 分钟

## 学习目标

- 构建一个生产级的 BPE 分词器，处理 Unicode、空白字符标准化和特殊标记
- 实现基于字节的回退机制，使分词器能编码任何输入（包括 emoji、中日韩文字和代码）而不产生未知标记
- 添加预处理正则模式，在应用 BPE 合并之前先在词边界处分割文本
- 在语料库上训练自定义分词器，并在多语言文本上与 tiktoken 比较其压缩比

## 问题所在

你从第 01 课构建的 BPE 分词器只能处理英文文本。现在试试对它输入日文。或者 emoji。或者混合了制表符和空格的 Python 代码。

它会崩溃。

不是因为 BPE 算法错了——是因为实现不完整。一个生产级分词器需要处理任何编码的原始字节、在分割前进行 Unicode 标准化、管理永不参与合并的特殊标记、将预处理与子词分割链式组合，并且所有这些操作必须足够快，不会成为处理 15 万亿个 token 的训练管道的瓶颈。

GPT-2 的分词器有 50,257 个 token。Llama 3 有 128,256 个。GPT-4 大约 100,000 个。这些数字可不是玩具级别。支撑这些词表的合并表是在数百 GB 文本上训练出来的，而周边的配套机制——标准化、预处理、特殊标记注入、对话模板格式化——才是区分"能处理 hello world"和"能处理整个互联网"的分词器的关键。

你要构建的正是这些配套机制。

## 核心概念

### 完整流水线

一个生产级分词器不是一个算法。它是五个阶段的流水线，每个阶段解决不同的问题。

```mermaid
graph LR
    A[原始文本] --> B[标准化]
    B --> C[预处理分割]
    C --> D[BPE 合并]
    D --> E[特殊标记]
    E --> F[Token ID]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个阶段都有特定的职责：

| 阶段 | 做什么 | 为什么重要 |
|-------|-------------|----------------|
| 标准化 | NFKC Unicode、可选小写、可选去重音符 | "fi" 连字（U+FB01）变为 "fi"（两个字符）。不做这个，同一个词会得到不同的 token。 |
| 预处理分割 | 在 BPE 之前将文本分割为片段 | 防止 BPE 跨词边界合并。"the cat" 绝不应产生 token "e c"。 |
| BPE 合并 | 对字节序列应用学到的合并规则 | 核心压缩。将原始字节转为子词 token。 |
| 特殊标记 | 注入 `[BOS]`、`[EOS]`、`[PAD]`、对话模板标记 | 这些 token 有固定 ID。它们从不参与 BPE 合并。模型需要它们来理解结构。 |
| ID 映射 | 将 token 字符串转为整数 ID | 模型看到的是整数，不是字符串。 |

### 基于字节的 BPE

第 01 课的分词器操作于 UTF-8 字节。这是正确的选择。但我们忽略了一个重要问题：当这些字节不是有效 UTF-8 时会发生什么？

基于字节的 BPE 通过将所有可能的字节值（0-255）都视为合法 token 来解决这个问题。你的基础词表恰好有 256 个条目。任何文件——文本、二进制、损坏的——都可以被分词化而不会产生未知 token。

GPT-2 添加了一个技巧：将每个字节映射到一个可打印的 Unicode 字符，使词表保持人类可读。字节 0x20（空格）在其映射中变为字符 "G"。这纯粹是外观上的处理。算法本身并不在乎。

真正的力量在于：基于字节的 BPE 可以处理地球上每种语言。中文字符每个是 3 个 UTF-8 字节。日文可能是 3-4 个字节。阿拉伯文、天城文、emoji——所有这些都只是字节序列。BPE 算法在这些字节序列中寻找模式的方式，与它在英文 ASCII 字节中寻找模式的方式完全相同。

### 预处理分割

在 BPE 接触你的文本之前，你需要先将它分割成片段。这能防止合并算法创建跨越词边界的 token。

GPT-2 使用一个正则模式来分割文本：

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

这个模式在缩约形式（"don't" 变为 "don" + "'t"）、带可选前导空格的单词、数字、标点和空白处分割。前导空格保留附加在单词上——所以 "the cat" 变为 [" the", " cat"]，而不是 ["the", " ", "cat"]。

Llama 使用 SentencePiece，完全跳过正则。它将原始字节流视为一个超长序列，让 BPE 算法自己找出边界。这更简单，但给 BPE 更多自由去创建跨词的 token。

这个选择很重要。GPT-2 的正则防止分词器学到"一个词尾的 the"和"另一个词首的 the"应该合并。SentencePiece 允许这样做，有时会产生更高效的压缩，但 token 的可解释性较差。

### 特殊标记

每个生产级分词器都为结构性标记保留 token ID：

| Token | 用途 | 被谁使用 |
|-------|---------|---------|
| `[BOS]` / `<s>` | 序列开头 | Llama 3、GPT |
| `[EOS]` / `</s>` | 序列结尾 | 所有模型 |
| `[PAD]` | 用于批次对齐的填充 | BERT、T5 |
| `[UNK]` | 未知 token（基于字节的 BPE 消除了这个） | BERT、WordPiece |
| `<\|im_start\|>` | 对话消息边界开始 | ChatGPT、Qwen |
| `<\|im_end\|>` | 对话消息边界结束 | ChatGPT、Qwen |
| `<\|user\|>` | 用户回合标记 | Llama 3 |
| `<\|assistant\|>` | 助手回合标记 | Llama 3 |

特殊标记从不被 BPE 分割。它们在合并算法运行前精确匹配，替换为固定 ID，周围文本正常分词。

### 对话模板

这是大多数人感到困惑、大多数实现出错的地方。

当你向对话模型发送消息时，API 接受一个消息列表：

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型看到的是 JSON。它看到的是平铺的 token 序列。对话模板使用特殊标记将消息转换为该平铺序列。每个模型的实现方式都不同：

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
ystem
You are helpful.
