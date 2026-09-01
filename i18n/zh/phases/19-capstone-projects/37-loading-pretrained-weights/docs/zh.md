# 加载预训练权重

> 从零训练一个 124M 参数的模型是一个预算决策；加载已发布的检查点只是又一个周二。本课从一个 safetensors 文件中将预训练的 GPT-2 风格权重加载到与第 35 课完全相同的架构中，逐步讲解参数名称映射，并生成一段续文以证明加载成功。无网络依赖，无第三方加载器，无黑魔法。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 19 课程 30 至 36
**时间：** 约 90 分钟

## 学习目标

- 使用 `safetensors` Python 库读取 safetensors 文件，并检查张量名称和形状。
- 将每个预训练参数名称映射到第 35 课 GPT 模型的对应参数上。
- 处理发布版 GPT-2 权重与本课程模型之间两种不同的命名约定：`wte/wpe/h.N.attn.c_attn/c_proj` 和 `mlp.c_fc/c_proj` 与本地命名 `tok_embed/pos_embed/blocks.N.attn.qkv/out_proj` 和 `mlp.fc1/fc2` 之间的差异。
- 在任何权重赋值发生之前，检测并拒绝形状不匹配的张量，并输出清晰的错误信息。
- 使用加载的权重生成一段简短续文，并确认生成的 token 来自已加载的分布，而非随机初始化的分布。

## 问题所在

已发布的权重并非为你的架构打包而成。它们使用的是原始实现中的名称。预训练文件中的 `transformer.h.0.attn.c_attn.weight` 形状为 `(2304, 768)`；而你的模型期望的是 `blocks.0.attn.qkv.weight`，形状也是 `(2304, 768)`（同一矩阵，不同布局约定），或者你的模型使用了 `nn.Linear`，其将矩阵以转置形式存储。同一个参数会呈现出三种略有不同的身份（名称、形状、字节布局），而加载器必须将这三者统一起来。

盲目复制的加载器会将正确的张量放到错误的位置，导致模型输出一堆乱码。而在形状不匹配时仅拒绝复制却什么也不记录的加载器，会让你无从判断是哪个张量加载失败。本课的加载器是显式的：每一次赋值都会记录日志，每一次形状都会检查，`LoadReport` 会汇总命中、遗漏和形状不匹配的情况，让你能清楚地看到发生了什么。

## 概念

```mermaid
flowchart LR
  SF[safetensors 文件<br/>gpt2-stub.safetensors] --> R[读取器<br/>safe_open]
  R --> N[参数名称迭代器]
  N --> M[名称映射器<br/>预训练 -> 本地]
  M --> S[形状检查]
  S -- 匹配 --> A[在 torch.no_grad() 下赋值张量]
  S -- 不匹配 --> E[记录不匹配<br/>不进行赋值]
  A --> RP[LoadReport]
  E --> RP
  RP --> G[generate<br/>验证样本]
```

名称映射器只是一个从字符串到字符串的函数。形状检查就是一行 if。赋值发生在 `torch.no_grad()` 内部，因此 autograd 不会追踪加载过程。报告记录每个名称的处理结果。

### GPT-2 命名约定

已发布的 GPT-2 权重使用如下名称：

| 预训练名称 | 形状 | 含义 |
|-----------------|-------|---------|
| `wte.weight` | (50257, 768) | 词元嵌入 |
| `wpe.weight` | (1024, 768) | 位置嵌入 |
| `h.N.ln_1.weight` | (768,) | 第 N 层的 LayerNorm 1 缩放系数 |
| `h.N.ln_1.bias` | (768,) | 第 N 层的 LayerNorm 1 偏移量 |
| `h.N.attn.c_attn.weight` | (768, 2304) | 融合 QKV 线性层权重 |
| `h.N.attn.c_attn.bias` | (2304,) | 融合 QKV 线性层偏置 |
| `h.N.attn.c_proj.weight` | (768, 768) | 注意力输出投影权重 |
| `h.N.attn.c_proj.bias` | (768,) | 注意力输出投影偏置 |
| `h.N.ln_2.weight` | (768,) | LayerNorm 2 缩放系数 |
| `h.N.ln_2.bias` | (768,) | LayerNorm 2 偏移量 |
| `h.N.mlp.c_fc.weight` | (768, 3072) | MLP fc1 权重 |
| `h.N.mlp.c_fc.bias` | (3072,) | MLP fc1 偏置 |
| `h.N.mlp.c_proj.weight` | (3072, 768) | MLP fc2 权重 |
| `h.N.mlp.c_proj.bias` | (768,) | MLP fc2 偏置 |
| `ln_f.weight` | (768,) | 最终 LayerNorm 缩放系数 |
| `ln_f.bias` | (768,) | 最终 LayerNorm 偏移量 |

有两个需要提前注意的要点。`c_attn`、`c_proj`、`c_fc` 这些线性层的权重，相对于 `nn.Linear.weight` 所期望的形式是转置存储的。加载器在赋值时会进行转置。LM head 根本不在文件中；模型依赖通过 `wte` 进行权重共享，因此在 `wte` 加载完成后，通过别名方式设置 head。

### 本地命名约定

本课程中的模型使用描述性名称：

| 本地名称 | 含义 |
|------------|---------|
| `tok_embed.weight` | 词元嵌入 |
| `pos_embed.weight` | 位置嵌入 |
| `blocks.N.ln1.scale` | 第 N 层的 LayerNorm 1 缩放系数 |
| `blocks.N.ln1.shift` | 第 N 层的 LayerNorm 1 偏移量 |
| `blocks.N.attn.qkv.weight` | 融合 QKV |
| `blocks.N.attn.qkv.bias` | 融合 QKV 偏置 |
| `blocks.N.attn.out_proj.weight` | 注意力输出投影权重 |
| `blocks.N.attn.out_proj.bias` | 输出投影偏置 |
| `blocks.N.ln2.scale` | LayerNorm 2 缩放系数 |
| `blocks.N.ln2.shift` | LayerNorm 2 偏移量 |
| `blocks.N.mlp.fc1.weight` | MLP fc1 |
| `blocks.N.mlp.fc1.bias` | MLP fc1 偏置 |
| `blocks.N.mlp.fc2.weight` | MLP fc2 |
| `blocks.N.mlp.fc2.bias` | MLP fc2 偏置 |
| `final_ln.scale` | 最终 LayerNorm 缩放系数 |
| `final_ln.shift` | 最终 LayerNorm 偏移量 |

映射关系是一个固定函数。本课将其作为字典提供，加载器会遍历该字典。

### 桩测试文件

真实的 GPT-2 权重文件为 0.5 GB。本演示不下载它们，而是在首次运行时生成一个小型 safetensors 测试文件，采用与 GPT-2 完全一致的命名约定，形状适配一个 d_model 为 192（而非 768）的 12 层模型。该测试文件具有正确的结构，可以覆盖加载器中的所有代码路径。将测试文件替换为真实文件后，加载器无需任何修改即可工作。

```figure
cc-weight-remap
```

## 构建代码

`code/main.py` 实现了：

- 第 35 课 `GPTModel` 的一个小型副本，使本课代码完全自包含。
- `make_pretrained_to_local(num_layers)`，用于展开每层的条目。
- `load_safetensors(model, path)`，用于遍历名称、进行映射、检查形状、转置 conv1d 风格的权重，并在 `torch.no_grad()` 下完成赋值。返回一个 `LoadReport`。
- `make_stub_safetensors(path, cfg)`，用于生成一个采用完整预训练命名约定的测试文件。
- 一个演示流程：首次运行时创建 `outputs/gpt2-stub.safetensors`，构建一个全新模型，从随机初始化捕获一段生成续文，加载测试文件后再次捕获续文，打印两次结果，并验证两者不同（证明加载确实改变了模型）。

运行方式：

```bash
python3 code/main.py
```

输出包括：测试文件路径、逐条加载日志、`LoadReport` 摘要、加载前的续文、加载后的续文，以及一个故意注入的坏张量产生的形状不匹配错误，以覆盖失败路径。

## 技术栈

- `safetensors`：用于磁盘格式和流式读取器。
- `torch`：用于模型和赋值计算。
- 不依赖 `transformers`、`huggingface_hub`，不进行任何网络调用。

## 生产环境中的实践模式

有三种模式让加载器能够处理他人提供的权重。

**在任何赋值之前先验证文件。** 打开文件，列出每个张量的名称、dtype 和形状，运行完整的映射和形状检查，只有在全部通过后开始赋值。半加载的模型是静默失败的源头。

**为每次赋值记录源名称和目标名称。** 当结果看起来不对时，日志会告诉你哪个张量落到了哪里；否则只能去读十六进制转储。本课中的 `LoadReport` 数据类会跟踪 `loaded`、`missing`、`unexpected` 和 `shape_mismatch` 四个列表，并在末尾打印摘要。

**LM head 是权重共享别名，不是独立副本。** 在加载 `tok_embed` 之后执行 `model.lm_head.weight = model.tok_embed.weight` 是标准做法。将嵌入矩阵复制到独立的 `lm_head.weight` 参数会破坏权重共享，并悄悄使参数数量翻倍。

## 使用方式

- 加载器适用于任何采用预训练命名约定的 safetensors 文件。真实 GPT-2 文件（small / medium / large / xl）无需修改代码即可使用；仅需调整模型配置。
- 同样的模式可推广到 LLaMA、Mistral、Qwen 权重，只需更新名称映射表。形状检查和报告机制保持不变。
- 加载后进行验证生成是一个快速的检查手段：如果加载后的样本看起来与加载前一样，说明加载没有改变模型，这意味着映射表可能完全漏掉了所有张量。

## 练习

1. 为加载器增加一个 `dtype` 参数，在赋值时将每个张量转换为目标 dtype（`bfloat16`、`float16`、`float32`）。确认一个 `float32` 模型可以向下转换为 `bfloat16` 并仍能正常生成。
2. 增加一个 `expected_layers` 参数，拒绝加载 `h.N` 索引与模型 `num_layers` 不匹配的检查点。
3. 将加载器接入第 35 课的生成函数，并排输出两个样本：一个来自随机初始化，一个来自加载的测试文件。
4. 增加一个导出路径：将当前模型状态以预训练命名约定写入新的 safetensors 文件。对加载器做往返测试，并确认报告中的形状不匹配数为零。
5. 扩展 `NAME_MAP` 以支持 LLaMA 命名约定（无偏置项、RMSNorm、融合 qkv 布局），并对你生成的 LLaMA 测试桩文件重新运行加载器。

## 关键术语

| 术语 | 人们常说的说法 | 实际含义 |
|------|----------------|------------------------|
| 名称映射表 | "Key remapping" | 从预训练张量名称到本地参数名称的函数；通常是一个逐层展开的字典字面量 |
| 形状不匹配 | "Bad shape" | 预训练张量经映射后存在于目标名称下，但其维度与本地参数不符；加载器拒绝赋值并记录这对名称 |
| 加载时转置 | "Conv1d layout" | 已发布的 GPT-2 将注意力 MLP 投影以 nn.Linear 期望形式的转置方式存储；加载器在赋值时进行转置 |
| 权重共享别名 | "Shared LM head" | 执行 model.lm_head.weight = model.tok_embed.weight，使 head 与嵌入共享同一块存储；head 不在文件中正是因为这一机制 |
| 加载报告 | "Coverage summary" | 一个小数据类，用于跟踪 loaded、missing、unexpected 和 shape_mismatch 四个列表；打印它即可判断加载是否成功 |

## 延伸阅读

- Phase 19 课程 35：本课程的架构定义，即接收权重的模型。
- Phase 19 课程 36：产生与本课程相同形状的 Checkpoint 的训练循环。
- Phase 10 课程 11（量化）：当显存紧张时如何处理已加载的权重。
- Phase 10 课程 13（构建完整的 LLM 流水线）：加载与推理的完整生命周期。
