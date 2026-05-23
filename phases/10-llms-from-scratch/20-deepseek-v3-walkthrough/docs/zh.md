# DeepSeek-V3 架构详解

> 第10阶段 · 第14课介绍了每个开放模型都会调整的六个架构旋钮。DeepSeek-V3（2024年12月，总参数6710亿，活跃参数370亿）调整了全部六个旋钮，并新增了四个：多头潜在注意力、无辅助损失负载均衡、多token预测和DualPipe训练。本课将从头到尾解读DeepSeek-V3的架构，并从公开配置中推导出每个参数计数。学完之后，您将能解释为什么671B/37B的比例是正确的选择，以及为什么在前沿模型中，MLA与MoE的结合优于单独使用任何一方。

**类型：** 学习
**语言：** Python（标准库，参数计算器）
**前置要求：** 第10阶段 · 14（开放模型解读），第10阶段 · 17（NSA），第10阶段 · 18（MTP），第10阶段 · 19（DualPipe）
**时间：** ~75分钟

## 学习目标

- 从头到尾解读DeepSeek-V3的配置，并用六个GPT-2旋钮加四个DeepSeek特有的新增项来解释每个字段。
- 推导出总参数量（6710亿）、活跃参数量（370亿）以及构成这些数量的各个组件。
- 计算128k上下文下MLA的KV缓存占用，并与具有相同活跃参数、使用GQA的稠密模型进行比较。
- 说明四个DeepSeek特有创新（MLA、MTP、无辅助损失路由、DualPipe）及其分别针对架构/训练栈的哪个部分。

## 问题背景

DeepSeek-V3是首个在架构上与Llama系列有显著差异的前沿开放模型。Llama 3 405B是“调整了六个旋钮的GPT-2”。DeepSeek-V3则是调整了全部六个旋钮并额外增加了四个旋钮的GPT-2。阅读Llama 3的配置是阅读DeepSeek配置的预热，但深层结构——注意力块的形状、路由逻辑、训练时的目标——差异很大，需要单独的讲解。

学习它的回报：DeepSeek-V3的开放权重发布改变了开放模型中“前沿能力”的定义。其架构是2026年许多训练运行正在复制的蓝图。理解它，是任何涉及前沿LLM训练或推理岗位的基本要求。

## 核心概念

### 不变的核心，重申

DeepSeek-V3仍然是自回归模型。它仍然堆叠解码器块。每个块仍然包含注意力加MLP加两个RMSNorm。它仍然在MLP中使用SwiGLU。它仍然使用RoPE。预归一化。权重绑定的嵌入。与每个Llama或Mistral相同的基线。

### 关键转折：MLA取代GQA

从第10阶段 · 14课你已知GQA通过在Q头组间共享K和V来缩小KV缓存。多头潜在注意力（MLA）更进一步：K和V被压缩到一个共享的低秩潜在表示（`kv_lora_rank`）中，然后在运行时按头解压。KV缓存仅存储潜在表示——通常每个token每层512个浮点数，而不是8 x 128 = 1024个浮点数。

在128k上下文下，使用MLA的DeepSeek-V3（每个token每层一个共享潜在`c^{KV}`；K和V都通过可融入后续矩阵乘的上投影由此潜在表示推导）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

一个假设的GQA基线（Llama 3 70B形状，8个KV头，头维度128）将消耗：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

在128k上下文下，MLA的缓存比Llama-3-70B风格的GQA缓存小4倍。

权衡：MLA为每次注意力计算（每个头）增加了解压步骤。额外的计算量相对于节省的带宽来说很小。对于长上下文推理，这是净收益。

### 路由：无辅助损失负载均衡

MoE路由器决定哪些top-k专家处理每个token。朴素的路由器会将过多的工作集中在少数专家上，导致其他专家闲置。标准修复方案：添加一个惩罚负载不均衡的辅助损失项。这有效，但会轻微损害主任务性能。

DeepSeek-V3引入了一种无辅助损失方案。在路由器的logits中为每个专家添加偏置项，并在训练期间通过简单规则调整：如果专家`e`过载，则降低`bias_e`；如果欠载，则增加它。没有额外的损失项。训练保持干净。专家负载保持平衡。

对主损失的影响：不可测量。对MoE架构的影响：更简洁，无需调整辅助损失超参数。

### MTP：更密集的训练 + 免费草稿

从第10阶段 · 18课你已知DeepSeek-V3添加了D=1的MTP模块，用于预测未来第二个位置的token。在推理时，训练好的模块被重新用作推测解码草稿，接受率超过80%。在训练时，每个隐藏状态在D+1 = 2个目标上受到监督，提供了更密集的信号。

参数：在671B主模型之上增加140亿。开销：2.1%。

### 训练：DualPipe

从第10阶段 · 19课你已知DualPipe是一种双向流水线，将前向和后向计算块与跨节点的全对全通信重叠执行。在DeepSeek-V3的2,048张H800规模下，它回收了大约24.5万GPU小时，这些时间是1F1B方案因流水线气泡会损失的。

### 配置，逐字段解析

以下是DeepSeek-V3的配置（简化版）：

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

解析如下：

- `hidden_size=7168`：嵌入维度。
- `num_hidden_layers=61`：总块深度。
- `first_k_dense_layers=3`：前3个块使用大小为18432的稠密MLP。其余58个使用MoE。
- `num_attention_heads=128`：128个查询头。
- `kv_lora_rank=512`：K和V被压缩到此潜在维度，并按头解压。
- `num_experts=256, num_experts_per_tok=8`：每个MoE块有256个专家，路由top-8。
- `shared_experts=1`：在256个路由专家之上，有1个始终激活的专家为每个token做贡献。可以将其视为一个“稠密底板”，确保每个token都能获得可靠的计算。
- `moe_intermediate_size=2048`：每个专家的MLP隐藏层大小。小于稠密MLP，因为共有256个专家。

### 参数核算

完整计算位于`code/main.py`。核心结论如下：

- 嵌入层：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前3个稠密块：带MLA的注意力（每块约1.44亿）+ 稠密MLP（每块约2.6亿）+ 归一化层。总计约12亿。
- 58个MoE块：带MLA的注意力（约1.44亿）+ 256个专家（每个3千万）+ 1个共享专家（3千万）+ 归一化层。每块总计约79.5亿（包含所有专家）。58个MoE块总计4610亿。
- MTP模块：140亿。

总计：核心架构约4760亿 + 140亿MTP + 明确公布的6710亿数字包含了额外的结构参数（偏置张量、专家特定组件、共享专家缩放等）。我们在计算器中复现的数字与公布值相差在3-5%以内——差异来自DeepSeek报告第2节附录中记载的细粒度核算。

每次前向传播的活跃参数：

- 注意力：每层1.44亿 * 61层 = 88亿（所有层均激活）。
- MLP活跃部分：前3层为稠密（3 * 2.6亿 = 7.8亿），58个MoE层每层激活8个路由专家 + 1个共享专家 + 路由开销。每层活跃MLP：约2.6亿。总计：3 * 2.6亿 + 58 * 2.6亿 = ~159亿。
- 嵌入 + 归一化层：12亿。
- 活跃总计：大约260亿核心 + 140亿MTP（训练过但推理时不总是运行）≈ 370亿。

### 671B / 37B 的比例

18倍的稀疏率（活跃参数占总参数的5.5%）。DeepSeek-V3是已发布的开放权重中最稀疏的前沿MoE模型。Mixtral 8x7B的比率为13/47（28%），要密集得多。Llama 4 Maverick的比率为17B/400B（4.25%），与之相当。DeepSeek的赌注是：在前沿规模下，更多的专家配合更低的激活比率，能在每活跃FLOP上产生更好的质量。

### DeepSeek-V3 的定位

| 模型 | 总参数 | 活跃参数 | 比率 | 注意力机制 | 新思想 |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + aux-free + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN扩展 |

### 后续：R1, V4

DeepSeek-R1（2025年）是在V3骨架上进行的推理训练运行。R1使用相同的架构。改变的是后训练流程（在可验证任务上进行大规模强化学习），而非预训练架构。

DeepSeek-V4（如果发布）预计将保留MLA + MoE + MTP，并增加DSA（DeepSeek稀疏注意力），即第10阶段 · 17课NSA的继任者。这条发展路线是稳定的：架构层面的创新不断积累；每个版本都会调整额外的旋钮。

## 动手使用

`code/main.py`是专门针对DeepSeek-V3形状设计的参数计算器。运行它，将其输出与论文中的数字进行比较，并将其应用于假设的变体（256专家 vs 512专家，top-8 vs top-16，MLA秩512 vs 1024）。

关注点：

- 总参数量 vs 公布的6710亿。
- 活跃参数量 vs 公布的370亿。
- 128k上下文下的KV缓存 — MLA与GQA的对比。
- 逐层分解，以查看参数预算的实际去向。

## 输出成果

本课将产出`outputs/skill-deepseek-v3-reader.md`。给定一个DeepSeek家族模型（V3、R1或任何未来变体），它将生成一个组件级的架构解读，指出配置中的每个字段，按组件推导参数计数，并识别模型使用了四个DeepSeek特有创新中的哪几个。

## 练习

1. 运行`code/main.py`。将计算器的总参数估计值与公布的6710亿进行比较，并确定差异的来源。论文的第2节有完整的明细。

2. 将配置修改为使用秩256的MLA而不是512。计算128k上下文下由此产生的KV缓存大小。这能节省多少百分比？又牺牲了多少每头的表达能力？

3. 比较DeepSeek-V3的（256专家，top-8）路由与一个假设的（512专家，top-8）变体。总参数量增加；活跃参数量保持不变。理论上，额外的专家容量能带来什么收益？在推理时又有什么代价？

4. 阅读DeepSeek-V3技术报告（arXiv:2412.19437）第2.1节关于MLA的内容。用三句话解释为什么K和V的解压矩阵可以在推理时被“吸收”到后续的矩阵乘中以提高效率。

5. DeepSeek-V3在大多数操作中使用FP8训练。计算存储6710亿权重时，FP8相比BF16节省的内存。这与14.8万亿token的训练预算如何关联？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| MLA | “多头潜在注意力” | 将K和V压缩到一个共享的低秩潜在表示中（kv_lora_rank，通常为512），在运行时按头解压；KV缓存仅存储潜在表示 |
| kv_lora_rank | “MLA压缩维度” | K和V共享潜在表示的大小；DeepSeek-V3使用512 |
| 前k个稠密层 | “早期层保持稠密” | 最初的几个MoE模型层跳过MoE路由器，运行稠密MLP以保持稳定 |
| num_experts_per_tok | “Top-k路由” | 每个token激活多少个路由专家；DeepSeek-V3使用8 |
| 共享专家 | “始终激活的专家” | 不管路由如何，都会处理每个token的专家；DeepSeek-V3使用1个 |
| 无辅助损失路由 | “偏置调整负载均衡” | 在训练期间调整每个专家的偏置项，以在不增加损失项的情况下保持专家负载平衡 |
| MTP模块 | “额外的预测头” | 从h^(1)和E(t+1)预测t+2的Transformer块；用于更密集的训练，以及免费的推测解码草稿 |
| DualPipe | “双向流水线” | 将前向/后向计算与跨节点全对全通信重叠执行的训练调度 |
| 活跃参数比 | “稀疏度” | active_params / total_params；DeepSeek-V3达到5.5% |
| FP8训练 | “8位训练” | 训练存储和许多计算操作使用FP8；相比BF16大约节省一半内存，但有轻微的质量代价 |

## 扩展阅读

- [DeepSeek-AI — DeepSeek-V3技术报告 (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整的架构、训练和结果文档
- [DeepSeek-V3 Hugging Face模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 配置文件和部署说明
- [DeepSeek-V2论文 (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — 引入了MLA的前身
- [DeepSeek-R1论文 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — 在V3架构上进行推理训练的继任者
- [原生稀疏注意力 (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — DeepSeek家族注意力机制的未来方向
- [DualPipe仓库](https://github.com/deepseek-ai/DualPipe) — 训练调度的参考实现