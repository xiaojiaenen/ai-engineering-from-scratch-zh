# BERT — 掩码语言建模

> GPT 预测下一个词。BERT 预测一个缺失的词。一句之差——却影响了此后五年的所有嵌入技术。

**类型：** 构建
**语言：** Python
**先修知识：** 阶段7 · 05（完整Transformer），阶段5 · 02（文本表示）
**时间：** 约45分钟

## 问题所在

2018年，每一个NLP任务——情感分析、命名实体识别、问答、蕴含关系——都是在自己的标注数据上从头训练一个模型。当时没有一个可供微调的、预训练好的“理解英语”检查点。ELMo (2018) 表明你可以用双向LSTM预训练上下文嵌入；这有所帮助，但未能普及。

BERT (Devlin et al. 2018) 提出了一个问题：如果我们取一个Transformer编码器，在互联网上每个句子上进行训练，并强制它从双向上下文预测缺失的单词，会怎样？然后你只需在下游任务上微调一个头。参数效率的提升令人惊叹。

结果：在18个月内，BERT及其变体（RoBERTa、ALBERT、ELECTRA）主导了当时存在的所有NLP排行榜。到2020年，地球上每一个搜索引擎、内容审核流程和语义搜索系统内部都嵌入了一个BERT。

到2026年，仅编码器模型仍然是分类、检索和结构化抽取的正确工具——它们的每token运行速度比解码器快5-10倍，其嵌入是每个现代检索栈的支柱。ModernBERT (2024年12月) 通过Flash Attention + RoPE + GeGLU将该架构的上下文长度推向8K。

## 核心概念

![掩码语言建模：选取token，遮盖它们，预测原始token](../assets/bert-mlm.svg)

### 训练信号

取一个句子：`the quick brown fox jumps over the lazy dog`。

随机遮盖15%的token：

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

训练模型预测被遮盖位置的原始token。由于编码器是双向的，在位置1预测`[MASK]`时可以利用位置2及以后的`brown fox jumps`。这正是GPT做不到的事情。

### BERT的掩码规则

在被选中用于预测的15%的token中：

- 80%被替换为`[MASK]`。
- 10%被替换为一个随机的token。
- 10%保持不变。

为什么不总是用`[MASK]`？因为`[MASK]`在推理时从未出现。如果训练模型在100%的遮盖位置都预期`[MASK]`，会在预训练和微调之间造成分布偏移。10%随机 + 10%不变的做法保持了模型的诚实性。

### 下一句预测 — 以及为何被弃用

最初的BERT还训练了NSP任务：给定两个句子A和B，预测B是否紧接在A之后。RoBERTa (2019) 通过消融实验证明NSP有害无益。现代编码器已跳过此任务。

### 2026年的变化：ModernBERT

2024年的ModernBERT论文使用2026年的原语重新构建了其模块：

| 组件       | 原始BERT (2018) | ModernBERT (2024) |
|------------|------------------|-------------------|
| 位置编码   | 可学习绝对位置   | RoPE              |
| 激活函数   | GELU             | GeGLU             |
| 归一化     | LayerNorm        | Pre-norm RMSNorm  |
| 注意力     | 全连接稠密       | 交替局部(128) + 全局 |
| 上下文长度 | 512              | 8192              |
| 分词器     | WordPiece        | BPE               |

并且不同于2018年的技术栈，它是原生支持Flash Attention的。在8K序列长度下，其推理速度比DeBERTa-v3快2-3倍，且GLUE分数更优。

### 2026年仍选择编码器的用例

| 任务                         | 编码器优于解码器的原因                                  |
|------------------------------|-------------------------------------------------------|
| 检索/语义搜索嵌入            | 双向上下文 = 每个token产生更优质的嵌入                  |
| 分类（情感、意图、毒性）     | 单次前向传播；无生成开销                               |
| 命名实体识别/token标注       | 逐位置输出，天然双向                                   |
| 零样本蕴含关系（自然语言推理）| 在编码器顶部添加分类头                                 |
| RAG重排序器                  | 交叉编码器评分，比LLM重排序器快10倍                    |

## 构建它

### 步骤1：掩码逻辑

参见`code/main.py`。函数`create_mlm_batch`接收一个token ID列表、词汇表大小和掩码概率。返回输入ID（已应用掩码）和标签（仅在掩码位置有值，其他位置为-100 —— PyTorch的忽略索引约定）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### 步骤2：在一个小型语料上运行掩码语言模型预测

在一个包含20个单词、200个句子的词汇表上，训练一个2层编码器 + 掩码语言模型头。不使用梯度——我们只做前向传播的完整性检查。完整训练需要PyTorch。

### 步骤3：比较掩码类型

展示三向规则如何在没有`[MASK]`的情况下保持模型可用性。在一个未遮盖的句子和一个遮盖的句子上进行预测。两者都应该产生合理的token分布，因为模型在训练中见过这两种模式。

### 步骤4：微调头

将掩码语言模型头替换为一个用于玩具情感数据集上的分类头。只有头部进行训练；编码器被冻结。这是每个BERT应用都遵循的模式。

## 使用它

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**嵌入模型是微调过的BERT。** `sentence-transformers`模型，如`all-MiniLM-L6-v2`，是使用对比损失训练的BERT。编码器是相同的。损失函数变了。

**交叉编码器重排序器也是微调过的BERT。** 在`[CLS] query [SEP] doc [SEP]`上进行配对分类。查询和文档之间的双向注意力正是交叉编码器相比双编码器质量更优的原因。

**2026年不选择BERT的场景。** 任何生成性任务。编码器没有合理的方式来自回归地生成token。此外：任何参数规模在1B以下的场景，一个小型解码器可以用更多灵活性匹配质量（例如Phi-3-Mini, Qwen2-1.5B）。

## 发布它

参见`outputs/skill-bert-finetuner.md`。该技能为一个新的分类或抽取任务规划BERT微调的范围（骨干网络选择、头部规格、数据、评估、停止策略）。

## 练习

1.  **简单。** 运行`code/main.py`并打印10,000个token上的掩码分布。确认约15%被选中，其中约80%变成了`[MASK]`。
2.  **中等。** 实现全词掩码：如果一个词被分词为多个子词，则要么全部遮盖，要么都不遮盖。在一个500句子的语料上测量这是否提高了掩码语言模型的准确率。
3.  **困难。** 在一个公开数据集的10,000个句子上训练一个小型（2层，d=64）BERT。为SST-2情感任务微调`[CLS]` token。在匹配的参数下与仅解码器基线进行比较——哪个更优？

## 关键术语

| 术语         | 人们通常怎么说 | 其实际含义                                                                 |
|--------------|----------------|--------------------------------------------------------------------------|
| MLM          | “掩码语言建模” | 训练信号：随机将15%的token替换为`[MASK]`，预测原始token。               |
| 双向         | “双向查看”     | 编码器注意力没有因果掩码——每个位置都能看到其他所有位置。                   |
| `[CLS]`  | “池化token”    | 添加到每个序列开头的一个特殊token；其最终嵌入被用作句子级表示。            |
| `[SEP]`  | “分段分隔符”   | 分隔成对序列（例如查询/文档，句子A/B）。                                   |
| NSP          | “下一句预测”   | BERT的第二个预训练任务；在RoBERTa中被证明无用，2019年后被弃用。            |
| 微调         | “适配到任务”   | 保持编码器基本冻结；在其顶部为下游任务训练一个小头部。                    |
| 交叉编码器   | “重排序器”     | 一个将查询和文档都作为输入，输出相关性分数的BERT。                        |
| ModernBERT   | “2024年刷新版” | 使用RoPE、RMSNorm、GeGLU、交替局部/全局注意力、8K上下文重建的编码器。     |

## 延伸阅读

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 原始论文。
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — 如何正确训练BERT；否定了NSP。
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — 在匹配计算量下，替换token检测优于MLM。
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — ModernBERT论文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — 权威的编码器参考实现。