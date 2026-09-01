# Tokenizers: BPE, WordPiece, SentencePiece

> Your LLM does not read English. It reads integers. The tokenizer decides whether those integers carry meaning or waste it.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 05 (NLP Foundations)
**Time:** ~90 minutes

## Learning Objectives

- Implement BPE, WordPiece, and Unigram tokenization algorithms from scratch and compare their merge strategies
- Explain how vocabulary size affects model efficiency: too small creates long sequences, too large wastes embedding parameters
- Analyze tokenization artifacts across languages and code, identifying where specific tokenizers break down
- Use the tiktoken and sentencepiece libraries to tokenize text and inspect the resulting token IDs

## The Problem

Your LLM does not read English. It does not read any language. It reads numbers.

The gap between "Hello, world!" and [15496, 11, 995, 0] is the tokenizer. Every word, every space, every punctuation mark must be converted into an integer before a model can process it. This conversion is not neutral. It bakes assumptions into the model that cannot be undone later.

Get this wrong and your model wastes capacity encoding common words with multiple tokens. "unfortunately" becomes four tokens instead of one. Your 128K context window just shrank by 75% for text heavy in multi-syllable words. Get it right and the same context window holds twice as much meaning. The difference between "this model handles code well" and "this model chokes on Python" often comes down to how the tokenizer was trained.

Every API call you make to GPT-4 or Claude is priced per token. Every token your model generates costs compute. The fewer tokens required to represent an output, the faster the end-to-end inference. Tokenization is not preprocessing. It is architecture.

## The Concept

### Three Approaches That Failed (and One That Won)

There are three obvious ways to convert text to numbers. Two of them do not work at scale.

**Word-level tokenization** splits on spaces and punctuation. "The cat sat" becomes ["The", "cat", "sat"]. Simple. But what about "tokenization"? Or "GPT-4o"? Or a German compound word like "Geschwindigkeitsbegrenzung"? Word-level requires a massive vocabulary to cover every word in every language. Miss a word and you get the dreaded `[UNK]` token -- the model's way of saying "I have no idea what this is." English alone has over a million word forms. Add code, URLs, scientific notation, and 100 other languages and you need an infinite vocabulary.

**Character-level tokenization** goes the other direction. "hello" becomes ["h", "e", "l", "l", "o"]. Vocabulary is tiny (a few hundred characters). No unknown tokens ever. But sequences become extremely long. A sentence that would be 10 word-level tokens becomes 50 character-level tokens. The model must learn that "t", "h", "e" together mean "the" -- burning attention capacity on something a human learns at age three.

**Subword tokenization** finds the sweet spot. Common words stay whole: "the" is one token. Rare words decompose into meaningful pieces: "unhappiness" becomes ["un", "happi", "ness"]. Vocabulary stays manageable (30K to 128K tokens). Sequences stay short. Unknown tokens essentially disappear because any word can be built from subword pieces.

Every modern LLM uses subword tokenization. GPT-2, GPT-4, BERT, Llama 3, Claude -- all of them. The question is which algorithm.

```mermaid
graph TD
    A["文本: 'unhappiness'"] --> B{"分词策略"}
    B -->|词级| C["['unhappiness']\n词表中有则 1 个 token\n否则 [UNK]"]
    B -->|字符级| D["['u','n','h','a','p','p','i','n','e','s','s']\n11 个 token"]
    B -->|子词 BPE| E["['un','happi','ness']\n3 个 token"]

    style C fill:#ff6b6b,color:#fff
    style D fill:#ffa500,color:#fff
    style E fill:#51cf66,color:#fff
```

### BPE: Byte Pair Encoding

BPE is a greedy compression algorithm repurposed for tokenization. The idea is simple enough to fit on an index card.

Start with individual characters. Count every adjacent pair in the training corpus. Merge the most frequent pair into a new token. Repeat until you reach your target vocabulary size.

```figure tokenizer-bpe
```

Here is BPE running on a tiny corpus with the words "lower", "lowest", and "newest":

```
语料（含词频）：
  "lower"  x5
  "lowest" x2
  "newest" x6

第 0 步 —— 以字符开始：
  l o w e r       (x5)
  l o w e s t     (x2)
  n e w e s t     (x6)

第 1 步 —— 统计相邻词对：
  (e,s): 8    (s,t): 8    (l,o): 7    (o,w): 7
  (w,e): 13   (e,r): 5    (n,e): 6    ...

第 2 步 —— 合并最频繁的词对 (w,e) -> "we"：
  l o we r        (x5)
  l o we s t      (x2)
  n e we s t      (x6)

第 3 步 —— 重新统计并合并 (e,s) -> "es"：
  l o we r        (x5)
  l o we s t      (x2)    <- 'es' 只能由 'e'+'s' 形成，而非 'we'+'s'
  n e we s t      (x6)    <- 等等，'we' 前面的 'e' 和后面的 's'

实际上精确跟踪如下：
  "we" 合并后，剩余的词对：
  (l,o): 7   (o,we): 7   (we,r): 5   (we,s): 8
  (s,t): 8   (n,e): 6    (e,we): 6

第 3 步 —— 合并 (we,s) -> "wes" 或 (s,t) -> "st"（并列第一为 8，取先出现的）：
  合并 (we,s) -> "wes"：
  l o we r        (x5)
  l o wes t       (x2)
  n e wes t       (x6)

第 4 步 —— 合并 (wes,t) -> "west"：
  l o we r        (x5)
  l o west        (x2)
  n e west        (x6)

...继续直到达到目标词表大小。
```

The merge table is the tokenizer. To encode new text, apply merges in the order they were learned. The training corpus determines which merges exist, and that choice permanently shapes what the model sees.

```mermaid
graph LR
    subgraph Training["BPE 训练循环"]
        direction TB
        T1["起始：字符词表"] --> T2["统计所有相邻词对"]
        T2 --> T3["合并最频繁的词对"]
        T3 --> T4["将合并后的 token 加入词表"]
        T4 --> T5{"达到目标\n词表大小？"}
        T5 -->|否| T2
        T5 -->|是| T6["完成：保存合并表"]
    end
```

### Byte-Level BPE (GPT-2, GPT-3, GPT-4)

Standard BPE operates on Unicode characters. Byte-level BPE operates on raw bytes (0-255). This gives you a base vocabulary of exactly 256, handles any language or encoding, and never produces an unknown token.

GPT-2 introduced this approach. The base vocabulary covers every possible byte. BPE merges build on top of that. OpenAI's tiktoken library implements byte-level BPE with these vocabulary sizes:

- GPT-2: 50,257 tokens
- GPT-3.5/GPT-4: ~100,256 tokens (cl100k_base encoding)
- GPT-4o: 200,019 tokens (o200k_base encoding)

### WordPiece (BERT)

WordPiece looks similar to BPE but picks merges differently. Instead of raw frequency, it maximizes the likelihood of the training data:

```
BPE merge criterion:      count(A, B)
WordPiece merge criterion: count(AB) / (count(A) * count(B))
```

BPE asks: "Which pair appears most often?" WordPiece asks: "Which pair appears together more often than you would expect by chance?" This subtle difference produces different vocabularies. WordPiece favors merges where co-occurrence is surprising, not just frequent.

WordPiece also uses a "##" prefix for continuation subwords:

```
"unhappiness" -> ["un", "##happi", "##ness"]
"embedding"   -> ["em", "##bed", "##ding"]
```

The "##" prefix tells you this piece continues a previous token. BERT uses WordPiece with a vocabulary of 30,522 tokens. Every BERT variant -- DistilBERT, RoBERTa's tokenizer is actually BPE, but BERT itself is WordPiece.

### SentencePiece (Llama, T5)

SentencePiece treats the input as a raw stream of Unicode characters, including whitespace. No pre-tokenization step. No language-specific rules about word boundaries. This makes it genuinely language-agnostic -- it works on Chinese, Japanese, Thai, and other languages where spaces do not separate words.

SentencePiece supports two algorithms:
- **BPE mode**: same merge logic as standard BPE, applied to raw character sequences
- **Unigram mode**: starts with a large vocabulary and iteratively removes tokens that least affect the overall likelihood. The reverse of BPE -- prune instead of merge.

Llama 2 uses SentencePiece BPE with a vocabulary of 32,000 tokens. T5 uses SentencePiece Unigram with 32,000 tokens. Note: Llama 3 switched to a tiktoken-based byte-level BPE tokenizer with 128,256 tokens.

### Vocabulary Size Tradeoffs

This is a real engineering decision with measurable consequences.

```mermaid
graph LR
    subgraph Small["小词表 (32K)\n例如：BERT、T5"]
        S1["每段文本 token 更多"]
        S2["序列更长"]
        S3["Embedding 矩阵更小"]
        S4["稀有词处理更好"]
    end
    subgraph Large["大词表 (128K+)\n例如：Llama 3、GPT-4o"]
        L1["每段文本 token 更少"]
        L2["序列更短"]
        L3["Embedding 矩阵更大"]
        L4["推理更快"]
    end
```

Concrete numbers. For a 128K vocabulary with 4,096-dimensional embeddings, the embedding matrix alone is 128,000 x 4,096 = 524 million parameters. For a 32K vocabulary, it is 131 million parameters. That is a 400M parameter difference from the tokenizer choice alone.

But larger vocabularies compress text more aggressively. The same English paragraph that takes 100 tokens with a 32K vocabulary might take 70 tokens with a 128K vocabulary. That means 30% fewer forward passes during generation. For a model serving millions of requests, that is a direct reduction in compute cost.

The trend is clear: vocabulary sizes are growing. GPT-2 used 50,257. GPT-4 uses ~100K. Llama 3 uses 128K. GPT-4o uses 200K.

| Model | Vocab Size | Tokenizer Type | Avg Tokens per English Word |
|-------|-----------|----------------|---------------------------|
| BERT | 30,522 | WordPiece | ~1.4 |
| GPT-2 | 50,257 | Byte-level BPE | ~1.3 |
| Llama 2 | 32,000 | SentencePiece BPE | ~1.4 |
| GPT-4 | ~100,256 | Byte-level BPE | ~1.2 |
| Llama 3 | 128,256 | Byte-level BPE (tiktoken) | ~1.1 |
| GPT-4o | 200,019 | Byte-level BPE | ~1.0 |

### The Multilingual Tax

Tokenizers trained primarily on English are brutal to other languages. Korean text in GPT-2's tokenizer averages 2-3 tokens per word. Chinese can be worse. This means a Korean user effectively has a context window that is half the size of an English user's -- paying the same price for less information density.

This is why Llama 3 quadrupled its vocabulary from 32K to 128K. More tokens dedicated to non-English scripts means fairer compression across languages.

```figure tokenizer-tradeoff
```

## Build It

### Step 1: Character-Level Tokenizer

Start at the foundation. A character-level tokenizer maps each character to its Unicode code point. No training needed. No unknown tokens. Just a direct mapping.

```python
class CharTokenizer:
    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)
```

"hello" becomes [104, 101, 108, 108, 111]. Every character is its own token. This is the baseline we improve on.

### Step 2: BPE Tokenizer from Scratch

The real implementation. We train on raw bytes (like GPT-2), count pairs, merge the most frequent, and record every merge in order. The merge table is the tokenizer.

```python
from collections import Counter

class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {}

    def _get_pairs(self, tokens):
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def _merge_pair(self, tokens, pair, new_token):
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(new_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def train(self, text, num_merges):
        tokens = list(text.encode("utf-8"))
        self.vocab = {i: bytes([i])
