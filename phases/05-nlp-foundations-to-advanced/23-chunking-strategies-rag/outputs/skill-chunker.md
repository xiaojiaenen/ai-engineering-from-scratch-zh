---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
description-zh: 选择合适的文本分块（chunking）策略、大小（size）和重叠（overlap）对于信息检索和问答系统的效果至关重要。具体选择取决于**语料库特性**和**查询分布**。以下是一个清晰的决策流程：

### 1. **
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
