---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
description-zh: # LDA vs. BERTopic for Topic Modeling

## Quick Decision

| Factor | LDA | BERTopic |
|---|---|---|
| Corpus size | 10K+ documents | Any (even small) |
| Short texts (tweets, titles) | Poor | Excellent |
| Interpretability | Good (bag-of-words) | Good (c-TF-IDF words) |
| Speed | Fast (CPU) | Slower (needs embeddings) |
| GPU required | No | Recommended |
| Overlapping topics | Yes (soft clustering) | Less natural |
| Domain with rare jargon | Decent | Better (semantic embeddings) |

**Rule of thumb**: Default to **BERTopic** unless you need probabilistic soft assignments or lack GPU resources.

---

## Option A: LDA

### Library
```python
from gensim.models import LdaMulticore, CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
```

### Knobs to Tune

| Knob |
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics (start at ~sqrt(n_docs)), `min_df` / `max_df` filters, embedding model for neural approaches.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, plus a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 or above 200 as likely wrong for real data.
