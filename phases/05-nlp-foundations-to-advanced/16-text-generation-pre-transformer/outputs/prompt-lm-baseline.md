---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
description-zh: # Reproducible N-Gram Language Model Baseline

## Project Structure

```
ngram_lm/
├── config.yaml
├── requirements.txt
├── data/
│   └── preprocess.py
├── model/
│   ├── ngram.py
│   └── smoothing.py
├── evaluate.py
├── generate.py
└── run_baseline.py
```

## 1. Configuration

```yaml
# config.yaml
data:
  corpus_path: "data/corpus.txt"
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  min_freq: 2              # OOV threshold

model:
  n: 3                     # trigram by default
  smoothing: "kneser_ney"  # options: add_k, kneser_ney, stupid_backoff, interpolated_kneser_ney
  add_k: 0.01              # for add-k smoothing
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn the math.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special `<UNK>` token during training.
