---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
description-zh: # Sentiment Analysis Baseline Design

## 1. Data Exploration & Preparation

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

# Load and inspect
df = pd.read_csv("dataset.csv")
print(f"Shape: {df.shape}")
print(f"Label distribution:\n{df['sentiment'].value_counts(normalize=True)}")
print(f"Null values:\n{df.isnull().sum()}")
print(f"Avg text length: {df['text'].str.len().mean():.0f} chars")
```



**Key checks:**
- Class imbalance ratio
- Text language(s), encoding issues
- Average token count per sample
- Duplicates

---

## 2. Text Preprocessing Pipeline

```python
import re
import nltk
from nltk.corpus import stopwords

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # URLs
    text = re.sub(r"
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm, aspect-based output, or cross-lingual coverage.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples. Never report accuracy alone on imbalanced data.
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two. Suggest a weekly sample audit.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced. Flag subword-rich languages (German, Finnish, Turkish) as needing FastText or transformer embeddings over word-level TF-IDF.
