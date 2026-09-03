# Multi-Label Toxic Comment Classification

An end-to-end NLP pipeline for detecting multiple toxicity categories in online comments — from EDA and classical ML baselines to DistilBERT fine-tuning.

> **F1-Score: 0.71 (best classical baseline) → 0.81 (DistilBERT) · 98.7% accuracy**

---

## Problem Statement

Given a comment, predict which of 6 toxicity labels apply simultaneously:
`toxic` · `severe_toxic` · `obscene` · `threat` · `insult` · `identity_hate`

This is a **multi-label classification** problem — a single comment can belong to multiple categories at once. Dataset: [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (Kaggle).

---

## Pipeline Overview

```
Raw Text
   │
   ▼
EDA & Data Quality          ← class imbalance, label correlation, length stats
   │
   ▼
Text Preprocessing          ← lowercase, remove URLs/HTML/punctuation, lemmatize
   │
   ▼
TF-IDF Vectorization        ← unigrams + bigrams
   │
   ├──▶ Classical ML (OvR)  ← Logistic Regression, Naive Bayes, Linear SVM
   │
   └──▶ DistilBERT          ← pretrained encoder + custom Dropout + Dense head
              │
              ▼
         Evaluation & Comparison
```

---

## Notebooks

| # | Notebook | Description |
|---|---|---|
| 1 | [EDA + Preprocessing](https://github.com/Aseem2004/Toxic_Comment_Classification/blob/main/01%20-%20EDA_Preprocessing.ipynb) | Data quality checks, label distribution, correlation analysis, text cleaning |
| 2 | [TF-IDF Vectorization](https://github.com/Aseem2004/Toxic_Comment_Classification/blob/main/02%20-%20Vectorization.ipynb) | Unigram + bigram feature extraction |
| 3 | [Classical ML Models](https://github.com/Aseem2004/Toxic_Comment_Classification/blob/main/03%20-%20Classical%20ML.ipynb) | Logistic Regression, Naive Bayes, Linear SVM with One-vs-Rest strategy |
| 4 | [DistilBERT Fine-tuning](https://github.com/Aseem2004/Toxic_Comment_Classification/blob/main/04%20-%20DistilBERT.ipynb) | Transfer learning with custom TF/Keras classification head using DistilBERT |

---

## Results

### Classical ML (One-vs-Rest · TF-IDF features)

| Model | Accuracy | F1-Score (Micro) |
|---|---|---|
| Logistic Regression | 0.918628 | 0.670937 |
| Multinomial Naive Bayes | 0.910951 | 0.554459 |
| **Linear SVM** | 0.918565 | **0.711642** ✦ best baseline |

### DistilBERT (Transfer Learning · Fine-tuned)

| Model | Accuracy | F1-Score (Micro) |
|---|---|---|
| **DistilBERT + Custom Head** | **98.7%** | **0.81** |

> DistilBERT improved F1 by **+14% over the best classical baseline** by capturing contextual semantics that TF-IDF features cannot represent.

---

## Key Techniques

- **Multi-label framing** — One-vs-Rest (OvR) strategy for classical models; sigmoid output layer for DistilBERT
- **Transfer learning** — pretrained DistilBERT weights fine-tuned on domain-specific toxic comment data
- **Custom architecture** — DistilBERT encoder → Dropout → Dense (sigmoid) built in TensorFlow/Keras
- **Text preprocessing** — URL/HTML stripping, stopword removal, lemmatization (classical pipeline)
- **Loss function** — Binary Crossentropy for independent per-label prediction

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Transformers | HuggingFace Transformers (DistilBERT) |
| Classical ML | Scikit-learn |
| NLP & Features | NLTK, TF-IDF |
| EDA & Viz | Pandas, NumPy, Matplotlib, Seaborn |
| Environment | Kaggle Notebooks |

---

## Project Structure

```
├── notebook_1_eda_preprocessing.ipynb
├── notebook_2_tfidf_vectorization.ipynb
├── notebook_3_classical_ml.ipynb
└── notebook_4_distilbert_finetuning.ipynb
```
