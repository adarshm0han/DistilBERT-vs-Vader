# 📰 News Sentiment Analysis: DistilBERT vs VADER

This project provides a **lightweight, serverless news-sentiment analysis service** that compares a rules-based lexicon model (**VADER**) with a fine-tuned transformer model (**DistilBERT**).  
It fetches real-world news headlines, analyzes them, and displays results via a simple web UI.

---

## 🚀 Executive Summary
- **Goal:** Directly compare VADER and a modern transformer on news headlines.  
- **Approach:** Serverless deployment (AWS Lambda + Hugging Face Inference API).  
- **Output:** Both models’ scores displayed in a shared "compound-like" scale for intuitive comparison.  
- **Key Result:** Fine-tuned DistilBERT achieved **81% accuracy** and **0.809 macro-F1** on held-out test data.

---

## 🎯 Project Objectives
- Deliver a **production-style comparison** of rule-based vs. transformer sentiment models.  
- Maintain a **lightweight, cost-efficient architecture** using AWS Lambda.  
- Provide **intuitive visualizations** by mapping DistilBERT outputs into a VADER-style scale.  

---

## 📊 Data & Labeling
- **Task:** 3-class sentiment classification (NEGATIVE, NEUTRAL, POSITIVE).  
- **Training Data:** News headlines.  
- **Label Mapping:** `NEGATIVE → 0`, `NEUTRAL → 1`, `POSITIVE → 2`.  
- **Test Set:** 360 balanced headlines (120 per class).  
- **Preprocessing:** Headlines truncated at `max_length=256` (handled by DistilBERT uncased).  
- **Consistency:** Both VADER and DistilBERT were fed only the headline text.  

---

## 🤖 Model & Training
- **Base Model:** `distilbert-base-uncased` (Hugging Face).  
- **Architecture:** Sequence classification with `num_labels=3`.  
- **Tokenizer:** `DistilBertTokenizer` with dynamic padding.  
- **Hyperparameters:**  
  - `learning_rate = 2e-5`  
  - `epochs = 3`  
  - `batch_size = 16 (train) / 32 (eval)`  
  - `weight_decay = 0.01`  
  - `max_length = 256`  
- **Reproducibility:** `seed=42`.  
- **Artifacts:** Model + `label_map.json` saved to `./distilbert_sentiment_model` and uploaded to Hugging Face Hub.  

---

## 📈 Evaluation
- **Accuracy:** 0.8083  
- **Macro-F1:** 0.8087  
- **Macro Precision / Recall:** 0.8118 / 0.8083  

**Per-class Metrics:**
| Class     | Precision | Recall | F1    |
|-----------|-----------|--------|-------|
| NEGATIVE | 0.87      | 0.78   | 0.825 |
| NEUTRAL  | 0.765     | 0.842  | 0.802 |
| POSITIVE | 0.80      | 0.80   | 0.80  |

👉 Misclassifications were mostly between **NEUTRAL** and the polar classes, which is typical for short, ambiguous news text.  

---

## ⚙️ Serving & Integration

### Backend Flow (AWS Lambda)
1. **Fetch:** Retrieve headlines from [TheNewsAPI](https://www.thenewsapi.com).  
2. **Cache:** Save responses in `/tmp` with unique AM/PM identifiers (reduces API calls).  
3. **VADER:** Compute sentiment compound score (−1…1).  
4. **DistilBERT (HF Inference):** Call Hugging Face Inference API (with retry on 503).  
5. **Output:** Return unified JSON with both scores.  

**Example JSON Output:**
```json
{
  "articles": [
    {
      "title": "Example headline",
      "source": "domain.com",
      "url": "https://...",
      "publishedAt": "2025-09-24T12:05:00Z",
      "vader": { "compound": -0.681 },
      "distilbert": { "label": "NEGATIVE", "score": 0.671, "compound": -0.440 }
    }
  ]
}
