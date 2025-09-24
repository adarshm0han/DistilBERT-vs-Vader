# ...existing code...
import json
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
from sklearn.model_selection import train_test_split

# reproducibility
set_seed(42)

# ===== Step 1: Load and Prepare Dataset =====
data = pd.read_csv("train.csv")

# use plural 'labels' to be compatible with various transformers versions
label_mapping = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
data["labels"] = data["label"].map(label_mapping)

# stratify to preserve class distribution
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data["text"].tolist(),
    data["labels"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=data["labels"]
)

train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "labels": val_labels})

# ===== Step 2: Tokenize =====
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)  # no fixed padding

train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=["text"])
val_dataset = val_dataset.map(preprocess, batched=True, remove_columns=["text"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataset.set_format(type="torch")
val_dataset.set_format(type="torch")

# ===== Step 3: Model =====
id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
label2id = {v: k for k, v in id2label.items()}

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)

# save label mapping for inference API
with open("label_map.json", "w", encoding="utf-8") as f:
    json.dump(id2label, f, ensure_ascii=False)

# ===== Step 4: Training Args =====
training_args = TrainingArguments(
    output_dir="./distilbert_sentiment_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    # Note: some transformers installs don't accept evaluation_strategy; keep this compatible
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
    report_to="none",
    fp16=False,  # set to True if CUDA available and you want mixed precision
)

# ===== Step 5: Metrics =====
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ===== Step 6: Train & Save =====
trainer.train()

# Run evaluation explicitly (works regardless of TrainingArguments support)
eval_metrics = trainer.evaluate(eval_dataset=val_dataset)
print("Evaluation metrics:", eval_metrics)

trainer.save_model("./distilbert_sentiment_model")
tokenizer.save_pretrained("./distilbert_sentiment_model")
print("Model training complete. Saved to ./distilbert_sentiment_model")
