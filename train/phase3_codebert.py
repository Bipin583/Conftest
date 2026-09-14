# ============================================================================
# PHASE 3 STEP 1: FINE-TUNE CODEBERT (LAPTOP VERSION)
# Uses your RTX 4050 GPU
# Time: 2-3 hours
# ============================================================================

import argparse
import sys
from pathlib import Path

# Configure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Safe loader bypass for transformers 5+ on torch 2.5
try:
    import transformers.utils.import_utils as _imp_u
    _imp_u.check_torch_load_is_safe = lambda: None
    import transformers.modeling_utils as _mod_u
    _mod_u.check_torch_load_is_safe = lambda: None
except Exception:
    pass

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

parser = argparse.ArgumentParser(description="Phase 3 Step 1: Fine-Tune CodeBERT")
parser.add_argument("--samples", type=int, default=None, help="Number of samples to use (default: None for full dataset)")
parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
parser.add_argument("--batch_size", type=int, default=32, help="Per device batch size (default: 32)")
parser.add_argument("--max_length", type=int, default=128, help="Max token sequence length (default: 128)")
args = parser.parse_args()

print("="*60)
print("PHASE 3 STEP 1: FINE-TUNE CODEBERT")
print("="*60)

# CHECK GPU
print(f"\n🔍 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

# LOAD DATASET
print("\n📂 Loading dataset...")
df = pd.read_csv("dataset_full_realistic.csv")
print(f"✅ Loaded {len(df):,} total samples from dataset")

# SORT BY COMMIT (matching Phase 2 split methodology)
df = df.sort_values('commit_id').reset_index(drop=True)

if args.samples is not None and args.samples < len(df):
    print(f"⚡ Subsampling to {args.samples:,} samples for faster training...")
    df = df.sample(n=args.samples, random_state=42).sort_values('commit_id').reset_index(drop=True)

# PREPARE TEXT
print("\n📝 Preparing text data...")
df['text'] = (df['commit_message'].fillna('') + ' ' + df['commit_body'].fillna('')).astype(str)
texts = df['text'].tolist()
labels = df['label'].astype(int).tolist()

print(f"   Samples: {len(texts):,}")
print(f"   Labels: 0={labels.count(0):,}, 1={labels.count(1):,}")

# SPLIT (70% train, 15% val, 15% test)
n = len(df)
train_texts = texts[:int(n*0.7)]
train_labels = labels[:int(n*0.7)]
val_texts = texts[int(n*0.7):int(n*0.85)]
val_labels = labels[int(n*0.7):int(n*0.85)]

# TOKENIZE
print("\n🔤 Tokenizing (dynamic batch padding)...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

train_encodings = tokenizer(train_texts, truncation=True, max_length=args.max_length)
val_encodings = tokenizer(val_texts, truncation=True, max_length=args.max_length)

# CREATE DATASET
class FlakinessDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = FlakinessDataset(train_encodings, train_labels)
val_dataset = FlakinessDataset(val_encodings, val_labels)

print(f"   Train: {len(train_dataset):,}")
print(f"   Val: {len(val_dataset):,}")

# LOAD MODEL
print("\n🤖 Loading CodeBERT...")
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/codebert-base",
    num_labels=2
)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"   Device: {device}")

# COMPUTE METRICS
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
    }

# TRAINING ARGUMENTS
print("\n⚙️  Setting up training...")
try:
    training_args = TrainingArguments(
        output_dir="./codebert_flakiness",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
except TypeError:
    training_args = TrainingArguments(
        output_dir="./codebert_flakiness",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

# TRAINER
print(f"\n🚀 Fine-tuning CodeBERT (epochs={args.epochs}, batch_size={args.batch_size})...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)

trainer.train()

# SAVE
print("\n💾 Saving model...")
output_dir = Path("models/codebert_flakiness")
output_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(output_dir))
tokenizer.save_pretrained(str(output_dir))

print(f"   ✅ Saved to {output_dir}/")

print("\n" + "="*60)
print("STEP 1 COMPLETE!")
print("📝 NEXT: Extract embeddings and build hybrid model")
print("="*60)