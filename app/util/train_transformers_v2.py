
"""
Modelos disponibles (--model):
  beto        → dccuchile/bert-base-spanish-wwm-uncased
  maria       → PlanTL-GOB-ES/roberta-base-bne
  distilbeto  → dccuchile/distilbert-base-spanish-uncased

Ejemplo:
python train_transformers_v2.py --train-file ../../data/final/final_train.csv --model beto       --epochs 3 --batch-size 16 --dev-ratio 0.1 --outdir ../../models/transformer_colab_beto
python train_transformers_v2.py --train-file ../../data/final/final_train.csv --model maria      --epochs 3 --batch-size 16 --dev-ratio 0.1 --outdir ../../models/transformer_colab_maria
python train_transformers_v2.py --train-file ../../data/final/final_train.csv --model distilbeto --epochs 3 --batch-size 16 --dev-ratio 0.1 --outdir ../../models/transformer_colab_distilbeto
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

# -------------------------- 1. Catálogo de modelos ------------------------- #
MODEL_LIST = {
    "beto":       "dccuchile/bert-base-spanish-wwm-uncased",
    "maria":      "PlanTL-GOB-ES/roberta-base-bne",
    "distilbeto": "dccuchile/distilbert-base-spanish-uncased",
}


# -------------------------- 2. Utilidades ---------------------------------- #
def load_dataset(csv_path: str) -> Dataset:
    return Dataset.from_pandas(pd.read_csv(csv_path))


def tokenize_function(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, max_length=256)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {"f1": f1_score(labels, preds, average="macro", zero_division=0)}


# -------------------------- 3. Script principal ---------------------------- #
def main() -> None:
    args = get_args()
    set_seed(args.seed)

    model_name = MODEL_LIST[args.model]        # traduce alias → HF hub id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Modelo:", args.model, "→", model_name)
    print("Dispositivo:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU")

    # 1) Datos ----------------------------------------------------------------
    full_ds = load_dataset(args.train_file)
    idx = np.arange(len(full_ds))
    train_idx, dev_idx = train_test_split(
        idx,
        test_size=args.dev_ratio,
        stratify=full_ds["label"],
        random_state=args.seed,
    )
    train_ds = full_ds.select(train_idx)
    dev_ds   = full_ds.select(dev_idx)

    # 2) Tokenización ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = train_ds.map(lambda b: tokenize_function(b, tokenizer), batched=True)
    dev_ds   = dev_ds.map(lambda b: tokenize_function(b, tokenizer),   batched=True)

    # 3) Modelo ---------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    # 4) Argumentos de entrenamiento -----------------------------------------
    training_args = TrainingArguments(
        output_dir=args.outdir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 4,
        evaluation_strategy="epoch",
        logging_steps=20,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    final_dir = Path(args.outdir) / "final_model"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("✅ Modelo y tokenizer guardados en", final_dir.resolve())


# -------------------------- 4. Argumentos CLI ------------------------------ #
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", required=True,
                   help="CSV con columnas text,label (conjunto de entrenamiento)")
    p.add_argument("--model", choices=list(MODEL_LIST), default="beto",
                   help="Alias del modelo a fine-tunear")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dev-ratio", type=float, default=0.1)
    p.add_argument("--outdir", default="../../models/transformer_colab")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()
