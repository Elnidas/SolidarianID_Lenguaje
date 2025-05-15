
"""


Modelos disponibles (--model):
  beto        → dccuchile/bert-base-spanish-wwm-uncased
  maria       → PlanTL-GOB-ES/roberta-base-bne
  distilbeto  → dccuchile/distilbert-base-spanish-uncased

Ejemplo:
python train_transformers_v3.py --train-file ../../data/v2/final_train.csv --model beto       --epochs 3 --batch-size 16 --dev-ratio 0.1 --outdir ../../models/transformer_colab_beto_2
python train_transformers_v3.py --train-file ../../data/v2/final_train.csv --model maria      --epochs 3 --batch-size 16 --dev-ratio 0.1 --outdir ../../models/transformer_colab_maria_2
python train_transformers_v3.py --train-file ../../data/v2/final_train.csv --model distilbeto --epochs 3 --batch-size 16 --dev-ratio 0.1 --outdir ../../models/transformer_colab_distilbeto_2


"""
from __future__ import annotations
import argparse, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from optimum.bettertransformer import BetterTransformer   # ⚡ optimización

# ──────────────────────────────────────────────────────────────────────────────
# 1. Catálogo de modelos (alias ➜ identificador Hugging-Face)
# ──────────────────────────────────────────────────────────────────────────────
MODEL_ZOO = {
    "beto":       "dccuchile/bert-base-spanish-wwm-uncased",
    "maria":      "PlanTL-GOB-ES/roberta-base-bne",
    "distilbeto": "dccuchile/distilbert-base-spanish-uncased",
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. Funciones auxiliares
# ──────────────────────────────────────────────────────────────────────────────
def load_dataset(csv_path: str) -> Dataset:
    """Lee un CSV con columnas *text,label* y lo pasa a `datasets.Dataset`."""
    return Dataset.from_pandas(pd.read_csv(csv_path))


def tokenize(batch, tokenizer):
    """Tokeniza un lote de ejemplos; truncado a 256 tokens."""
    return tokenizer(batch["text"], truncation=True, max_length=256)


def macro_f1(pred):
    """Función de métrica que devuelve F1-macro (requisito de la práctica)."""
    logits, labels = pred
    preds = logits.argmax(-1)
    return {"macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}


# ──────────────────────────────────────────────────────────────────────────────
# 3. Focal-loss con α (class-weight) y γ (dificultad)
# ──────────────────────────────────────────────────────────────────────────────
def make_focal_loss(alpha: torch.Tensor, gamma: float = 2.0):
    """
    Devuelve una función focal-loss cerrando sobre los pesos alpha.
    • alpha: tensor tamaño [num_labels] con el peso por clase.
    • gamma: factor de modulación de dificultad (γ=2 → estándar).
    """
    def focal_loss(outputs, labels):
        """
        outputs: logits (batch, classes)
        labels : ground-truth  (batch,)
        """
        ce = torch.nn.functional.cross_entropy(
            outputs, labels, reduction="none", weight=alpha
        )
        pt = torch.exp(-ce)        # probabilidad de la clase correcta
        return ((1 - pt) ** gamma * ce).mean()
    return focal_loss


# ──────────────────────────────────────────────────────────────────────────────
# 4. Script principal
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    # 4-A) Preparar dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🔹 Dispositivo:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU")

    # 4-B) Datos
    full_ds = load_dataset(args.train_file)
    idx_train, idx_dev = train_test_split(
        np.arange(len(full_ds)),
        test_size=args.dev_ratio,
        stratify=full_ds["label"],
        random_state=args.seed,
    )
    train_ds = full_ds.select(idx_train)
    dev_ds   = full_ds.select(idx_dev)

    # 4-C) Tokenización (map batched=True para eficiencia)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ZOO[args.model])
    train_ds = train_ds.map(lambda b: tokenize(b, tokenizer), batched=True)
    dev_ds   = dev_ds.map(lambda b: tokenize(b, tokenizer),   batched=True)

    # 4-D) Carga del modelo base
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ZOO[args.model], num_labels=2
    ).to(device)

    # 4-E) Pesos de clase (alpha) inversos a la frecuencia
    y_np = np.array(train_ds["label"])
    freq = np.bincount(y_np)                 # [n_0, n_1]
    alpha = torch.tensor(len(y_np) / (2 * freq),
                         dtype=torch.float32,
                         device=device)      # tamaño [2]
    print("α (pesos de clase):", alpha.tolist())

    # 4-F) Sustituir compute_loss por focal-loss
    focal_fn = make_focal_loss(alpha=alpha, gamma=args.gamma)
    model.compute_loss = lambda outputs, labels: focal_fn(outputs, labels)

    # 4-G) Optimización de inferencia (BetterTransformers)
    if args.better:
        t0 = time.time()
        model = BetterTransformer.transform(model)
        print(f"✅ BetterTransformers aplicado ({time.time()-t0:.2f}s)")

    # 4-H) Argumentos de entrenamiento (idénticos al colab + tu script)
    training_args = TrainingArguments(
        output_dir=args.outdir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 4,
        evaluation_strategy="epoch",
        logging_steps=20,
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
    )

    # 4-I) Entrenamiento
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=macro_f1,
    )
    trainer.train()

    # 4-J) Guardar modelo + tokenizer
    final_dir = Path(args.outdir) / "final_model"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("📦 Modelo final guardado en", final_dir.resolve())


# ──────────────────────────────────────────────────────────────────────────────
# 5. CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", required=True,
                   help="CSV con columnas text,label (entrenamiento)")
    p.add_argument("--model", choices=list(MODEL_ZOO), default="maria",
                   help="Alias del Transformer a fine-tunear")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dev-ratio", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=2.0,
                   help="γ para focal-loss (2.0 estándar)")
    p.add_argument("--better", action="store_true",
                   help="Aplica BetterTransformers si se indica")
    p.add_argument("--outdir", default="../../models/transformer_focal")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()
