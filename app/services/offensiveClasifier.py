from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


BASE_DIR  = Path(__file__).resolve().parents[2]        # SolidarianIDLenguajeProject/
MODEL_DIR = (BASE_DIR / "models" / "transformer_colab_distilbeto" / "final_model").resolve()


if not MODEL_DIR.is_dir():
    raise FileNotFoundError(f"⚠️  No encuentro el modelo en {MODEL_DIR}")

# --- Se carga una sola vez al importar -----------------------------------------
_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
_model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
_model.eval().to("cuda" if torch.cuda.is_available() else "cpu")


def classify(text: str) -> Tuple[bool, str]:
    """
    Devuelve (es_ofensivo, etiqueta_texto)
      • es_ofensivo : bool
      • etiqueta_texto : 'ofensivo' | 'no_ofensivo'
    """
    device = _model.device
    inputs = _tokenizer(text, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        logits = _model(**inputs).logits
    pred = int(torch.argmax(logits, dim=-1).item())   # 0 ó 1

    return pred == 1, ("ofensivo" if pred == 1 else "no_ofensivo")
