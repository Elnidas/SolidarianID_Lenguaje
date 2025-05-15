
"""

Fusiona *todos* los corpus de discurso ofensivo/odio en español indicados en
`DATASETS`, los limpia y genera **un único** particionado aleatorio:

Al final del proceso imprime el *conteo* y el *porcentaje* de cada clase en
los dos conjuntos.

"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore

DATA_DIR = Path("../../data")  # directorio raíz con los ficheros
OUT_DIR = Path("../../data/final")  # directorio raíz con los ficheros

# ---------------------------------------------------------------------------
# Expresiones regulares para limpieza
# ---------------------------------------------------------------------------

USER_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Elimina menciones, URLs y espacios repetidos."""
    if pd.isna(text):
        return ""
    text = USER_RE.sub("", str(text))
    text = URL_RE.sub("", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()

# ---------------------------------------------------------------------------
# Carga genérica
# ---------------------------------------------------------------------------

def load_dataset(desc: Dict) -> pd.DataFrame:
    """Lee el archivo descrito en *desc* y devuelve DataFrame con columnas text/label."""
    path: Path = desc["path"]
    text_col: str = desc["text_col"]
    label_col: str = desc["label_col"]
    label_map: Optional[Dict] = desc.get("map")   # puede ser None si ya es 0/1
    sep: str = desc.get("sep", ",")  # para leer tanto csv como tsv

    df = pd.read_csv(path, sep=sep)

    df = df.rename(columns={text_col: "text", label_col: "orig_label"})


    if label_map is not None:
        df["label"] = df["orig_label"].map(label_map)
    else:
        df["label"] = df["orig_label"]  # asumir valores ya binarios

    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df["text"]  = df["text"].astype(str).apply(clean_text)
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    df["text"] = df["text"].astype(str)

    return df[["text", "label"]]

# ---------------------------------------------------------------------------
# Descriptores de datasets
# ---------------------------------------------------------------------------

MAP_OFFENDES = {"OFP": 1, "OFG": 1, "NOE": 0, "NO": 0}

DATASETS: List[Dict] = [
    # OffendES (train, dev, test)
    {
        "path": DATA_DIR / "training_set.tsv",
        "text_col": "comment",
        "label_col": "label",
        "map": MAP_OFFENDES,
        "sep": "\t",
    },
    {
        "path": DATA_DIR / "dev_set.tsv",
        "text_col": "comment",
        "label_col": "label",
        "map": MAP_OFFENDES,
        "sep": "\t",
    },
    {
        "path": DATA_DIR / "test_set.tsv",
        "text_col": "comment",
        "label_col": "label",
        "map": MAP_OFFENDES,
        "sep": "\t",
    },
    # Otros corpus
    {
        "path": DATA_DIR / "hatenet.csv",
        "text_col": "tweet",
        "label_col": "label",
        "map": {"hateful": 1, "non_hateful": 0},
    },
    {
        "path": DATA_DIR / "full_dataset_1.csv",
        "text_col": "tweet",
        "label_col": "label",
        "map": {"misogyny": 1, "non-misogyny": 0},
    },
    {
        "path": DATA_DIR / "Ami.csv",
        "text_col": "tweet",
        "label_col": "label",
        "map": {"misogynous": 1, "non_misogynous": 0},
    },
    {
        "path": DATA_DIR / "hateeval.csv",
        "text_col": "tweet",
        "label_col": "label",
        "map": {"hatespeech": 1, "non_hatespeech": 0},
    },
    # Spanish Hate Speech Superset (ya 0/1)
    {
        "path": DATA_DIR / "spanish_hate_speech_superset_train.csv",
        "text_col": "text",
        "label_col": "labels",
        "map": None,  # valores numéricos
    },
]

# ---------------------------------------------------------------------------
# Función auxiliar para estadísticas
# ---------------------------------------------------------------------------

def stats(df: pd.DataFrame) -> str:
    counts = df["label"].value_counts().sort_index()
    total = len(df)
    pct = (counts / total * 100).round(2)
    return (
        f"No ofensivo(0): {counts.get(0,0):,}  — {pct.get(0,0.0)}%\n"
        f"Ofensivo  (1): {counts.get(1,0):,}  — {pct.get(1,0.0)}%\n"
        f"Total          : {total:,}"
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(test_size: float = 0.2, out_dir: Path = DATA_DIR, seed: int = 42):
    # 1. Cargar y concatenar todos los datasets
    frames = [load_dataset(cfg) for cfg in DATASETS]
    full_df = pd.concat(frames, ignore_index=True)

    # 2. Eliminar duplicados exactos
    full_df.drop_duplicates(subset="text", inplace=True)

    # 3. Split estratificado train / test
    train_df, test_df = train_test_split(
        full_df, test_size=test_size, random_state=seed, stratify=full_df["label"], shuffle=True
    )


    # 4. Guardar
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "final_train.csv"
    test_path  = out_dir / "final_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # 5. Estadísticas
    print("── Train set ─────────────")
    print(stats(train_df))
    print(f"→ {train_path}\n")

    print("── Test set ──────────────")
    print(stats(test_df))
    print(f"→ {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusiona datasets y genera partición train/test")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proporción para el conjunto de prueba")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR, help="Directorio de salida de los CSV resultantes")
    args = parser.parse_args()
    main(test_size=args.test_size, out_dir=Path(args.out_dir))
