"""
=============================

1. Carga un CSV con columnas: text, label
2. Divide en entrenamiento y validación (stratify)
3. Extrae características con TfidfVectorizer (1-2 gramas)
4. Entrena uno o varios clasificadores
5. Muestra classification_report y matriz de confusión
6. Guarda los modelos entrenados como .pkl para reutilizarlos


Ejemplo de uso:
---------------
python train_tfidf_classifiers_v2.py --csv ../../data/final/final_train.csv --dev-ratio 0.1 --models linear_svc  --outdir ../../models
python train_tfidf_classifiers_v2.py --csv ../../data/final/final_train.csv --dev-ratio 0.1 --models logreg       --outdir ../../models
python train_tfidf_classifiers_v2.py --csv ../../data/final/final_train.csv --dev-ratio 0.1 --models multinomial_nb --outdir ../../models
python train_tfidf_classifiers_v2.py --csv ../../data/final/final_train.csv --dev-ratio 0.1 --models random_forest --outdir ../../models

"""
from __future__ import annotations
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------------------------- #
# 1. Modelos disponibles                                                #
# --------------------------------------------------------------------- #
def get_available_models() -> dict[str, object]:
    """
    Devuelve un diccionario «nombre ➜ instancia de clasificador».

    •  LinearSVC         – SVM lineal
    •  LogisticRegression– Modelo probabilístico
    •  MultinomialNB     – Naïve Bayes multinomial
    •  RandomForest      – Bosque de árboles
    """
    return {
        "linear_svc": LinearSVC(class_weight="balanced"),
        "logreg":     LogisticRegression(max_iter=1000,
                                         class_weight="balanced",
                                         n_jobs=-1),
        "multinomial_nb": MultinomialNB(),
        "random_forest":  RandomForestClassifier(n_estimators=300,
                                                 class_weight="balanced",
                                                 n_jobs=-1,
                                                 random_state=0),
    }


# --------------------------------------------------------------------- #
# 2. Funciones auxiliares                                               #
# --------------------------------------------------------------------- #
def build_vectorizer() -> TfidfVectorizer:
    """
    Crea un TF-IDF que:

    • convierte a minúsculas
    • elimina acentos (strip_accents='unicode')
    • considera uni- y bi-gramas
    • limita el vocabulario a 50 000 tokens más frecuentes
    """
    return TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents="unicode",
    )


def print_conf_matrix(cm: np.ndarray) -> None:
    """
    Imprime la matriz de confusión en formato tabla.
    """
    print("\nMatriz de confusión")
    print("            pred 0   pred 1")
    print(f"real 0    {cm[0,0]:7d} {cm[0,1]:7d}")
    print(f"real 1    {cm[1,0]:7d} {cm[1,1]:7d}")


# --------------------------------------------------------------------- #
# 3. Script principal                                                   #
# --------------------------------------------------------------------- #
def main() -> None:
    args = get_args()
    np.random.seed(args.seed)

    # 3.1  Leer datos --------------------------------------------------- #
    df = pd.read_csv(args.csv)
    X = df["text"]
    y = df["label"]

    # 3.2  Train / Dev  ------------------------------------------------- #
    X_train, X_dev, y_train, y_dev = train_test_split(
        X, y,
        test_size=args.dev_ratio,
        stratify=y,
        random_state=args.seed
    )

    # 3.3  Ajustar vectorizador solo con TRAIN -------------------------- #
    vect = build_vectorizer()
    X_train_tfidf = vect.fit_transform(X_train)
    X_dev_tfidf   = vect.transform(X_dev)

    # 3.4  Entrenar y evaluar cada modelo ------------------------------ #
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    models_available = get_available_models()
    for name in args.models:
        if name not in models_available:
            raise ValueError(f"Modelo '{name}' no reconocido. "
                             f"Opciones: {list(models_available)}")

        print(f"\n================ {name.upper()} ================\n")

        clf = models_available[name]
        clf.fit(X_train_tfidf, y_train)         # entrenamiento

        # ---------- Evaluación ----------
        y_pred = clf.predict(X_dev_tfidf)
        print(classification_report(y_dev, y_pred, digits=4))

        cm = confusion_matrix(y_dev, y_pred)
        print_conf_matrix(cm)



        # ---------- Guardar modelo + vectorizador ----------
        fname = outdir / f"tfidf_{name}.pkl"
        joblib.dump((vect, clf), fname)
        print("Modelo guardado en", fname.resolve())


# --------------------------------------------------------------------- #
# 4. CLI argument parser                                                #
# --------------------------------------------------------------------- #
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",      required=True,
                   help="CSV con columnas text,label (mezcla OFF + NO OFF).")
    p.add_argument("--dev-ratio", type=float, default=0.1,
                   help="Proporción de instancias reservadas para validación.")
    p.add_argument("--models", nargs="+",
                   default=["linear_svc"],
                   help="Lista de modelos a entrenar. "
                        "Ver get_available_models() para nombres.")
    p.add_argument("--outdir",   default="../../models")
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()
