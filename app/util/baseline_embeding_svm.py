"""
Entrena un clasificador SVM (kernel lineal) con embeddings FastText
y guarda tanto el scaler como el modelo en ../../models/svm_model.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd
import fasttext
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import Normalizer

# ------------------------------------------------------------------ #
# Embeddings                                                         #
# ------------------------------------------------------------------ #

FASTTEXT_BIN = "../../embeddings-s-model.bin"
fasttext_model = fasttext.load_model(FASTTEXT_BIN)

def sentence_embedding(text: str) -> np.ndarray:
    """
    Convierte una oración en el promedio de los vectores FastText.
    Si no hay tokens válidos devuelve un vector-cero.
    """
    tokens = text.lower().split()
    if not tokens:
        return np.zeros(fasttext_model.get_dimension(), dtype=np.float32)
    vecs = [fasttext_model.get_word_vector(tok) for tok in tokens]
    return np.mean(vecs, axis=0)

# ------------------------------------------------------------------ #
# Entrenamiento y evaluación                                         #
# ------------------------------------------------------------------ #

def main() -> None:
    # 1) Cargar datos
    df_train = pd.read_csv("../../data/final_train.csv")
    df_test  = pd.read_csv("../../data/final_test.csv")

    # 2) Construir matrices X, y
    X_train = np.vstack([sentence_embedding(t) for t in df_train["text"]])
    X_test  = np.vstack([sentence_embedding(t) for t in df_test["text"]])

    y_train = df_train["label"].to_numpy()
    y_test  = df_test["label"].to_numpy()

    # 3) Normalizar vectores (L2) – muy recomendable para SVM
    scaler = Normalizer()          # normaliza cada fila a norma 1
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 4) Entrenar SVM lineal con pesos balanceados
    clf = SVC(kernel="linear", C=1.0, class_weight="balanced", probability=False)
    clf.fit(X_train, y_train)

    # 5) Evaluar
    y_pred = clf.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # 6) Guardar scaler + modelo
    os.makedirs("../../models", exist_ok=True)
    joblib.dump((scaler, clf), "../../models/svm_model.pkl")
    print("✅ Modelo y scaler guardados en ../../models/svm_model.pkl")

if __name__ == "__main__":
    main()
