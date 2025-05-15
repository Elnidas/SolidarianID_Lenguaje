"""
Evalúa un modelo de lenguaje ofensivo sobre un CSV de test.

• Si --model-path termina en *.pkl    →  carga TF-IDF (Pipeline ó (vect, clf))
• Si es una carpeta con config.json   →  carga Transformer + tokenizer

# ------------------------------------------------------------------
# 1) Baselines TF-IDF  (archivos .pkl)
# ------------------------------------------------------------------

# Linear-SVC
python evaluate_model_v2.py --model-path ../../models/tfidf_linear_svc.pkl --data-file  ../../data/final/final_test.csv --out        ../../results/tfidf_linear_svc.json

# Logistic Regression
python evaluate_model_v2.py --model-path ../../models/tfidf_LogReg.pkl --data-file  ../../data/final/final_test.csv --out        ../../results/tfidf_logreg.json

# Multinomial Naïve Bayes
python evaluate_model_v2.py --model-path ../../models/tfidf_multinomial_nb.pkl --data-file  ../../data/final/final_test.csv --out        ../../results/tfidf_multinb.json

# Random Forest
python evaluate_model_v2.py --model-path ../../models/tfidf_random_forest.pkl --data-file  ../../data/final/final_test.csv --out        ../../results/tfidf_rf.json


# ------------------------------------------------------------------
# 2) SVM + FastText  (.pkl escala+clasificador)
# ------------------------------------------------------------------

python evaluate_model_v2.py --model-path ../../models/svm_model.pkl --data-file  ../../data/final/final_test.csv --out        ../../results/fasttext_svm.json


# ------------------------------------------------------------------
# 3) Transformers fine-tuneados (carpetas con config.json)
# ------------------------------------------------------------------

# BETO (versión colab)
python evaluate_model_v2.py --model-path ../../models/transformer_colab_beto/final_model --data-file  ../../data/final/final_test.csv --out        ../../results/beto_test_metrics.json

# MarIA
python evaluate_model_v2.py --model-path ../../models/transformer_colab_maria/final_model --data-file  ../../data/final/final_test.csv --out        ../../results/maria_test_metrics.json

# DistilBETO
python evaluate_model_v2.py --model-path ../../models/transformer_colab_distilbeto/final_model --data-file  ../../data/final/final_test.csv --out        ../../results/distilbeto_test_metrics.json
"""
import argparse, json, os, joblib, torch, pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import time, tracemalloc, psutil

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


# --------------------------------------------------------------------------- #
# utilidades                                                                  #
# --------------------------------------------------------------------------- #
def load_tfidf(pkl_path: Path):
    """Devuelve función predict(texts) para un modelo TF-IDF."""
    obj = joblib.load(pkl_path)
    if isinstance(obj, tuple):              # (vectorizer, clf)
        vectorizer, clf = obj
        def predict(texts):
            X = vectorizer.transform(texts)
            return clf.predict(X).tolist()
    else:                                   # Pipeline
        pipe = obj
        predict = lambda texts: pipe.predict(texts).tolist()
    return predict, "TFIDF"


def load_transformer(dir_path: Path):
    """Devuelve función predict(texts) para un modelo HF Transformer."""
    tokenizer = AutoTokenizer.from_pretrained(dir_path)
    model     = AutoModelForSequenceClassification.from_pretrained(dir_path)
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        batch_size=64,
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
    )
    def predict(texts):
        return [int(r["label"].split("_")[-1]) for r in pipe(texts)]
    return predict, "TRANSFORMER"


def load_model(path: str):
    path = Path(path)
    if path.is_file() and path.suffix == ".pkl":
        return load_tfidf(path)
    elif path.is_dir() and (path / "config.json").exists():
        return load_transformer(path)
    else:
        raise ValueError("Ruta no reconocida: debe ser .pkl o carpeta Transformer")






# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    args = get_args()

    # 1) Cargar modelo ------------------------------------------------------- #
    predict, kind = load_model(args.model_path)
    print(f"🔹 Modelo cargado ({kind}):", args.model_path)

    # 1-bis) Preparación de medición memoria / tiempo ------------------------ #
    proc = psutil.Process()  # proceso actual
    tracemalloc.start()  # mide memoria Python

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 2) Leer test ----------------------------------------------------------- #
    df = pd.read_csv(args.data_file)
    texts, y_true = df["text"].tolist(), df["label"].tolist()

    # 3) Inferencia ---------------------------------------------------------- #
    t0 = time.perf_counter()
    y_pred = predict(texts)
    inf_time = time.perf_counter() - t0

    # 3-bis) Métricas de memoria -------------------------------------------- #
    cur, peak_py = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_gpu = (
        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    )
    rss_after = proc.memory_info().rss

    # 4) Métricas ------------------------------------------------------------ #
    acc = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, digits=4, output_dict=True)
    report_str = classification_report(y_true, y_pred, digits=4)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)

    cm = confusion_matrix(y_true, y_pred).tolist()
    cm_df = pd.DataFrame(cm,  # DataFrame para darle el formato “real/pred”
                         index=["real 0", "real 1"],
                         columns=["pred 0", "pred 1"])

    # 5) Mostrar por pantalla ------------------------------------------------
    print(f"\nAccuracy  : {acc:.4f}")
    print("\n" + report_str)
    print("\nMatriz de confusión")
    print(cm_df.to_string())
    print(
        f"\nTiempo inferencia : {inf_time:.2f} s  "
        f"({inf_time / len(texts):.{4}f} s por muestra)"
    )
    print(
        f"Pico memoria CPU  : {peak_py / 1024 ** 2:.1f} MB  "
        f"(RSS final {rss_after / 1024 ** 2:.1f} MB)"
    )
    if peak_gpu:
        print(f"Pico memoria GPU  : {peak_gpu / 1024 ** 2:.1f} MB")

    # 6) Guardar resultados --------------------------------------------------
    if args.out:
        base = Path(args.out)
        base.parent.mkdir(parents=True, exist_ok=True)

        inference_block = {
            "total_seconds": inf_time,
            "per_sample_seconds": inf_time / len(texts),
            "peak_cpu_mb": peak_py / 1024 ** 2,
            "rss_after_mb": rss_after / 1024 ** 2,
            "peak_gpu_mb": peak_gpu / 1024 ** 2,
        }

        # -- JSON ------------------------------------------------------------
        json_path = base.with_suffix(".json")
        json_path.write_text(
            json.dumps(
                {
                    "accuracy": acc,
                    "classification_report": report_dict,
                    "confusion_matrix": cm_df.to_dict(),
                    "inference": inference_block,
                },
                indent=2,
                ensure_ascii=False,
            )
        )


        print("📄 Métricas JSON   :", json_path.resolve())


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True,
                   help=".pkl (TF-IDF) o carpeta Transformer")
    p.add_argument("--data-file",  required=True,
                   help="CSV con columnas text,label")
    p.add_argument("--out",
                   help="Ruta JSON para guardar las métricas (opcional)")
    return p.parse_args()


if __name__ == "__main__":
    main()
