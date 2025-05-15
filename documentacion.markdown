## 📚 Índice

1. [Compilación del dataset (prepare_data_v3.py)](#-documentación-de-utilpreparedatav3py)
2. [Entrenamiento y evaluación con TF-IDF (train_tfidf_classifiers_v2.py)](#-documentación-de-utiltraintfidfclassifiersv2py)
3. [Fine-tuning con Transformers (train_transformers_v2.py)](#-documentación-de-utiltraintransformersv2py)
4. [Evaluación de modelos (evaluate_model_v2.py)](#-documentación-de-utilevaluatemodelv2py)
5. [Clasificador para la API (offensiveClasifier.py)](#-documentación-de-servicesoffensiveclasifierpy)
6. [Sistema de recomendación (recommendation.py)](#-documentación-de-servicesrecommendationpy)
7. [Cliente de la API (api_client.py)](#-documentación-servicesapiclientpy)





# 📄 Documentación de `util/prepare_data_v3.py`

> _“Fusiona todos los corpus de discurso ofensivo/odio en español…, los limpia y genera **un único** particionado aleatorio”_ —Docstring original del script.

---

## Objetivo del script

`prepare_data_v3.py` automatiza **cuatro** tareas habituales en proyectos de PLN:

1. **Carga heterogénea:** leer múltiples corpus con formatos y etiquetas distintas.
    
2. **Normalización:** unificar columnas (`text`, `label`) y convertir las etiquetas a binario (0 = no ofensivo, 1 = ofensivo).
    
3. **Limpieza ligera:** eliminar menciones, URLs y espacios extra para reducir ruido sin alterar el lenguaje natural.
    
4. **Particionado reproducible:** generar `final_train.csv`y `final_test.csv` con _stratified split_ para preservar la distribución de clases.
    

El resultado es un dataset listo para ser usado en _pipelines_ de modelado (e.g. fine‑tuning de BERT, entrenamiento clásico con TF‑IDF + SVM, etc.).

---

## Datasets integrados

| Alias interno        |Fichero| Columnas originales | Mapeo → binario                        | Notas                                                 |
|----------------------|---|---------------------|----------------------------------------|-------------------------------------------------------|
| **OffendES** (train) |`training_set.tsv`| `comment`, `label`  | `{OFP, OFG → 1; NOE, NO → 0}`          | Corpus de ofensas y no‑ofensas en redes.              |
| **OffendES** (dev)   |`dev_set.tsv`| idem                | idem                                   | Se combina para aumentar tamaño.                      |
| **OffendES** (test)  |`test_set.tsv`| idem                | idem                                   | Se ignora condición de _held‑out_ para fusionar todo. |
| **HateNet**          |`hatenet.csv`| `tweet`, `label`    | `{hateful → 1; non_hateful → 0}`       | Tweets anotados manualmente.                          |
| **Full Dataset 1**   |`full_dataset_1.csv`| `tweet`, `label`    | `{misogyny → 1; non‑misogyny → 0}`     | Especializado en misoginia.                           |
| **AMI**              |`Ami.csv`| `tweet`, `label`    | `{misogynous → 1; non_misogynous → 0}` | Variante del anterior.                                |
| **HateEval (ES)**    |`hateeval.csv`| `tweet`, `label`    | `{hatespeech → 1; non_hatespeech → 0}` | Subconjunto español del SemEval 2019.                 |
| **Superset**         |`spanish_hate_speech_superset_train.csv`| `text`, `labels`    | _ya 0/1_                               | Conjunto masivo con varias fuentes.                   |




---

Ubicaciones de los datasets y de los csv combinados


```python
DATA_DIR = Path("../../data")
OUT_DIR  = Path("../../data/final")
```



### Expresiones regulares de limpieza

```python
USER_RE       = re.compile(r"@\w+")
URL_RE        = re.compile(r"https?://\S+|www\.\S+")
MULTISPACE_RE = re.compile(r"\s+")
```

- Se eliminan menciones, urls y espacios en blanco extra que puedan quedar
    


### Función `clean_text()`

```python
def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = USER_RE.sub("", str(text))
    text = URL_RE.sub("", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()
```

1. **`pd.isna`** evita errores con nulos.
    
2. Conversión explícita a `str` — hay datasets con enteros o floats.
    
3. Sustituciones vacías conservan longitud del texto ≈ original (importante para modelos que usan posición).
    
4. `strip()` elimina espacios residuales.
    

### Función `load_dataset()`

Responsable de **homogeneizar** cada archivo:

1. Leer CSV/TSV con separador configurable (`sep`).
    
2. Renombrar columnas a `text`, `orig_label`.
    
3. Mapear etiquetas si `label_map` ≠ `None`.
    
4. Convertir a `int` y filtrar filas vacías.
    

```python
df["label"] = df["orig_label"].map(label_map)  # ó copia directa
```


### Estructura `DATASETS`

Una **lista de diccionarios** describe cada corpus. Ejemplo:

```python
{
  "path": DATA_DIR / "hatenet.csv",
  "text_col": "tweet",
  "label_col": "label",
  "map": {"hateful": 1, "non_hateful": 0},
}
```

Ventajas:

- Evita código duplicado
    
- Facilita añadir/quitar corpus con una sola línea.
    

### Función `stats()`

Calcula conteo y porcentaje por clase para verificar balance 

```python
def stats(df: pd.DataFrame) -> str:
    counts = df["label"].value_counts().sort_index()
    total = len(df)
    pct = (counts / total * 100).round(2)
    return (
        f"No ofensivo(0): {counts.get(0,0):,}  — {pct.get(0,0.0)}%\n"
        f"Ofensivo  (1): {counts.get(1,0):,}  — {pct.get(1,0.0)}%\n"
        f"Total          : {total:,}"
    )
```

### Función `main()`

|Paso|Código clave| Explicación                                   |
|---|---|-----------------------------------------------|
|1|`frames = [load_dataset(cfg) for cfg in DATASETS]`| Carga los datasets                            |
|2|`full_df.drop_duplicates(subset="text")`| Eliminar duplicados idénticos                 |
|3|`train_test_split(... stratify=full_df["label"])`| Estratificación asegura proporciones iguales. |
|4|`to_csv(train_path, index=False)`| Persistencia en disco                         |
|5|`print(stats(...))`| Feedback inmediato sobre el csv generado      |

### Ejecución desde línea de comandos

```bash
python prepare_data_v3.py \
  --test_size 0.25 \
  --out_dir ../../data/final_v2
```

Parámetros:

- `--test_size` (float): proporción para test (por defecto 0.2).
    
- `--out_dir` (str): carpeta de salida.
    

---


# 📄 Documentación de `util/train_tfidf_classifiers_v2.py`

> _«Carga un CSV con columnas text y label, divide en train/dev, extrae TF-IDF, entrena varios clasificadores, muestra métricas y guarda los modelos en .pkl»_

---

## Objetivo del script


1. **Carga** de un CSV etiquetado (`text`, `label`).
2. **Particionado reproducible** en entrenamiento y validación (`train_test_split`, `random_state`).
3. **Extracción de características** con **TF-IDF** de uni- y bi-gramas.
4. **Entrenamiento y evaluación** de hasta **cuatro** modelos  (SVM lineal, Regresión Logística, Naïve Bayes, Random Forest) con `classification_report` y matriz de confusión.
5. **Serialización** (“checkpointing”) de vectorizador + clasificador en un único archivo `.pkl`, listo para reutilizarse en inferencia o en un pipeline posterior.

---

## Modelos integrados

| Alias CLI        | Clase de scikit-learn                  | 
|------------------|----------------------------------------|
| `linear_svc`     | `sklearn.svm.LinearSVC`                |  
| `logreg`         | `sklearn.linear_model.LogisticRegression` | 
| `multinomial_nb` | `sklearn.naive_bayes.MultinomialNB`    | 
| `random_forest`  | `sklearn.ensemble.RandomForestClassifier` | 
*(Se invocan con `--models linear_svc logreg ...`)*

---

## Directorios de trabajo

El script **no** fija rutas absolutas; las recibe por CLI:

* `--csv` → dataset de entrada.  
* `--outdir` → carpeta donde se guardarán los `.pkl` generados (se crea si no existe).

---

## Secciones del código

A continuación se presenta cada bloque **completo**, seguido de una explicación detallada.

### `get_available_models()`

```python
def get_available_models() -> dict[str, object]:
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
```   

- Devuelve un diccionario “nombre → instancia” para facilitar su recorrido en el loop principal.

- Los modelos incorporan class_weight="balanced" (o equivalente) para compensar clases desbalanceadas.

- El bosque usa n_estimators=300 y semilla fija (random_state=0) para reproducibilidad.

### build_vectorizer()

```python
def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents="unicode",
    )
```

- Limita el vocabulario a 50 000 términos más frecuentes → controla la dimensionalidad.

- Usa uni- y bi-gramas (ngram_range=(1,2)) para capturar contextos cortos comunes en discurso ofensivo.

- strip_accents='unicode' normaliza acentos.

- lowercase=True reduce variantes en mayúsculas.

### print_conf_matrix()

```python
def print_conf_matrix(cm: np.ndarray) -> None:
    print("\nMatriz de confusión")
    print("            pred 0   pred 1")
    print(f"real 0    {cm[0,0]:7d} {cm[0,1]:7d}")
    print(f"real 1    {cm[1,0]:7d} {cm[1,1]:7d}")

```
-  Imprime la matriz de confusión en formato tabla.

### Flujo principal (main())

```python

def main() -> None:
    args = get_args()
    np.random.seed(args.seed)

    # 4.1 Carga de datos
    df = pd.read_csv(args.csv)
    X, y = df["text"], df["label"]

    # 4.2 Split estratificado
    X_train, X_dev, y_train, y_dev = train_test_split(
        X, y,
        test_size=args.dev_ratio,
        stratify=y,
        random_state=args.seed
    )

    # 4.3 Vectorización
    vect = build_vectorizer()
    X_train_tfidf = vect.fit_transform(X_train)
    X_dev_tfidf   = vect.transform(X_dev)

    # 4.4 Entrenamiento + evaluación en bucle
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    models_available = get_available_models()

    for name in args.models:
        if name not in models_available:
            raise ValueError(f"Modelo '{name}' no reconocido…")

        clf = models_available[name]
        clf.fit(X_train_tfidf, y_train)

        # Evaluación
        y_pred = clf.predict(X_dev_tfidf)
        print(classification_report(y_dev, y_pred, digits=4))
        cm = confusion_matrix(y_dev, y_pred)
        print_conf_matrix(cm)

        # Persistencia
        fname = outdir / f"tfidf_{name}.pkl"
        joblib.dump((vect, clf), fname)
        print("Modelo guardado en", fname.resolve())

```
| Paso | Código clave                            | Explicación                                                                                 |
| ---- | --------------------------------------- | ------------------------------------------------------------------------------------------- |
| 4.1  | `pd.read_csv`                           | Lee el dataset completo en memoria.                                                         |
| 4.2  | `train_test_split(... stratify=y)`      | Divide manteniendo la proporción original de clases (impide sesgos).                        |
| 4.3  | `vect.fit_transform` / `vect.transform` | **Fittea** el vectorizador **solo** en train; dev se transforma para evitar *data leakage*. |
| 4.4  | Bucle `for name in args.models`         | Permite entrenar **varios** modelos en una sola ejecución (p.ej. pruebas rápidas).          |
| 4.5  | `classification_report` + matriz        | Métricas detalladas (precision, recall, F1) y errores crudos.                               |
| 4.6  | `joblib.dump((vect, clf), fname)`       | Guarda **vectorizador + clasificador** juntos para inferencia inmediata (`joblib.load`).    |

---

# 📄 Documentación de `util/train_transformers_v2.py`

> _"Fine-tune de modelos tipo BERT en español usando HuggingFace Transformers + evaluación + guardado de resultados."_

---

## Objetivo del script

Este script permite el fine-tuning de modelos Transformer preentrenados en español (como BETO o Maria) para tareas de clasificación binaria (texto ofensivo vs. no ofensivo), aplicando buenas prácticas como validación estratificada, tokenización por lotes, y evaluación con F1-macro.

---

## Modelos disponibles

### Alias → ID del modelo (HuggingFace)

```python
MODEL_LIST = {
    "beto":       "dccuchile/bert-base-spanish-wwm-uncased",
    "maria":      "PlanTL-GOB-ES/roberta-base-bne",
    "distilbeto": "dccuchile/distilbert-base-spanish-uncased",
}

```
- Los modelos se seleccionan con el parámetro --model y se traducen internamente a su identificador en el hub de HuggingFace.

### Función load_dataset()

```python

def load_dataset(csv_path: str) -> Dataset:
    return Dataset.from_pandas(pd.read_csv(csv_path))

```

- Carga un CSV (con columnas text y label) como objeto Dataset de HuggingFace para procesamiento eficiente.

### Función tokenize_function()

```python

def tokenize_function(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, max_length=256)
```

- Aplica tokenización a cada texto del batch, truncando a 256 tokens (máximo típico para tareas rápidas).


### Función compute_metrics()

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {"f1": f1_score(labels, preds, average="macro", zero_division=0)}

```
- Calcula el F1 macro (equilibrado entre clases), evitando divisiones por cero en clases poco representadas.

---

## Entrenamiento
### Función main()

```python
def main() -> None:
    args = get_args()
    set_seed(args.seed)

    model_name = MODEL_LIST[args.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Modelo:", args.model, "→", model_name)
    print("Dispositivo:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU")

    # 1) Carga y partición del dataset
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

    # 2) Tokenización
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = train_ds.map(lambda b: tokenize_function(b, tokenizer), batched=True)
    dev_ds   = dev_ds.map(lambda b: tokenize_function(b, tokenizer),   batched=True)

    # 3) Cargar modelo preentrenado
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    # 4) Configuración de entrenamiento
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

    # Guardar modelo y tokenizer fine-tuneados
    final_dir = Path(args.outdir) / "final_model"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("✅ Modelo y tokenizer guardados en", final_dir.resolve())

```


## Tabla de análisis por secciones

| Paso | Código / Acción | Descripción |
|------|------------------|-------------|
| 1. | `args = get_args()` | Lee argumentos desde CLI como `--model`, `--epochs`, etc. |
| 2. | `set_seed(args.seed)` | Establece semilla global para reproducibilidad (datos, pesos, splits). |
| 3. | `model_name = MODEL_LIST[args.model]` | Traduce el alias a su ID real en HuggingFace Hub. |
| 4. | `device = "cuda" if torch.cuda.is_available() else "cpu"` | Detecta si hay GPU disponible para acelerar el entrenamiento. |
| 5. | `print(...)` | Informa al usuario qué modelo y dispositivo se usarán. |

---

### Carga y partición del dataset

| Paso | Código / Acción | Descripción |
|------|------------------|-------------|
| 6. | `full_ds = load_dataset(args.train_file)` | Carga CSV en un objeto `datasets.Dataset`. |
| 7. | `idx = np.arange(len(full_ds))` | Genera índices explícitos para hacer split. |
| 8. | `train_test_split(..., stratify=full_ds["label"])` | Divide el dataset manteniendo la proporción de clases. |
| 9. | `train_ds = full_ds.select(train_idx)` | Extrae subconjunto de entrenamiento. |
| 10. | `dev_ds = full_ds.select(dev_idx)` | Extrae subconjunto de validación. |

---

### Tokenización de texto

| Paso | Código / Acción | Descripción |
|------|------------------|-------------|
| 11. | `tokenizer = AutoTokenizer.from_pretrained(model_name)` | Descarga el tokenizer del modelo HuggingFace. |
| 12. | `train_ds = train_ds.map(...)` | Aplica tokenización en batch sobre `train`. |
| 13. | `dev_ds = dev_ds.map(...)` | Aplica tokenización en batch sobre `dev`. |

---

### Cargar modelo preentrenado

| Paso | Código / Acción | Descripción |
|------|------------------|-------------|
| 14. | `model = AutoModelForSequenceClassification.from_pretrained(...)` | Carga el modelo con una capa de clasificación binaria (`num_labels=2`). |
| 15. | `.to(device)` | Envía el modelo a GPU o CPU según disponibilidad. |

---

###  Configuración del entrenamiento

| Paso | Código / Acción | Descripción |
|------|------------------|-------------|
| 16. | `training_args = TrainingArguments(...)` | Define hiperparámetros: épocas, tamaño de batch, logging, etc. |
| 17. | `Trainer(...)` | Instancia el `Trainer` con modelo, args, datasets y métricas. |

---

### Entrenamiento

| Paso | Código / Acción | Descripción |
|------|------------------|-------------|
| 18. | `trainer.train()` | Ejecuta el ciclo de entrenamiento con evaluación periódica. |

---

### Guardado del modelo

| Paso | Código / Acción | Descripción |
|------|------------------|-------------|
| 19. | `final_dir = Path(args.outdir) / "final_model"` | Define ruta de guardado. |
| 20. | `trainer.save_model(final_dir)` | Guarda los pesos del modelo fine-tuneado. |
| 21. | `tokenizer.save_pretrained(final_dir)` | Guarda el tokenizer para futura inferencia. |
| 22. | `print(...)` | Confirma ubicación de los archivos resultantes. |

---

## Resumen general

| Componente | Descripción |
|------------|-------------|
| Tokenizador | Se adapta al modelo (`AutoTokenizer`) y se guarda junto al modelo final. |
| Dataset | Se carga con `datasets.Dataset` y se divide de forma estratificada. |
| Modelo | Basado en BERT o variantes, adaptado a clasificación binaria. |
| Entrenamiento | Se realiza con `Trainer`, incluyendo evaluación y métricas. |
| Métrica | Se evalúa con F1-macro para contemplar desbalance de clases. |
| Guardado | Todo se almacena en un subdirectorio `final_model` para despliegue. |

---
# 📄 Documentación de `util/evaluate_model_v2.py`

> _"Evalúa modelos entrenados (TF-IDF o Transformers) sobre un conjunto de test, y genera métricas detalladas."_

---

## Objetivo del script

Este script permite evaluar modelos de clasificación binaria sobre un archivo CSV con datos de prueba (`text`, `label`) y generar métricas como:

- Accuracy
- Precision, Recall, F1 por clase
- F1-macro
- Matriz de confusión
- Reporte `classification_report` (scikit-learn)
- Resultados exportables en `.json` y `.txt`

Admite tanto:

- Modelos TF-IDF (`.pkl`)
- Modelos Transformers fine-tuneados (carpetas con `config.json`)

---

## Funciones auxiliares
### load_tfidf()

```python
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

```
- Carga un archivo .pkl que contiene:

  - Un pipeline completo Pipeline(...)

  - Una tupla (vectorizer, classifier)

- Devuelve una función predict(texts).

### def load_transformer(dir_path: Path):

```python
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
```

- Carga un modelo fine-tuneado en una carpeta con config.json.

- Usa pipeline() de HuggingFace con device=0 (GPU si está disponible).

- Devuelve una función predict(texts) que transforma labels tipo "LABEL_1" → 1.

### def load_model(path: str):

```python
def load_model(path: str):
    path = Path(path)
    if path.is_file() and path.suffix == ".pkl":
        return load_tfidf(path)
    elif path.is_dir() and (path / "config.json").exists():
        return load_transformer(path)
    else:
        raise ValueError("Ruta no reconocida: debe ser .pkl o carpeta Transformer")

```
- Decide qué loader usar en función de la extensión o archivos presentes:

  - .pkl → load_tfidf()

  - Carpeta con config.json → load_transformer()
---
## Evaluación: Función `main()`

```python
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


```

### Flujo de trabajo paso a paso:

| Paso | Acción | Detalle |
|------|--------|---------|
| 1. | `load_model()` | Carga y envuelve el modelo en una función `predict`. |
| 2. | `pd.read_csv()` | Carga archivo de test con columnas `text`, `label`. |
| 3. | `predict(texts)` | Ejecuta inferencia sobre todo el corpus. |
| 4. | Métricas | Calcula: accuracy, F1 por clase, F1 macro, matriz de confusión, `classification_report`. |
| 5. | Exportar | Guarda los resultados en `.json` estructurado y `.txt` legible si se especifica `--out`. |

---

## Métricas generadas

### Métricas impresas en consola:

```
🔹 Modelo cargado (TRANSFORMER): ../../models/transformer_colab_beto/final_model

Accuracy  : 0.8852

              precision    recall  f1-score   support

           0     0.9043    0.9512    0.9272     10949
           1     0.8049    0.6666    0.7292      3305

    accuracy                         0.8852     14254
   macro avg     0.8546    0.8089    0.8282     14254
weighted avg     0.8813    0.8852    0.8813     14254


Matriz de confusión
        pred 0  pred 1
real 0   10415     534
real 1    1102    2203

Tiempo inferencia : 57.60 s  (0.0040 s por muestra)
Pico memoria CPU  : 8.3 MB  (RSS final 783.1 MB)
Pico memoria GPU  : 1548.4 MB

```

---

## Consideraciones

- El script se adapta automáticamente al tipo de modelo.
- Soporta evaluación en CPU o GPU.
- Acepta modelos fine-tuneados locales, sin necesidad de conexión a Internet.
- Puede ser usado como paso final luego de `train_tfidf_classifiers_v2.py` o `train_transformers_v2.py`.


---


# 📄 Documentación de `services/offensiveClasifier.py`

A continuación se describe **línea por línea** el código que implementa un clasificador binario de texto ofensivo.  
Se explica lo que hace cada instrucción y, cuando procede, el significado de los argumentos que recibe.

> **Nota:** Las líneas vacías se omiten porque sólo sirven para mejorar la legibilidad.

---

## Código

```python
BASE_DIR  = Path(__file__).resolve().parents[2] 
MODEL_DIR = (BASE_DIR / "models" / "transformer_colab_distilbeto" / "final_model").resolve()


if not MODEL_DIR.is_dir():
    raise FileNotFoundError(f"⚠️  No encuentro el modelo en {MODEL_DIR}")


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
```


1. `_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)`  
    * Llama a **`AutoTokenizer.from_pretrained`** con un único argumento posicional:  
      * `MODEL_DIR`— ruta local al directorio que contiene los archivos del tokenizer (`tokenizer.json`, `vocab.txt`, etc.).  
    * Devuelve un objeto tokenizer configurado y lo guarda en la variable de módulo `_tokenizer`.

2. `_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)`  
    * Llama a **`AutoModelForSequenceClassification.from_pretrained`** con el mismo argumento (`MODEL_DIR`).  
    * Carga los pesos (`pytorch_model.bin`) y la configuración del modelo, devolviendo una instancia lista para inferencia.  
    * Se guarda en `_model`.

3. `_model.eval().to("cuda" if torch.cuda.is_available() else "cpu")`  
    * `eval()` pone el modelo en modo **evaluación** (desactiva *dropout* y *batch‑norm*).  
    * `to(device)` mueve los pesos al dispositivo adecuado:  
      * `"cuda"` si hay una GPU disponible.  
      * `"cpu"` en caso contrario.

4. `def classify(text: str) -> Tuple[bool, str]:`  
    Declara la función **`classify`**.  
    * **Parámetros**  
      * `text: str` — Cadena con el texto que se quiere clasificar.  
    * **Valor de retorno** (`Tuple[bool, str]`)  
      * Primero: `True` si el texto es ofensivo, `False` si no lo es.  
      * Segundo: etiqueta legible `'ofensivo'` / `'no_ofensivo'`.

5. `device = _model.device`  
    Recupera el dispositivo (`cpu` o `cuda`)
6. `inputs = _tokenizer(text, return_tensors="pt", truncation=True).to(device)`  
    * Tokeniza `text`; los argumentos más relevantes son:  
      * `return_tensors="pt"` — Indica que la salida debe ser tensores *PyTorch*.  
      * `truncation=True` — Corta el texto si supera la longitud máxima admitida por el modelo.  
    * El resultado es un diccionario de tensores (`input_ids`, `attention_mask`, …).  
    * `.to(device)` mueve esos tensores al mismo dispositivo del modelo.

7. 
```python
with torch.no_grad():
    logits = _model(**inputs).logits
```  
* El bloque `torch.no_grad()` PyTorch deja de seguir la traza de operaciones. Para acelerar la inferencia y reducir memoria.  
* Se hace una pasada adelante del modelo con los tensores de `inputs`; el resultado contiene los **`logits`**, es decir, las puntuaciones sin normalizar para cada clase.

8. `pred = int(torch.argmax(logits, dim=-1).item())   # 0 ó 1`  
    * `torch.argmax(logits, dim=-1)` obtiene el índice (0 o 1) de la clase con mayor logit.  
    * `.item()` lo convierte a *Python scalar* y luego a `int`.  
    * Se guarda en `pred`.

9. `return pred == 1, ("ofensivo" if pred == 1 else "no_ofensivo")`  
    Devuelve una tupla:  
    * El primer elemento es `True` si `pred == 1`, es decir, si la clase predicha es la ofensiva.  
    * El segundo elemento es la etiqueta en español correspondiente.

---

```python
>>> from offensiveClasifier import classify
>>> classify("¡Eres un fenómeno!")
(False, 'no_ofensivo')
```

---

# 📄 Documentación de `services/recommendation.py`

Este script (`recommendation.py`) implementa un conjunto de **módulos de recomendación** para una plataforma de causas y comunidades solidarias. A grandes rasgos, realiza lo siguiente:  
1. **Carga de modelos de lenguaje**: FastText pre-entrenado y varios Transformers en español (BETO, RoBERTa-BNE).  
2. **Preparación de datos**: obtiene información de usuario, causas y comunidades vía API.  
3. **Preprocesado y filtrado**: valida biografía, construye textos de usuario, descarta items sin descripción o ya unidos.  
4. **Cálculo de similitud**: usa TF-IDF, FastText o Sentence-Transformers para medir la afinidad semántica.  
5. **Funciones de recomendación**: expone pipelines asíncronos que devuelven los _top-N_ causas y comunidades, o comparan distintos modelos. :

---

## 1. Carga de modelos de lenguaje

1. **FastText**  
   - fasttext_model = fasttext.load_model("embeddings-s-model.bin")

     - Se carga un modelo de FastText en español que convierte oraciones en vectores densos promedio de sus palabras.

2. **Sentence-Transformers**
    
    - Se define un diccionario `paths` con tres identificadores de Hugging Face:
        
        - `beto-uncased`, `beto-cased` (BERT en español)
            
        - `maria` (RoBERTa-BNE)
            
    - Para cada ruta:
        
        1. Se crea un **tokenizer** de `transformers.AutoTokenizer`.
            
        2. Se carga el **encoder** base con `transformers.AutoModel`.
            
        3. Con `sentence_transformers.models.Transformer` y `Pooling` se monta un `SentenceTransformer` que:
            
            - Tokeniza y pasa el texto por la red Transformer.
                
            - Agrega los embeddings de tokens en un vector de oración (pooling por media).
                
    - Resultado: un diccionario `sentence_transformers_models` con instancias listas para inferencia en CPU o GPU.
        

---

## 2. Obtención y preprocesado de datos

### 2.1 _Obtención desde la API_

Función asíncrona `_fetch_user_and_data(user_id)` que realiza cuatro peticiones REST usando `fetch_data`:

- Perfil de usuario (`User`)
    
- Lista completa de causas (`Cause`)
    
- Lista completa de comunidades (`Community`)
    
- Comunidades en las que el usuario ya participa (`CommunityJoined`)
    

Si falta información esencial lanza `ValueError`.

### 2.2 Validación de biografía

`_validate_bio(user)` comprueba que `user.bio` existe y no sea solo espacios. Impide hacer recomendaciones sin texto de referencia.

### 2.3 Construcción del texto de usuario

`_build_user_text(user, joined)` concatena la biografía con los nombres de las comunidades de las que ya forma parte. Este texto se usará como “consulta” para comparar semánticamente con las descripciones de causas o comunidades.

### 2.4 Embeddings mixtos 

`_build_user_embedding_separate(...)`

- Calcula embedding de la bio y de cada nombre de comunidad por separado.
    
- Promedia las matrices y mezcla ambos vectores con peso `alpha` (por defecto 0.7 para bio / 0.3 para comunidades).

    

---

## 3. Filtrado de items

La función `_prepare_filters(causes, communities, joined_ids)` devuelve varios subconjuntos:

- **Comunidades no unidas** con descripción no vacía
    
- **Comunidades activas** no unidas
    
- **Causas verificadas** con descripción
    
- **Todas las causas** con descripción
    
- Y versiones que filtran por título en lugar de descripción
    

Este filtrado previene recomendar items obsoletos, sin contenido o pertenecientes ya al usuario

---

## 4. Cálculo de similitud y ranking

El script ofrece cuatro métodos internos que devuelven el “top-N” más parecidos tupla `{ item, score }`:

1. **TF-IDF + coseno**
    
    - Construye un corpus con `[texto_usuario, descripciones…]`.
        
    - Aplica `TfidfVectorizer().fit_transform(corpus)`.
        
    - Mide `cosine_similarity(vector_usuario, vectores_items)`.
        
    - Ordena por puntuación y extrae los mejores.
        
2. **FastText + coseno**
    
    - Convierte `texto_usuario` e `items` en vectores promedio de FastText.
        
    - Aplica `cosine_similarity` y top-N.
        
3. **Sentence-Transformers** (`_get_top_n_by_model`)
    
    - Codifica oraciones completas con el modelo escogido (BETO/RoBERTa).
        
    - Mide la similitud de coseno de embeddings.
        
4. **Embeddings pre-combinados** (`_get_top_n_by_user_emb`)
    
    - Recibe `user_emb` ya mezclado con comunidades.
        
    - Compara contra embeddings de items.
        

Cada uno maneja internamente la eliminación de textos vacíos y retorna una lista de diccionarios con el item (reducido o completo) y la puntuación de similitud.

---

## 5. Funciones de recomendación expuestas

A partir de los bloques anteriores se definen varias interfaces asíncronas:

- `recommend_causes_tfidf(user_id, reduced_output)`
    
- `recommend_causes_fasttext(user_id, reduced_output)`
    
- `recommend_causes_any_model_extended(user_id, model_key, top_n, reduced_output)`
    
- `compare_models(user_id, reduced_output)`
    

Estas funciones orquestan:

1. **_fetch_user_and_data** → datos base
    
2. **_validate_bio** → asegurarse de que haya texto
    
3. **_build_user_text** / `_build_user_embedding_separate` → preparar consulta
    
4. **_prepare_filters** → filtrar items
    
5. **_get_top_n_by_…** → ranking con el método elegido
    

`compare_models` combina en un solo resultado TF-IDF y FastText para comparar resultados de ambos.

---

## 6. Flujo de una petición

1. El cliente invoca, por ejemplo, `await recommend_causes_tfidf("user-123")`.
    
2. Se obtienen y validan datos del usuario y catálogo de causas/comunidades.
    
3. Se construye el texto de usuario y los filtros aplicables.
    
4. Se calcula similitud TF-IDF con cada lista filtrada.
    
5. Se devuelve un diccionario JSON con varias llaves:
    
    - `user_bio`: el texto usado como consulta.
        
    - `top_communities_by_description`, etc., con los `n` items mejor valorados.
        

Para FastText o Sentence-Transformers el flujo es análogo, cambiando solo la forma de generar y comparar embeddings.


---


# 📄 Documentación `services/api_client.py`

- **Importaciones y constantes**
    
    - Se importa `httpx`, una librería asíncrona para realizar peticiones HTTP.
        
    - Se importa `API_BASE_URL` desde `app.config`, que define la URL base de la API a la que se llamará.
        
- **Variable global `TOKEN`**
    
    - Se declara `TOKEN = None` al principio del módulo.
        
    - Esta variable se utilizará para guardar el token JWT que devuelve el servidor después del inicio de sesión, de modo que pueda reutilizarse en llamadas posteriores.
        
## Función asíncrona `login(email, password)`
Nos sirve para hacer login en la aplicación y posteriormente poder hacer las peticiones pertinentes. Idealmente, deberíamos de generar un token sin expiración y guardarlo en el .env. **_Ya no se necesita esto puesto que se ha proporcionado un token sin caducidad_**
```python
async def login(email: str, password: str):
    """
    Realiza el login en la API y almacena el token de autenticación.
    """
    global TOKEN
    async with httpx.AsyncClient() as client:

        response = await client.post(f"{API_BASE_URL}/auth/login", json={"email": email, "password": password})
        response.raise_for_status()
        TOKEN = response.json().get("token")
        print(TOKEN)
```
    
1. Se marca con `async` para poder usarse en un _event loop_.
        
2. Crea un cliente HTTP asíncrono con `httpx.AsyncClient()`.
        
3. Envía una petición `POST` a `"{API_BASE_URL}/auth/login"` con las credenciales en JSON.
        
4. Si la respuesta contiene un error HTTP, `response.raise_for_status()` lanza una excepción.
        
5. Recupera el token del _payload_ (`response.json().get("token")`) y lo guarda en la variable global `TOKEN`.
        
6. Imprime el token por consola (útil en desarrollo, aunque en producción convendría eliminarlo por seguridad).




## Función asíncrona `fetch_data(endpoint, params=None)`

Nos sirve para realizar las distintas peticiones al servidor backend de la aplicación

````python
async def fetch_data(endpoint: str,params: dict = None):
    """
    Hace una petición GET al servidor REST local, incluyendo el token si existe.
    """
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}/{endpoint}", headers=headers,params=params)
        response.raise_for_status()
        return response.json()

````
    
1. Construye el encabezado `Authorization` con el valor de `TOKEN` si éste existe; de lo contrario envía cabeceras vacías.
        
2. Abre de nuevo un `httpx.AsyncClient()` y realiza una petición `GET` a `"{API_BASE_URL}/{endpoint}"`, pasando los parámetros opcionales y las cabeceras.
        
3. Lanza una excepción en caso de error HTTP y, si todo va bien, devuelve el cuerpo de la respuesta ya deserializado (`response.json()`).
        
