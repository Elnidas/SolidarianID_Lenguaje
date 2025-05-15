# 1. Recomendación de causas solidarias

Para este ejercicio se ha desarrollado un servicio llamado `recommendation`, ubicado en `app/services/`[recommendation.py](app%2Fservices%2Frecommendation.py) . Este servicio puede ser accedido y probado a través de la API disponible en `http://127.0.0.1:8000/docs`. Dentro de esta interfaz, en la sección **recommendation**, es posible interactuar con las funciones expuestas que permiten generar recomendaciones personalizadas.

El sistema está diseñado para, dado un usuario, recomendar las 3 comunidades y 3 causas más afines. Para ello se emplean varios modelos de representación de texto y diferentes combinaciones de parámetros.

A continuación, se muestra un ejemplo de salida del sistema:

![Ejemplo comparacion recomendacion.png](Ejemplo%20comparacion%20recomendacion.png)
### Comparativa entre modelos

Se han evaluado múltiples enfoques de modelado: **TF-IDF**, **FastText**, **BETO (cased y uncased)**, y **maria**. A continuación, se muestra una tabla resumen con las tres recomendaciones principales por modelo, obtenidas por la anterior llamada:

| Modelo           | Comunidades (Título)                                                                           | Comunidades (Descripción)                                                                         | Comunidades (Título+Descripción)                                                                  | Causas (Título)                                                                                                       | Causas (Descripción)                                                                                            | Causas (Título+Descripción)                                                                                           |
| ---------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **TF-IDF**       | Tecnología para el Bien (0.1722)<br>Salud Comunitaria (0.0812)<br>Arte y Cultura (0.0000)      | Salud Comunitaria (0.2690)<br>Tecnología para el Bien (0.1837)<br>Acción Climática Local (0.1106) | Salud Comunitaria (0.2536)<br>Tecnología para el Bien (0.2040)<br>Acción Climática Local (0.0802) | Talleres de Reciclaje (0.1473)<br>Campaña de Vacunación (0.1473)<br>Festival de Arte Local (0.1243)                   | Talleres de Reciclaje (0.2370)<br>Clínica Móvil (0.2070)<br>Paneles Solares Comunitarios (0.1772)               | Talleres de Reciclaje (0.2117)<br>Clínica Móvil (0.1846)<br>Campaña de Vacunación (0.1333)                            |
| **FastText**     | Tecnología para el Bien (0.8411)<br>Arte y Cultura (0.8182)<br>Igualdad y Derechos (0.8156)    | Salud Comunitaria (0.9721)<br>Tecnología para el Bien (0.9450)<br>Igualdad y Derechos (0.9341)    | Salud Comunitaria (0.9735)<br>Tecnología para el Bien (0.9434)<br>Arte y Cultura (0.9430)         | Campaña de Vacunación (0.8850)<br>Capacitación en Derechos Humanos (0.8713)<br>Apoyo a Mujeres Emprendedoras (0.8661) | Apoyo a Mujeres Emprendedoras (0.9400)<br>Capacitación en Derechos Humanos (0.9315)<br>Aulas Digitales (0.9312) | Capacitación en Derechos Humanos (0.9484)<br>Apoyo a Mujeres Emprendedoras (0.9466)<br>Campaña de Vacunación (0.9414) |
| **beto-uncased** | Tecnología para el Bien (0.6409)<br>Salud Comunitaria (0.5673)<br>Igualdad y Derechos (0.5523) | Salud Comunitaria (0.7654)<br>Igualdad y Derechos (0.7211)<br>Arte y Cultura (0.7192)             | Arte y Cultura (0.7464)<br>Acción Climática Local (0.7347)<br>Salud Comunitaria (0.7144)          | Capacitación en Derechos Humanos (0.6224)<br>Apoyo a Mujeres Emprendedoras (0.5948)<br>Murales Comunitarios (0.5833)  | Capacitación en Derechos Humanos (0.7296)<br>Aulas Digitales (0.7189)<br>Talleres de Reciclaje (0.7101)         | Capacitación en Derechos Humanos (0.7464)<br>Talleres de Reciclaje (0.7371)<br>Aulas Digitales (0.7253)               |
| **beto-cased**   | Tecnología para el Bien (0.8494)<br>Salud Comunitaria (0.8429)<br>Igualdad y Derechos (0.8122) | Salud Comunitaria (0.9201)<br>Acción Climática Local (0.8812)<br>Igualdad y Derechos (0.8776)     | Salud Comunitaria (0.9231)<br>Igualdad y Derechos (0.8950)<br>Tecnología para el Bien (0.8909)    | Capacitación en Derechos Humanos (0.8647)<br>Paneles Solares Comunitarios (0.8155)<br>Becas para Estudiantes (0.8107) | Capacitación en Derechos Humanos (0.9146)<br>Aulas Digitales (0.9139)<br>Campaña de Vacunación (0.8966)         | Capacitación en Derechos Humanos (0.9253)<br>Aulas Digitales (0.9060)<br>Biblioteca Móvil (0.8939)                    |
| **maria**        | Tecnología para el Bien (0.9357)<br>Igualdad y Derechos (0.9330)<br>Arte y Cultura (0.9294)    | Salud Comunitaria (0.9674)<br>Arte y Cultura (0.9587)<br>Tecnología para el Bien (0.9587)         | Salud Comunitaria (0.9632)<br>Tecnología para el Bien (0.9585)<br>Arte y Cultura (0.9583)         | Capacitación en Derechos Humanos (0.9362)<br>Telemedicina Rural (0.9265)<br>Apoyo a Mujeres Emprendedoras (0.9249)    | Aulas Digitales (0.9658)<br>Biblioteca Móvil (0.9654)<br>Capacitación en Derechos Humanos (0.9647)              | Capacitación en Derechos Humanos (0.9675)<br>Biblioteca Móvil (0.9670)<br>Aulas Digitales (0.9617)                    |


### Análisis del Caso: Usuario Ana García (`id: 20ed7f97-4ca6-47f8-a7ca-b9bdebd996c3`)

**Nota:** Ana forma parte de la comunidad _Educación para Todos_, por lo tanto, esta no aparece entre las comunidades recomendadas. _Este caso es el de la tabla anterior._

_Para más detalle sobre el conjunto de datos [dataset.markdown](dataset.markdown)_

#### Comunidades

- Todos los modelos identifican de manera consistente “**Salud Comunitaria**” y “**Tecnología para el Bien**” como afines a los intereses expresados por Ana (educación y salud).
    
- Las recomendaciones adicionales varían: algunos modelos priorizan “**Igualdad y Derechos**”, otros “**Arte y Cultura**”.
    
- El modelo `maria` y `FastText` no destacan “Salud Comunitaria” únicamente por el título, y lo mismo sucede con `beto` respecto a “Tecnología para el Bien”, según la estrategia de comparación utilizada.
    

#### Causas

- Las causas consideradas como más relevantes para Ana (por su contexto y descripción) incluyen: “**Biblioteca Móvil**”, “**Becas para Estudiantes**”, “**Clínica Móvil**”, y “**Aulas Digitales**”.
    
- Solo `beto-cased` y `maria` logran recuperar dos de estas causas en sus Top-3.
    
- TF-IDF identifica correctamente “Clínica Móvil” pero introduce elementos ajenos como “Talleres de Reciclaje”.
    
- FastText y `beto-uncased` tienden hacia causas relacionadas con derechos humanos y mujeres, debido a la semántica asociada con “comunidades marginadas” y “desarrollo”.
    

### Conclusiones

- El modelo `maria` presenta las puntuaciones más altas en general (~0.93–0.97), lo cual puede interpretarse como mayor “confianza” en sus recomendaciones.
    
- TF-IDF muestra menor discriminación entre opciones, reflejado en la caída pronunciada de sus puntuaciones (de ~0.17 a 0.08).
    
- `beto-cased` y `maria` son los modelos más robustos, ya que alinean adecuadamente los intereses del perfil con las causas disponibles, respetando tanto el contenido como la intención del usuario.
    
- FastText y `beto-uncased` tienden a recomendaciones temáticamente amplias, mientras que TF-IDF carece de capacidad semántica profunda.


### Recomendación inversa


![Ejemplo recomendacion inversa.png](Ejemplo%20recomendacion%20inversa.png)

Por último, se ha implementado de forma sencilla un método inverso de recomendación. A través del ID de una causa, el sistema puede identificar y recomendar usuarios potencialmente compatibles, utilizando para ello los modelos de representación semántica FastText o TF-IDF. Esta funcionalidad permite, por ejemplo, que los administradores de una comunidad puedan encontrar usuarios afines a una nueva causa y así fomentar su participación.

![Ejemplo recomendacion inversa resultado.png](Ejemplo%20recomendacion%20inversa%20resultado.png)

---

# 2. Sistema de análisis de polaridad subjetiva 

El objetivo de esta segunda tarea es desarrollar un sistema automático que clasifique los textos introducidos por los usuarios al crear causas o comunidades, identificando si contienen contenido ofensivo o no. Este análisis se enmarca dentro del análisis de polaridad subjetiva, evaluando la carga emocional y el carácter potencialmente dañino del lenguaje, mediante modelos de clasificación binaria entrenados sobre corpus anotados en español.
Para cumplir con los requisitos del ejercicio, el sistema ha sido dividido en tres fases:

1. Compilación y preprocesamiento de datasets multifuente.
    
2. Entrenamiento de modelos base (TF-IDF + clasificadores clásicos).
    
3. Fine-tuning y evaluación de modelos modernos basados en Transformers.

## 2.1 Compilación del dataset

En este tipo de tareas, es fundamental realizar una correcta partición de los datos destinados al entrenamiento y validación del modelo, ya que el objetivo es lograr una buena capacidad de generalización y evitar el sobreajuste.

En nuestro caso particular, se ha abordado el problema de detección de mensajes de odio en español. Para ello, se han utilizado los siguientes conjuntos de datos:



### Datasets integrados

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


La recopilación, normalización y preprocesamiento de los datos se ha automatizado mediante el script [prepare_data_v3.py](app%2Futil%2Fprepare_data_v3.py), el cual realiza las siguientes tareas:

- Carga y limpieza de los textos.

- Unificación de etiquetas si estas difieren entre datasets.

- Conversión de las etiquetas a una representación binaria (mensaje de odio / no mensaje de odio), si es necesario.

- Partición del conjunto en train/test usando una proporción de 80/20

- Para más detalle [Ver explicación del script](documentacion.markdown#-documentación-de-utilpreparedatav3py)

### 2.2. Entrenamiento y validación de modelos baseline

Como modelo **baseline**, se ha empleado el enfoque clásico basado en **TF-IDF + clasificadores supervisados**, tal como se recomienda en el enunciado. Para ello, se ha implementado el script [train_tfidf_classifiers_v2.py](app%2Futil%2Ftrain_tfidf_classifiers_v2.py), el cual realiza las siguientes operaciones:

1. **Carga y partición de datos**:
    
    - Lee un CSV con las columnas `text` y `label`.
        
    - Realiza una partición estratificada en conjuntos de entrenamiento y validación (usualmente 90% - 10%).
        
2. **Extracción de características**:
    
    - Se construye un `TfidfVectorizer` que considera unigramas y bigramas, elimina acentos y limita el vocabulario a los 50.000 términos más frecuentes.
        
    - El vectorizador se ajusta **solo con el conjunto de entrenamiento** para evitar fuga de datos.
        
3. **Entrenamiento y evaluación**:
    
    - Se permite elegir entre varios clasificadores: `LinearSVC`, `LogisticRegression`, `MultinomialNB` y `RandomForestClassifier`.
        
    - Se entrena cada modelo y se evalúa sobre el conjunto de validación, reportando:
        
        - Métricas estándar (`precision`, `recall`, `f1-score`) por clase y globales.
            
        - Matriz de confusión textual.
            
4. **Persistencia de modelos**:
    
    - Tanto el vectorizador como los modelos entrenados se guardan en archivos `.pkl` para su reutilización.
        
   

## 2.3 Entrenamiento y validación de modelos Transformers

Para la evaluación con modelos más avanzados, se ha desarrollado un pipeline de fine-tuning con modelos **preentrenados tipo Transformer**, en conformidad con el ejercicio 6 de las prácticas. Esta parte del sistema está implementada en el script [train_transformers_v2.py](app%2Futil%2Ftrain_transformers_v2.py).

### Modelos evaluados

Se han considerado tres modelos disponibles públicamente en Hugging Face:

- **BETO** (`dccuchile/bert-base-spanish-wwm-uncased`)
    
- **MarIA** (`PlanTL-GOB-ES/roberta-base-bne`)
    
- **DistilBETO** (`dccuchile/distilbert-base-spanish-uncased`)
    

Estos modelos utilizan **word embeddings contextuales**, permitiendo que una palabra como “banco” tenga distintas representaciones según el contexto de uso, lo cual mejora significativamente el rendimiento frente a vectores tradicionales como TF-IDF.

### Pipeline de entrenamiento

1. **Carga y tokenización**:
    
    - Los datos se cargan en formato CSV y se convierten en objetos `Dataset` de Hugging Face.
        
    - Se realiza una tokenización truncada a 256 tokens por entrada.
        
2. **Separación estratificada**:
    
    - El conjunto se divide en entrenamiento y validación usando `train_test_split`, garantizando la distribución proporcional de clases.
    - _Haciendo pruebas con y sin esto me ha dado mejores resultados con la distribucin proporcional_
        
3. **Fine-tuning**:
    
    - Se utiliza la clase `Trainer` para gestionar el entrenamiento y evaluación.
        
    - El entrenamiento se realiza durante varias épocas, con evaluación por época (`evaluation_strategy="epoch"`).
        
    - La métrica principal es el **F1-score macro**.
        
4. **Exportación del modelo**:
    
    - El modelo fine-tuneado y su tokenizer se guardan para reutilización en producción.



### Resultado modelos

Los resultados presentados a continuación han sido obtenidos mediante la ejecución del script de evaluación [evaluate_model_v2.py](app%2Futil%2Fevaluate_model_v2.py).  
Cada evaluación genera un archivo `.json` que contiene las métricas clave (accuracy, F1-score, matriz de confusión, entre otras), lo cual permite su consulta y análisis posterior de forma estructurada.


#### beto_test_metrics


| label | precision | recall | f1-score | support |
|-------|-----------|--------|----------|---------|
| 0 | 0.9043 | 0.9512 | 0.9272 | 10949 |
| 1 | 0.8049 | 0.6666 | 0.7292 | 3305 |
| accuracy |  |  | 0.8852 | 14254 |
| macro avg | 0.8546 | 0.8089 | 0.8282 | 14254 |
| weighted avg | 0.8813 | 0.8852 | 0.8813 | 14254 |

| | pred 0 | pred 1 |
|---|-------|-------|
| real 0 | 10415 | 534 |
| real 1 | 1102 | 2203 |

**Tiempo inferencia:** 57.60 s (0.0040 s/ej.)  
**Pico CPU:** 8.3 MB  (RSS final 783.1 MB)  
**Pico GPU:** 1548.4 MB  

---

#### distilbeto_test_metrics


| label | precision | recall | f1-score | support |
|-------|-----------|--------|----------|---------|
| 0 | 0.9236 | 0.9318 | 0.9277 | 10949 |
| 1 | 0.7671 | 0.7446 | 0.7557 | 3305 |
| accuracy |  |  | 0.8884 | 14254 |
| macro avg | 0.8454 | 0.8382 | 0.8417 | 14254 |
| weighted avg | 0.8873 | 0.8884 | 0.8878 | 14254 |

| | pred 0 | pred 1 |
|---|-------|-------|
| real 0 | 10202 | 747 |
| real 1 | 844 | 2461 |

**Tiempo inferencia:** 29.80 s (0.0021 s/ej.)  
**Pico CPU:** 8.3 MB  (RSS final 795.9 MB)  
**Pico GPU:** 1385.9 MB  

---

#### maria_test_metrics


| label | precision | recall | f1-score | support |
|-------|-----------|--------|----------|---------|
| 0 | 0.9195 | 0.9414 | 0.9303 | 10949 |
| 1 | 0.7892 | 0.7271 | 0.7569 | 3305 |
| accuracy |  |  | 0.8917 | 14254 |
| macro avg | 0.8543 | 0.8342 | 0.8436 | 14254 |
| weighted avg | 0.8893 | 0.8917 | 0.8901 | 14254 |

| | pred 0 | pred 1 |
|---|-------|-------|
| real 0 | 10307 | 642 |
| real 1 | 902 | 2403 |

**Tiempo inferencia:** 58.59 s (0.0041 s/ej.)  
**Pico CPU:** 8.3 MB  (RSS final 801.2 MB)  
**Pico GPU:** 1605.4 MB  

---

#### tfidf_linear_svc


| label | precision | recall | f1-score | support |
|-------|-----------|--------|----------|---------|
| 0 | 0.9126 | 0.8930 | 0.9027 | 10949 |
| 1 | 0.6691 | 0.7165 | 0.6920 | 3305 |
| accuracy |  |  | 0.8521 | 14254 |
| macro avg | 0.7908 | 0.8048 | 0.7973 | 14254 |
| weighted avg | 0.8561 | 0.8521 | 0.8538 | 14254 |

| | pred 0 | pred 1 |
|---|-------|-------|
| real 0 | 9778 | 1171 |
| real 1 | 937 | 2368 |

**Tiempo inferencia:** 1.25 s (0.0001 s/ej.)  
**Pico CPU:** 13.1 MB  (RSS final 560.9 MB)  

---


#### tfidf_logreg 


| label | precision | recall | f1-score | support |
|-------|-----------|--------|----------|---------|
| 0 | 0.9183 | 0.8741 | 0.8957 | 10949 |
| 1 | 0.6403 | 0.7422 | 0.6875 | 3305 |
| accuracy |  |  | 0.8436 | 14254 |
| macro avg | 0.7793 | 0.8082 | 0.7916 | 14254 |
| weighted avg | 0.8538 | 0.8436 | 0.8474 | 14254 |

| | pred 0 | pred 1 |
|---|-------|-------|
| real 0 | 9571 | 1378 |
| real 1 | 852 | 2453 |

**Tiempo inferencia:** 1.18 s (0.0001 s/ej.)  
**Pico CPU:** 13.1 MB  (RSS final 560.4 MB)  

---

#### tfidf_multinb 


| label | precision | recall | f1-score | support |
|-------|-----------|--------|----------|---------|
| 0 | 0.8196 | 0.9894 | 0.8965 | 10949 |
| 1 | 0.8881 | 0.2787 | 0.4242 | 3305 |
| accuracy |  |  | 0.8246 | 14254 |
| macro avg | 0.8539 | 0.6340 | 0.6604 | 14254 |
| weighted avg | 0.8355 | 0.8246 | 0.7870 | 14254 |

| | pred 0 | pred 1 |
|---|-------|-------|
| real 0 | 10833 | 116 |
| real 1 | 2384 | 921 |

**Tiempo inferencia:** 1.18 s (0.0001 s/ej.)  
**Pico CPU:** 13.1 MB  (RSS final 557.7 MB)  

---

#### tfidf_rf 


| label | precision | recall | f1-score | support |
|-------|-----------|--------|----------|---------|
| 0 | 0.8484 | 0.9780 | 0.9086 | 10949 |
| 1 | 0.8523 | 0.4209 | 0.5635 | 3305 |
| accuracy |  |  | 0.8488 | 14254 |
| macro avg | 0.8503 | 0.6994 | 0.7360 | 14254 |
| weighted avg | 0.8493 | 0.8488 | 0.8286 | 14254 |

| | pred 0 | pred 1 |
|---|-------|-------|
| real 0 | 10708 | 241 |
| real 1 | 1914 | 1391 |

**Tiempo inferencia:** 1.36 s (0.0001 s/ej.)  
**Pico CPU:** 18.4 MB  (RSS final 1190.1 MB)  



Tras la evaluación cuantitativa de múltiples modelos, se ha observado que MarIA ofrece el mejor compromiso entre rendimiento predictivo y robustez general. Con un F1-macro de 0.8436, supera tanto a BETO (0.8282) como a DistilBETO (0.8417), y también a los modelos clásicos basados en TF-IDF (máximo ≈ 0.7973). Aunque DistilBETO presenta una ligera ventaja en velocidad de inferencia, MarIA compensa con una mayor precisión global y estabilidad en ambas clases.

A partir de este análisis, se ha decidido utilizar MarIA como modelo definitivo para detectar automáticamente texto ofensivo en causas y comunidades creadas por los usuarios. Este modelo será integrado en el sistema de clasificación desplegado en producción.

# 3. Despliegue e integración con la plataforma

Para esta fase, se ha desarrollado un servicio REST utilizando **FastAPI**, el cual expone los modelos entrenados para análisis de texto. La interfaz completa puede consultarse accediendo a la documentación generada automáticamente en:

`http://127.0.0.1:8000/docs/`

Esta documentación incluye todos los endpoints disponibles, sus parámetros, y el formato esperado de entrada, facilitando así su integración desde otros componentes de la plataforma.

Adicionalmente, se han implementado dos integraciones a modo de ejemplo (simuladas como _mockups_):

- **Recomendación de comunidades**: usando el modelo **MarIA**, se realiza una sugerencia de comunidades basadas en la similitud semántica del texto proporcionado (título + descripción).
![ejemplo recomendacion.gif](ejemplo%20recomendacion.gif)    
- **Detección de contenido ofensivo**: se aplica al momento de crear nuevas causas o comunidades, verificando si el texto ingresado es ofensivo según el modelo fine-tuneado.
    
![Ejemplo texto ofensivo.gif](Ejemplo%20texto%20ofensivo.gif)

Ambas funcionalidades están ilustradas mediante gifs para mostrar el flujo esperado en un entorno de usuario simulado.

Como hallazgo adicional durante la evaluación del sistema, se observó que el modelo ofensivo clasifica correctamente entradas sin sentido o _ruido textual_ (e.g. `"asdkjhasiu"`, ` "daiosjdoi"` o `"dauiysbnd"`) como ofensivas. Este comportamiento actúa como una barrera preventiva adicional, impidiendo el ingreso de descripciones irrelevantes o vacías desde el punto de vista semántico.