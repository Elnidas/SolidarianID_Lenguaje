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

Esta tarea tiene como objetivo construir un sistema capaz de clasificar el contenido subjetivo de los textos introducidos por el usuario a la hora de crear causas y comunidades, para detectar si contienen texto ofensivo o no

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

Carga y limpieza de los textos.

Unificación de etiquetas si estas difieren entre datasets.

Conversión de las etiquetas a una representación binaria (mensaje de odio / no mensaje de odio), si es necesario.

Partición del conjunto en train/test usando una proporción de 80/20

Para más detalle [Ver explicación del script](documentacion.markdown#-documentación-de-utilpreparedatav3py)

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
        

Este enfoque cumple plenamente con los requisitos indicados en el enunciado para la construcción de un modelo baseline.

## 2.3 Entrenamiento y validación de modelos Transformers (2 puntos)

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
