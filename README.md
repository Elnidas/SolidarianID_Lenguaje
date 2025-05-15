# 📁 Proyecto: Sistema de Clasificación y Recomendación de Texto Ofensivo y Solidario

Este repositorio contiene todos los recursos necesarios para compilar datasets, entrenar modelos, realizar evaluaciones y desplegar servicios REST relacionados con la detección de discurso ofensivo y la recomendación de causas solidarias. (Alguno de dichos datos no se pueden acceder ya que son demasiado pesados y no se han subido, cualquier duda preguntadme)

---

## 📄 Archivos principales

### [documentacion.markdown](documentacion.markdown)
Documentación técnica detallada de cada uno de los scripts usados en el proyecto:
- `prepare_data_v3.py`: unifica y limpia datasets de discurso ofensivo.
- `train_tfidf_classifiers_v2.py`: entrena clasificadores clásicos (TF-IDF + SVM, etc.).
- `train_transformers_v2.py`: fine-tuning de modelos como BETO o MARIA.
- `evaluate_model_v2.py`: evalúa modelos y genera métricas detalladas.
- `offensiveClasifier.py`: módulo de inferencia para clasificar texto ofensivo.
- `recommendation.py`: sistema de recomendación basado en similitud semántica.
- `api_client.py`: utilidades para interactuar con la API de SolidarianID

### [ejercicios.markdown](ejercicios.markdown)
Documento con la redacción y justificación de cada uno de los apartados del ejercicio práctico, enlazando los scripts y modelos empleados.

### [dataset.markdown](dataset.markdown)
Contiene la descripción del dataset de prueba para el sistema de recomendación, con usuarios, comunidades y causas ficticias diseñadas para validar el comportamiento del sistema.

---

