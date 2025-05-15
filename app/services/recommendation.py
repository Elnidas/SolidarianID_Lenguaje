from typing import Dict, Tuple, List, Set, Any, Callable

import transformers
from sentence_transformers import SentenceTransformer
from torch import nn
import torch
from sentence_transformers import models as STmodels
from app.models.models import Cause, Community, Support, CommunityJoined
from app.models.user import User
from app.services.api_client import fetch_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fasttext
import numpy as np

# Cargar modelo FastText preentrenado
fasttext_model = fasttext.load_model("embeddings-s-model.bin")

##############################################
##############################################
##############################################

# 1) Diccionario con los modelos
paths = {
    "beto-uncased": "dccuchile/bert-base-spanish-wwm-uncased",
    "beto-cased": "dccuchile/bert-base-spanish-wwm-cased",
    "maria": "PlanTL-GOB-ES/roberta-base-bne",
    # Se pueden añadir más:
    # "mbert": "bert-base-multilingual-cased",
    # ...
}

# 2) Crear un diccionario de tokenizers
tokenizers = {
    key: transformers.AutoTokenizer.from_pretrained(value)
    for key, value in paths.items()
}

# 3) Crear un diccionario de modelos base (AutoModel)
models_base = {
    key: transformers.AutoModel.from_pretrained(value)
    for key, value in paths.items()
}

sentence_transformers_models = {}
"""
Descripción detallada del pipeline:

STmodels.Transformer
Carga el modelo transformer original usando el path correspondiente. Se limita la longitud máxima de las secuencias a 512 tokens.

STmodels.Pooling
Agrega una capa de pooling que resume todos los embeddings de las palabras en un solo vector (por ejemplo, mediante la media).

SentenceTransformer
Se compone de las capas anteriores para formar el modelo final capaz de convertir textos completos en vectores de características densas.
"""

for key, auto_model in models_base.items():
    # 1) Capa de Transformer de Sentence-Transformers
    word_embedding_model = STmodels.Transformer(paths[key], max_seq_length=768)

    # 2) Pooling: Combina las salidas del modelo para obtener un vector por oración
    pooling_model = STmodels.Pooling(
        word_embedding_model.get_word_embedding_dimension()
    )

    # 3) Instanciar SentenceTransformer
    st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    if torch.cuda.is_available():
        st_model = st_model.to("cuda")
    else:
        st_model = st_model.to("cpu")

    # 4) Guardar en el diccionario
    sentence_transformers_models[key] = st_model


###########################
###########################
###########################
###########################


# Helpers

async def _fetch_user_and_data(user_id: str) -> Tuple[User, List[Cause], List[Community], List[CommunityJoined]]:
    """
    Obtiene el usuario, todas las causas, todas las comunidades y las comunidades en las que participa el usuario.
    Lanza ValueError si falta algún dato esencial.
    """
    # params = {
    #     "isVerified": True,
    #     "limit": 20,
    #     "offset": 0,
    #     "title": "educación"
    # }

    user_data = await fetch_data(f"users/user/{user_id}")
    causes_data = await fetch_data("causes", {"limit": -1})
    communities_data = await fetch_data("communities")
    joined_data = await fetch_data(f"communities/members/{user_id}")

    if not user_data or not causes_data or not communities_data:
        raise ValueError("No se encontraron datos esenciales para recomendación")

    user = User(**user_data)
    causes = [Cause(**c) for c in causes_data.get("data", [])]
    communities = [Community(**c) for c in communities_data.get("data", [])]
    joined = [CommunityJoined(**item.get("props", item)) for item in joined_data.get("data", [])]
    return user, causes, communities, joined


def _validate_bio(user: User) -> None:
    """
    Asegura que la biografía del usuario exista y no esté vacía.
    """
    if not user.bio or not user.bio.strip():
        raise ValueError("La biografía del usuario está vacía")


def _build_user_text(user: User, joined: List[CommunityJoined]) -> str:
    """
    Concatena la bio del usuario con los nombres de comunidades en que participa.
    """
    names = [cj.name for cj in joined if cj.name]
    return f"{user.bio} {' '.join(names)}" if names else user.bio


def _build_user_embedding_separate(
        user: User,
        joined: List[CommunityJoined],
        st_model: SentenceTransformer,
        alpha: float = 0.7,  # peso de la biografía frente a las comunidades
) -> np.ndarray:
    """
    Devuelve un vector shape == (1, 768) con la mezcla:
        user_emb = alpha · emb_bio  +  (1 - alpha) · emb_com
    Si el usuario no tiene comunidades, emb_com = emb_bio (no rompe).
    """
    # ---------- 1) Embedding BIO ------------------------------------------------
    emb_bio = st_model.encode([user.bio])  # (1, 768)

    # ---------- 2) Embedding COMUNIDADES ---------------------------------------
    names = [cj.name for cj in joined if cj.name]
    if names:
        # 2.1 Codifica cada nombre por separado  →  lista de shape (1,768)
        emb_names = [st_model.encode([name]) for name in names]  # [(1,768), …]
        # 2.2 Apila en una matriz                →  (N,768)
        emb_names = np.vstack(emb_names)
        # 2.3 Media (o max/attention)            →  (1,768)
        emb_com = emb_names.mean(axis=0, keepdims=True)
    else:
        emb_com = emb_bio.copy()

    # ---------- 3) Mezcla ponderada --------------------------------------------
    user_emb = alpha * emb_bio + (1 - alpha) * emb_com  # (1,768)
    return user_emb


def _prepare_filters(
        causes: List[Cause],
        communities: List[Community],
        joined_ids: Set[str]
) -> Dict[str, List[Any]]:
    """
    Prepara los conjuntos filtrados de causas y comunidades:
      - filtered_communities: descripciones no vacías y no unidas
      - filtered_active_communities: status ACTIVE, descripciones no vacías y no unidas
      - filtered_verified_causes: causas verificadas con descripción no vacía
      - filtered_causes: todas las causas con descripción no vacía
      - filtered_causes_by_title: todas las causas con título no vacío
      - filtered_verified_causes_by_title: causas verificadas con título no vacío
    """
    active = [c for c in communities if c.status == "ACTIVE"]
    verified = [c for c in causes if c.isVerified]
    return {
        "filtered_communities": [
            c for c in communities
            if c.description and c.description.strip() and c.id not in joined_ids
        ],
        "filtered_active_communities": [
            c for c in active
            if c.description and c.description.strip() and c.id not in joined_ids
        ],
        "filtered_verified_causes": [
            c for c in verified
            if c.description and c.description.strip()
        ],
        "filtered_causes": [
            c for c in causes
            if c.description and c.description.strip()
        ],
        "filtered_causes_by_title": [
            c for c in causes
            if c.title and c.title.strip()
        ],
        "filtered_verified_causes_by_title": [
            c for c in verified
            if c.title and c.title.strip()
        ]
    }


def _format_item(item: Any, reduced_output: bool) -> Any:
    """
    Formatea un objeto Cause o Community para salida reducida o completa.
    """
    if not reduced_output:
        return item
    if isinstance(item, Community):
        return {
            "id": item.id,
            "name": item.name,
            "description": item.description,
            "status": item.status
        }
    if isinstance(item, Cause):
        return {
            "id": item.id,
            "title": item.title,
            "description": item.description,
            "isVerified": item.isVerified
        }
    return item


def _get_title(obj):
    """Devuelve el título de Community o Cause."""
    # Community → name   |  Cause → title
    return getattr(obj, "title", getattr(obj, "name", "")).strip()

def _get_description(obj):
    """Devuelve la descripción (si existe) o cadena vacía."""
    return (obj.description or "").strip()

def _title_and_description(obj):
    """Concatena título + descripción con un espacio intermedio."""
    return f"{_get_title(obj)} {_get_description(obj)}".strip()


def _get_top_n_by_tfidf(
        user_text: str,
        items: List[Any],
        text_getter: Callable[[Any], str],
        reduced_output: bool,
        n: int = 3
) -> List[Dict]:
    """
        Calcula similitud TF-IDF entre user_text y textos de items:
          1) Construye corpus = [user_text] + [text_getter(i) for i in items]
          2) Aplica TfidfVectorizer().fit_transform(corpus)
          3) Obtiene cosine_similarity(vector_user, vectors_items)
          4) Ordena y toma top-n

        Returns:
            Lista de dicts { item: ..., score: ... }
        """
    """
        Aplica fit_transform() para convertir los textos en una matriz TF-IDF
        fit_transform() primero aprende las palabras clave y luego transforma cada texto en un vector numérico.
        Devuelve una matriz dispersa (sparse matrix) de tamaño (n_textos, n_palabras_unicas).
    """
    """
        Calcula la similitud del coseno entre user_text y cada texto en valid_texts

        tfidf_matrix[0] representa el vector TF-IDF de la biografía del usuario.
        tfidf_matrix[1:] representa los vectores TF-IDF de las comunidades o causas.
        cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:]) devuelve una matriz con las similitudes.
        .flatten() convierte la matriz en un array de 1D.
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
    """
    valid_items = [i for i in items if text_getter(i) and text_getter(i).strip()]
    if not valid_items:
        return []
    corpus = [user_text] + [text_getter(i) for i in items]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(matrix[0], matrix[1:]).flatten()
    """
    Obtiene los 3 elementos mejor clasificados según sus puntuaciones de similitud.
    """
    ranked = sorted(zip(items, sims), key=lambda x: x[1], reverse=True)[:n]
    return [{"item": _format_item(i, reduced_output), "score": round(float(score), 4)} for i, score in ranked]


def _get_top_n_by_fasttext(
        user_text: str,
        items: List[Any],
        text_getter: Callable[[Any], str],
        reduced_output: bool,
        n: int = 3
) -> List[Dict]:
    """
    Recomienda top-n items usando embeddings de FastText.
    """

    """
    - Conversión de la biografía del usuario en un vector numérico
    Se utiliza el modelo de FastText preentrenado para convertir la biografía del usuario en un vector numérico.
    FastText asigna un vector a cada palabra y genera un vector final promediando los vectores de todas las palabras.
    La conversión a `np.array` con `dtype=np.float32` asegura compatibilidad con operaciones de similitud.
    Finalmente, `.reshape(1, -1)` transforma el vector en una matriz de una fila, lo cual es necesario para
    su uso en `cosine_similarity()`.
    """
    user_vec = np.array(fasttext_model.get_sentence_vector(user_text), dtype=np.float32).reshape(1, -1)
    texts = [text_getter(i) for i in items]
    valid_texts = [t for t in texts if t and t.strip()]
    if not valid_texts:
        return []
    item_vecs = np.array([fasttext_model.get_sentence_vector(t) for t in valid_texts], dtype=np.float32)
    sims = cosine_similarity(user_vec, item_vecs).flatten()
    ranked = sorted(zip(items, sims), key=lambda x: x[1], reverse=True)[:n]
    return [{"item": _format_item(i, reduced_output), "score": round(float(score), 4)} for i, score in ranked]


# No se llega a usar
def _get_top_n_by_model(
        user_text: str,
        items: List[Any],
        text_getter: Callable[[Any], str],
        reduced_output: bool,
        st_model: SentenceTransformer,
        top_n: int
) -> List[Dict]:
    """
    Calcula embeddings con `st_model` para user_text e items,
    mide similitud coseno y retorna top_n formateados.
    Maneja casos donde algunos textos están vacíos.
    """
    # Filtrar items con texto válido
    valid_pairs = [(item, text_getter(item).strip()) for item in items if
                   text_getter(item) and text_getter(item).strip()]
    if not valid_pairs:
        return []
    valid_items, valid_texts = zip(*valid_pairs)

    # Obtener embeddings
    user_emb = st_model.encode([user_text])  # shape (1, D)
    emb_items = st_model.encode(list(valid_texts))  # shape (N, D)

    # Calcular similitudes
    sims = cosine_similarity(user_emb, emb_items)[0]

    # Ordenar y top_n
    scored = sorted(zip(valid_items, sims), key=lambda x: x[1], reverse=True)[:top_n]

    # Formatear resultados con score
    results: List[Dict] = []
    for item, score in scored:
        results.append({
            "item": _format_item(item, reduced_output),
            "score": round(float(score), 4)
        })
    return results


def _get_top_n_by_user_emb(
        user_emb: np.ndarray,
        items: List[Any],
        text_getter: Callable[[Any], str],
        reduced_output: bool,
        st_model: SentenceTransformer,
        top_n: int,
) -> List[Dict]:
    """Igual que _get_top_n_by_model pero recibe user_emb ya calculado."""
    valid_pairs = [(it, text_getter(it).strip())
                   for it in items if text_getter(it) and text_getter(it).strip()]
    if not valid_pairs:
        return []
    valid_items, valid_texts = zip(*valid_pairs)

    emb_items = st_model.encode(list(valid_texts))  # (N,768)
    sims = cosine_similarity(user_emb, emb_items)[0]  # (N,)

    scored = sorted(zip(valid_items, sims), key=lambda x: x[1], reverse=True)[:top_n]
    results = []
    for it, sc in scored:
        results.append({
            "item": _format_item(it, reduced_output),
            "score": round(float(sc), 4)
        })
    return results


###########################
###########################
###########################
###########################


async def recommend_causes_tfidf(user_id: str, reduced_output: bool = False) -> Dict:
    """
    Recomienda causas y comunidades usando similitud TF-IDF basada en múltiples criterios.

    TF-IDF es una técnica que convierte texto en números para que los modelos de aprendizaje automático puedan
    procesarlo. Se usa en procesamiento de lenguaje natural (NLP) para evaluar la importancia de una palabra en
    un documento dentro de un conjunto de documentos.

    TF (Frecuencia de Término):
    Representa cuántas veces aparece una palabra en un documento en comparación con la longitud total del documento.

    IDF (Frecuencia Inversa del Documento):
    Da mayor peso a palabras poco comunes y reduce el peso de palabras muy frecuentes en todos los documentos.
    Por lo que si una palabra aparece en todos los documentos, su peso será muy bajo.

    Args:
        user_id (str): Identificador único del usuario.
        reduced_output (bool): Si es True, devuelve una versión simplificada de la salida.

    Returns:
        Dict: Diccionario con las causas y comunidades mejor clasificadas.
    """
    """
     - Obtención de datos desde la API
     Se obtienen los datos del usuario, causas y comunidades desde la API REST de la aplicación.
     
    """
    user, causes, communities, joined = await _fetch_user_and_data(user_id)

    """
    - Comunidades a las que el user pertenece
    """
    joined_ids = {cj.id for cj in joined}

    """
    - Verificación de datos
     Se evita continuar el proceso si faltan datos esenciales.
    """

    _validate_bio(user)
    # Texto a comparar
    user_text = _build_user_text(user, joined)

    """
    - Conversión de datos a objetos
    Se convierten los datos en instancias de modelos definidos (User, Cause, Community) para facilitar su manipulación.
    
     -  Filtrado de comunidades y causas
    Se filtran las comunidades y causas para considerar solo aquellas que están activas y verificadas.
    """

    filters = _prepare_filters(causes, communities, joined_ids)

    # Generar recomendaciones para comunidades y causas
    return {
        "user_bio": user_text,

        # ──────────  COMUNIDADES (TODAS)  ──────────
        "top_communities_by_title": _get_top_n_by_tfidf(
            user_text, filters["filtered_communities"],
            _get_title, reduced_output
        ),
        "top_communities_by_description": _get_top_n_by_tfidf(
            user_text, filters["filtered_communities"],
            _get_description, reduced_output
        ),
        "top_communities_title_description": _get_top_n_by_tfidf(
            user_text, filters["filtered_communities"],
            _title_and_description, reduced_output
        ),

        # ──────────  COMUNIDADES ACTIVAS  ──────────
        "top_communities_active_by_title": _get_top_n_by_tfidf(
            user_text, filters["filtered_active_communities"],
            _get_title, reduced_output
        ),
        "top_communities_active_by_description": _get_top_n_by_tfidf(
            user_text, filters["filtered_active_communities"],
            _get_description, reduced_output
        ),
        "top_communities_active_title_description": _get_top_n_by_tfidf(
            user_text, filters["filtered_active_communities"],
            _title_and_description, reduced_output
        ),

        # ──────────  CAUSAS (TODAS)  ──────────
        "top_causes_by_title": _get_top_n_by_tfidf(
            user_text, filters["filtered_causes_by_title"],
            _get_title, reduced_output
        ),
        "top_causes_by_description": _get_top_n_by_tfidf(
            user_text, filters["filtered_causes"],
            _get_description, reduced_output
        ),
        "top_causes_title_description": _get_top_n_by_tfidf(
            user_text, filters["filtered_causes"],
            _title_and_description, reduced_output
        ),

        # ──────────  CAUSAS VERIFICADAS  ──────────
        "top_causes_verified_by_title": _get_top_n_by_tfidf(
            user_text, filters["filtered_verified_causes_by_title"],
            _get_title, reduced_output
        ),
        "top_causes_verified_by_description": _get_top_n_by_tfidf(
            user_text, filters["filtered_verified_causes"],
            _get_description, reduced_output
        ),
        "top_causes_verified_title_description": _get_top_n_by_tfidf(
            user_text, filters["filtered_verified_causes"],
            _title_and_description, reduced_output
        ),
    }


async def recommend_causes_fasttext(user_id: str, reduced_output: bool = False) -> Dict:
    """
    Recomienda causas y comunidades usando similitud de embeddings de FastText.

    FastText es un modelo de aprendizaje profundo para representar palabras como vectores densos (word embeddings).
    Utiliza el modelo preentrenado para calcular la similitud entre la biografía del usuario y las descripciones de
    causas y comunidades. A diferencia de TF-IDF, este enfoque tiene en cuenta el significado semántico de las palabras.

    Args:
        user_id (str): Identificador único del usuario.
        reduced_output (bool): Si es True, devuelve una versión simplificada de la salida.

    Returns:
        Dict: Diccionario con las causas y comunidades mejor clasificadas.
    """
    """
     - Obtención de datos desde la API
     Se obtienen los datos del usuario, causas y comunidades desde la API REST de la aplicación.
    """
    user, causes, communities, joined = await _fetch_user_and_data(user_id)

    """
    - Verificación de datos
     Se evita continuar el proceso si faltan datos esenciales.
    """
    _validate_bio(user)

    user_text = _build_user_text(user, joined)

    joined_ids = {cj.id for cj in joined}

    filters = _prepare_filters(causes, communities, joined_ids)

    # Generar recomendaciones para comunidades y causas con los datos filtrados
    return {
        "user_bio": user_text,

        # ────────── COMUNIDADES (TODAS) ──────────
        "top_communities_by_title": _get_top_n_by_fasttext(
            user_text, filters["filtered_communities"],
            _get_title, reduced_output
        ),
        "top_communities_by_description": _get_top_n_by_fasttext(
            user_text, filters["filtered_communities"],
            _get_description, reduced_output
        ),
        "top_communities_title_description": _get_top_n_by_fasttext(
            user_text, filters["filtered_communities"],
            _title_and_description, reduced_output
        ),

        # ────────── COMUNIDADES ACTIVAS ──────────
        "top_communities_active_by_title": _get_top_n_by_fasttext(
            user_text, filters["filtered_active_communities"],
            _get_title, reduced_output
        ),
        "top_communities_active_by_description": _get_top_n_by_fasttext(
            user_text, filters["filtered_active_communities"],
            _get_description, reduced_output
        ),
        "top_communities_active_title_description": _get_top_n_by_fasttext(
            user_text, filters["filtered_active_communities"],
            _title_and_description, reduced_output
        ),

        # ────────── CAUSAS (TODAS) ──────────
        "top_causes_by_title": _get_top_n_by_fasttext(
            user_text, filters["filtered_causes_by_title"],
            _get_title, reduced_output
        ),
        "top_causes_by_description": _get_top_n_by_fasttext(
            user_text, filters["filtered_causes"],
            _get_description, reduced_output
        ),
        "top_causes_title_description": _get_top_n_by_fasttext(
            user_text, filters["filtered_causes"],
            _title_and_description, reduced_output
        ),

        # ────────── CAUSAS VERIFICADAS ──────────
        "top_causes_verified_by_title": _get_top_n_by_fasttext(
            user_text, filters["filtered_verified_causes_by_title"],
            _get_title, reduced_output
        ),
        "top_causes_verified_by_description": _get_top_n_by_fasttext(
            user_text, filters["filtered_verified_causes"],
            _get_description, reduced_output
        ),
        "top_causes_verified_title_description": _get_top_n_by_fasttext(
            user_text, filters["filtered_verified_causes"],
            _title_and_description, reduced_output
        ),
    }


# Este no se llega a usar, se usa la version extendida.
async def recommend_causes_any_model(
        user_id: str,
        model_key: str,
        top_n: int = 3,
        reduced_output: bool = False
):
    """
    Recomienda causas sociales a un usuario usando un modelo específico de embeddings (beto, roberta, etc.).

    Este método permite usar un modelo seleccionado por el usuario para generar recomendaciones basadas en similitud
    semántica entre la biografía del usuario y las descripciones de las causas disponibles.

    Args:
        user_id (str): Identificador del usuario al que se le desean recomendar causas.
        model_key (str): Clave del modelo a usar, debe existir en `sentence_transformers_models`.
        top_n (int): Número de causas más similares a retornar.
        reduced_output (bool): Si es True, devuelve una salida simplificada.

    Returns:
        Dict: Diccionario con el nombre del modelo usado y una lista de causas recomendadas con su puntuación.
    """
    # 1. Validar que el modelo existe
    if model_key not in sentence_transformers_models:
        return {"error": f"El modelo '{model_key}' no está disponible"}

    # 2. Obtener la instancia de SentenceTransformer
    st_model = sentence_transformers_models[model_key]

    # 3. Cargar info de usuario y causas (similares a tus funciones)
    user_data = await fetch_data(f"users/user/{user_id}")
    if not user_data:
        return {"error": "No se encontró el usuario"}
    user_obj = User(**user_data)

    causes_data = await fetch_data("causes")
    if not causes_data:
        return {"error": "No se encontraron causas"}
    raw_causes = causes_data["data"]

    # Convertir a objetos Pydantic
    valid_causes = []
    for cdict in raw_causes:
        if cdict.get("description") and cdict["description"].strip():
            valid_causes.append(Cause(**cdict))

    # 4. Embeddings
    user_text = user_obj.bio or ""
    if not user_text.strip():
        return {"error": "La biografía del usuario está vacía"}

    # 5. Obtener embeddings
    user_emb = st_model.encode([user_text])  # shape (1,768)
    cause_embs = st_model.encode([c.description for c in valid_causes])  # shape (N,768)

    # 6. Calcular similitud entre usuario y cada causa
    similarities = cosine_similarity(user_emb, cause_embs)[0]  # (N,)

    # 7. Ordenar
    scored = list(zip(valid_causes, similarities))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_scored = scored[:top_n]

    # 8. Formatear salida
    # 8. Función de formateo: completa o recortada
    def format_cause(cause: Cause, score: float):
        if reduced_output:
            return {
                "id": cause.id,
                "title": cause.title,
                "description": cause.description,
                "isVerified": cause.isVerified,
                "similarity": round(float(score), 4)
            }
        else:
            cause_dict = cause.dict()
            # Puedes agregar el score si lo consideras útil
            cause_dict["similarity"] = round(float(score), 4)
            return cause_dict

    # Retornar directamente la lista de recomendaciones
    return [format_cause(c, s) for (c, s) in top_scored]


async def recommend_causes_any_model_extended(
        user_id: str,
        model_key: str,
        top_n: int = 3,
        reduced_output: bool = False
) -> Dict:
    """
    Versión extendida que recomienda tanto causas como comunidades,
    evitando las comunidades ya unidas y usando un modelo de SentenceTransformer.
    """
    # Validar modelo
    if model_key not in sentence_transformers_models:
        return {"error": f"Modelo '{model_key}' no disponible"}
    st_model = sentence_transformers_models[model_key]

    # Cargar datos de usuario, causas y comunidades
    user, causes, communities, joined = await _fetch_user_and_data(user_id)
    _validate_bio(user)
    # user_text = _build_user_text(user, joined) ya no sumamos los textos sino que calculamos los embedings y los juntamos
    user_text = user.bio
    user_emb = _build_user_embedding_separate(user, joined, st_model,
                                              alpha=0.7)  # Calcula el embeding del user con las comunidades
    joined_ids: Set[str] = {cj.id for cj in joined}

    # Preparar filtros (evitar comunidades unidas)
    filters = _prepare_filters(causes, communities, joined_ids)

    # Rankings con el modelo
    return {
        "model": model_key,
        "user_bio": user_text,

        # ────────── COMUNIDADES (TODAS) ──────────
        "top_communities_by_title": _get_top_n_by_user_emb(
            user_emb, filters["filtered_communities"],
            _get_title, reduced_output, st_model, top_n
        ),
        "top_communities_by_description": _get_top_n_by_user_emb(
            user_emb, filters["filtered_communities"],
            _get_description, reduced_output, st_model, top_n
        ),
        "top_communities_title_description": _get_top_n_by_user_emb(
            user_emb, filters["filtered_communities"],
            _title_and_description, reduced_output, st_model, top_n
        ),

        # ────────── COMUNIDADES ACTIVAS ──────────
        "top_communities_active_by_title": _get_top_n_by_user_emb(
            user_emb, filters["filtered_active_communities"],
            _get_title, reduced_output, st_model, top_n
        ),
        "top_communities_active_by_description": _get_top_n_by_user_emb(
            user_emb, filters["filtered_active_communities"],
            _get_description, reduced_output, st_model, top_n
        ),
        "top_communities_active_title_description": _get_top_n_by_user_emb(
            user_emb, filters["filtered_active_communities"],
            _title_and_description, reduced_output, st_model, top_n
        ),

        # ────────── CAUSAS (TODAS) ──────────
        "top_causes_by_title": _get_top_n_by_user_emb(
            user_emb, filters["filtered_causes_by_title"],
            _get_title, reduced_output, st_model, top_n
        ),
        "top_causes_by_description": _get_top_n_by_user_emb(
            user_emb, filters["filtered_causes"],
            _get_description, reduced_output, st_model, top_n
        ),
        "top_causes_title_description": _get_top_n_by_user_emb(
            user_emb, filters["filtered_causes"],
            _title_and_description, reduced_output, st_model, top_n
        ),

        # ────────── CAUSAS VERIFICADAS ──────────
        "top_causes_verified_by_title": _get_top_n_by_user_emb(
            user_emb, filters["filtered_verified_causes_by_title"],
            _get_title, reduced_output, st_model, top_n
        ),
        "top_causes_verified_by_description": _get_top_n_by_user_emb(
            user_emb, filters["filtered_verified_causes"],
            _get_description, reduced_output, st_model, top_n
        ),
        "top_causes_verified_title_description": _get_top_n_by_user_emb(
            user_emb, filters["filtered_verified_causes"],
            _title_and_description, reduced_output, st_model, top_n
        ),
    }


async def compare_models(user_id: str, reduced_output: bool = False) -> Dict:
    """ Comparar TF-IDF vs FastText """
    return {
        "TF-IDF": await recommend_causes_tfidf(user_id, reduced_output=reduced_output),
        "FastText": await recommend_causes_fasttext(user_id, reduced_output=reduced_output)
    }


async def recommend_users_for_cause_fasttext(
        cause_id: str,
        top_n: int = 3,
        reduced_output: bool = False
) -> Dict:
    """
    Recomienda usuarios a partir de la descripción de una causa usando embeddings de FastText.

    Args:
        cause_id (str): ID de la causa a analizar.
        top_n (int): Número de usuarios a recomendar.
        reduced_output (bool): Salida simplificada o completa.

    Returns:
        Dict: Diccionario con la información de la causa y lista de usuarios recomendados.
    """
    # 1. Obtener datos de la causa

    cause_data = await fetch_data(f"causes/{cause_id}")

    if not cause_data:
        return {"error": f"No se encontró la causa con ID {cause_id}."}

    cause_dict = cause_data.get("data")

    if not cause_dict:
        return {"error": "La causa no viene en la respuesta o está vacía."}

    # Ahora sí creas el objeto Pydantic con la parte que corresponde
    cause_obj = Cause(**cause_dict)

    # 2. Obtener todos los usuarios
    users_list = await fetch_data("users/all")
    if not users_list or len(users_list) == 0:
        return {"error": "No hay usuarios disponibles."}
    users_objects = [User(**u) for u in users_list]

    # 3. Filtrar usuarios sin bio
    valid_users = [u for u in users_objects if u.bio and u.bio.strip()]
    if not valid_users:
        return {"error": "No hay usuarios con bio válida."}

    # 4. Obtener embedding de la causa
    cause_text = cause_obj.description or ""
    cause_vector = fasttext_model.get_sentence_vector(cause_text)
    # Reshape para usar con cosine_similarity
    import numpy as np
    cause_vector = np.array(cause_vector, dtype=np.float32).reshape(1, -1)

    # 5. Obtener embeddings de cada user
    user_vectors = []
    for u in valid_users:
        v = fasttext_model.get_sentence_vector(u.bio or "")
        user_vectors.append(v)
    user_vectors = np.array(user_vectors, dtype=np.float32)

    # 6. Similaridad
    similarities = cosine_similarity(cause_vector, user_vectors).flatten()

    # 7. Ordenar y top N
    scored_users = list(zip(valid_users, similarities))
    scored_users.sort(key=lambda x: x[1], reverse=True)
    top_users = scored_users[:top_n]

    # 8. Formar resultado
    def format_user(u: User, score: float) -> Dict:
        if reduced_output:
            return {
                "solidarianId": u.solidarianId,
                "name": f"{u.firstName} {u.lastName}",
                "bio": u.bio,
                "score": float(score)
            }
        else:
            return {
                "user": u.__dict__,
                "score": float(score)
            }

    result = {
        "cause": {
            "id": cause_obj.id,
            "title": cause_obj.title,
            "description": cause_obj.description,
        },
        "recommended_users": [
            format_user(u, s) for (u, s) in top_users
        ],
    }
    return result


async def recommend_users_for_cause_tfidf(
        cause_id: str,
        top_n: int = 3,
        reduced_output: bool = False
) -> Dict:
    """
    Recomienda usuarios a partir de la descripción de una causa usando TF-IDF.

    Args:
        cause_id (str): ID de la causa a analizar.
        top_n (int, opcional): Número de usuarios a recomendar. Defaults a 3.
        reduced_output (bool, opcional): Si es True, se devuelve un output simplificado.

    Returns:
        Dict: Diccionario con la información de la causa y el top de usuarios más afines.
    """
    # 1. Obtener datos de la causa
    cause_data = await fetch_data(f"causes/{cause_id}")
    if not cause_data:
        return {"error": f"No se encontró la causa con ID {cause_id}."}

    # Convierte cause_data a modelo si lo necesitas
    # O bien, si el endpoint 'causes/{id}' ya devuelve algo como { "id", "title", "description", ... }
    # ajusta según tu respuesta real.

    cause_dict = cause_data.get("data")
    if not cause_dict:
        return {"error": "La causa no viene en la respuesta o está vacía."}

    # Ahora sí creas el objeto Pydantic con la parte que corresponde
    cause_obj = Cause(**cause_dict)

    # 2. Obtener datos de todos los usuarios
    users_list = await fetch_data("users/all")
    if not users_list or len(users_list) == 0:
        return {"error": "No hay usuarios disponibles."}

    # En tu ejemplo, GET /users/ getAll? te devuelve directamente un array JSON.
    # Por lo tanto, users_list podría ser una lista de dicts:
    # [
    #   {
    #       "firstName": "...",
    #       "lastName": "...",
    #       "bio": "...",
    #       "solidarianId": "...",
    #       ...
    #   }, ...
    # ]
    # Conviértelo a objetos si deseas:
    users_objects = [User(**u) for u in users_list]  # Ajusta con tus campos

    # 3. Filtrar usuarios SIN bio
    valid_users = [u for u in users_objects if u.bio and u.bio.strip()]
    if not valid_users:
        return {"error": "No hay usuarios con bio válida."}

    # 4. Preparar textos
    # Asumimos que la causa es representada por cause_obj.description
    cause_text = cause_obj.description or ""
    user_texts = [u.bio for u in valid_users]

    # Vectorizar
    vectorizer = TfidfVectorizer()
    # El primer vector (índice 0) será la causa, y luego vienen todos los usuarios
    tfidf_matrix = vectorizer.fit_transform([cause_text] + user_texts)
    # Calculamos similitud de coseno del vector 0 (causa) vs [1:]
    similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:]).flatten()

    # 5. Ordenar y obtener top_n
    # Emparejamos (usuario, score)
    scored_users = list(zip(valid_users, similarities))
    scored_users = sorted(scored_users, key=lambda x: x[1], reverse=True)
    top_users = scored_users[:top_n]

    # 6. Formar resultado final
    def format_user(u: User, score: float) -> Dict:
        if reduced_output:
            return {
                "solidarianId": u.solidarianId,
                "name": f"{u.firstName} {u.lastName}",
                "bio": u.bio,
                "score": float(score)
            }
        else:
            # Retorna el objeto completo (o un dict con más campos)
            return {
                "user": u.__dict__,
                "score": float(score)
            }

    result = {
        "cause": {
            "id": cause_obj.id,
            "title": cause_obj.title,
            "description": cause_obj.description,
        },
        "recommended_users": [
            format_user(u, s) for (u, s) in top_users
        ],
    }
    return result
