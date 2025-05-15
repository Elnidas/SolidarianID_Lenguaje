from typing import List, Optional
from enum import Enum
from fastapi import APIRouter, HTTPException, Query, Path


from app.services.recommendation import recommend_causes_tfidf, recommend_causes_fasttext, \
    compare_models, recommend_users_for_cause_tfidf, recommend_users_for_cause_fasttext,recommend_causes_any_model_extended

router = APIRouter(prefix="/recommendation", tags=["Recommendation"])

example_id = "20ed7f97-4ca6-47f8-a7ca-b9bdebd996c3"
class ModelKey(str, Enum):
    beto_uncased = "beto-uncased"
    beto_cased = "beto-cased"
    maria = "maria"



@router.get("/tfidf/{user_id}")
async def get_recommendation_tfidf(user_id: str = Path(..., example=example_id), reduced_output: bool = Query(False, description="Devuelve solo información reducida si es True")):
    """ Endpoint para recomendación con TF-IDF """
    try:
        return await recommend_causes_tfidf(user_id, reduced_output=reduced_output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fasttext/{user_id}")
async def get_recommendation_fasttext(user_id: str = Path(..., example=example_id), reduced_output: bool = Query(False, description="Devuelve solo información reducida si es True")):
    """ Endpoint para recomendación con FastText """
    try:
        return await recommend_causes_fasttext(user_id, reduced_output=reduced_output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/compare/{user_id}")
async def compare_recommendations(user_id: str = Path(..., example=example_id), reduced_output: bool = Query(False, description="Devuelve solo información reducida si es True")):
    """ Endpoint para comparar TF-IDF vs FastText """
    try:
        return await compare_models(user_id, reduced_output=reduced_output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users-for-cause/tfidf", summary="Recomendar usuarios (TF-IDF)")
async def get_users_for_cause_tfidf(
        cause_id: str,
        top_n: Optional[int] = 3,
        reduced_output: Optional[bool] = False
):
    """
    Dado el ID de una causa, devuelve los usuarios más afines usando TF-IDF.

    - **cause_id**: Identificador único de la causa.
    - **top_n**: Número de usuarios recomendados (por defecto 3).
    - **reduced_output**: Devuelve información reducida de los usuarios (True/False).
    """
    result = await recommend_users_for_cause_tfidf(cause_id, top_n, reduced_output)
    return result


@router.get("/users-for-cause/fasttext", summary="Recomendar usuarios (FastText)")
async def get_users_for_cause_fasttext(
        cause_id: str,
        top_n: Optional[int] = 3,
        reduced_output: Optional[bool] = False
):
    """
    Dado el ID de una causa, devuelve los usuarios más afines usando embeddings de FastText.

    - **cause_id**: Identificador único de la causa.
    - **top_n**: Número de usuarios recomendados (por defecto 3).
    - **reduced_output**: Devuelve información reducida de los usuarios (True/False).
    """

    result = await recommend_users_for_cause_fasttext(cause_id, top_n, reduced_output)
    return result


@router.get(
    "/compare-all/{user_id}",
    summary="Compara TF-IDF, FastText y uno o varios modelos de SentenceTransformer",
)
async def compare_all_models(
    user_id: str = Path(..., example=example_id),
    reduced_output: bool = Query(False, description="Si es True, salida reducida"),
    model_keys: Optional[List[ModelKey]] = Query(
        None,
        description="Lista de modelos a comparar (p.ej. beto_uncased, maria)."
    ),
):
    """
    Devuelve un diccionario con las recomendaciones de:
      - TF-IDF
      - FastText
      - Cada modelo de SentenceTransformer pasado en `model_keys`
    """
    try:
        # Siempre incluir TF-IDF y FastText
        results = {
            "TF-IDF": await recommend_causes_tfidf(user_id, reduced_output=reduced_output),
            "FastText": await recommend_causes_fasttext(user_id, reduced_output=reduced_output),
        }

        # Si se han pedido modelos adicionales, los procesamos
        if model_keys:
            for key in model_keys:
                # recommend_causes_any_model_extended devuelve { model, user_bio, top_... }
                rec = await recommend_causes_any_model_extended(
                    user_id,
                    key.value,
                    top_n=3,
                    reduced_output=reduced_output
                )
                # Lo metemos bajo su clave
                results[key.value] = rec

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


'''

@router.get("/users-for-cause/compare", summary="Comparar TF-IDF vs FastText")
async def compare_users_for_cause(
        cause_id: str,
        top_n: Optional[int] = 3,
        reduced_output: Optional[bool] = False
):
    """
    Compara ambos modelos (TF-IDF y FastText) para recomendar usuarios basándose en la descripción de una causa.

    Retorna un diccionario con dos llaves: "TF-IDF" y "FastText", cada una con su top-N.
    """


    tfidf_result = await recommend_users_for_cause_tfidf(cause_id, top_n, reduced_output)
    fasttext_result = await recommend_users_for_cause_fasttext(cause_id, top_n, reduced_output)

    return {
        "TF-IDF": tfidf_result,
        "FastText": fasttext_result
    }
'''

@router.get("/model/multiple/{user_id}")
async def get_recommendation_by_multiple_models(
        user_id: str = Path(..., example=example_id),
        reduced_output: Optional[bool] = False,
        model_keys: List[ModelKey] = Query(..., description="Selecciona uno o más modelos a utilizar"),

):
    """
    Endpoint para recomendar causas usando múltiples modelos.
    """
    try:
        recommendations = []
        for key in model_keys:
            rec = await recommend_causes_any_model_extended(user_id, key.value,3,reduced_output)
            recommendations.append({"model": key.value, "recommendations": rec})
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))