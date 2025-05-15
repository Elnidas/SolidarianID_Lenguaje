from fastapi import APIRouter, HTTPException, Path
from typing import List, Optional, Union

from app.models.models import Cause, Community, CauseResponseModel
from app.models.user import User
from app.services.api_client import fetch_data

router = APIRouter(prefix="/debug", tags=["Debug"])


@router.get("/users/{user_id}", response_model=User)
async def test_fetch_user(  user_id: str = Path(..., example="5739a3f2-9441-422e-977f-e1e81f2467c4")):
    """
    Endpoint para probar la obtención de datos de un usuario.
    """
    try:
        user_data = await fetch_data(f"users/user/{user_id}")
        return User(**user_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/", response_model=List[User])
async def test_fetch_all_user():
    """
    Endpoint para probar la obtención de todos los usuarios.

    """
    try:
        users_data = await fetch_data("users/all")
        return [User(**user) for user in users_data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/causes", response_model=List[Cause])
async def test_fetch_causes():
    """
    Obtener todas las causas solidarias con sus datos anidados.
    """
    try:
        raw_json = await fetch_data("causes")
        # Parseamos el JSON global con el modelo de respuesta de causas
        parsed_response = CauseResponseModel(**raw_json)
        # Retornamos solo la lista de causas ya parseadas
        return parsed_response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/causes/{cause_id}", response_model=Cause)
async def test_fetch_one_cause(cause_id: str):
    """
    Obtener una causa específica.
    """
    try:
        cause_data = await fetch_data(f"causes/{cause_id}")

        if isinstance(cause_data.get("community"), dict):
            cause_data["community"] = {"id": cause_data["community"].get("id"),
                                       "name": cause_data["community"].get("name")}
        else:
            cause_data["community"] = None

        return Cause(**cause_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from app.models.models import ResponseModel, Community

@router.get("/communities", response_model=List[Community])
async def test_fetch_communities():
    """
    Obtener todas las comunidades con sus causas y detalles anidados.
    """
    try:
        # 1) Llamas a la API remota que te devuelve el JSON con la forma global
        raw_json = await fetch_data("communities")

        # 2) Parseas directamente el JSON completo con el modelo ResponseModel
        parsed_response = ResponseModel(**raw_json)

        # 3) Retornas sólo la lista de comunidades, ya parseadas y validadas
        return parsed_response.data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/communities/{community_id}", response_model=Community)
async def test_fetch_one_community(community_id: str):
    """
    Endpoint para probar la obtención de una comunidad específica.
    """
    try:
        community_data = await fetch_data(f"communities/{community_id}")
        causes_data = community_data.get("causes", [])

        if not isinstance(causes_data, list):
            community_data["causes"] = []

        return Community(**community_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
