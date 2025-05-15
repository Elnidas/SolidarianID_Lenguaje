from fastapi import APIRouter, HTTPException
from app.services.api_client import fetch_data

router = APIRouter(prefix="/users", tags=["Users"])

@router.get("/")
async def get_users():
    try:
        users = await fetch_data("users")  # Llama a http://localhost:5000/api/users
        return users
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
