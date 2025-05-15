
from fastapi import APIRouter,  Query
from app.services.offensiveClasifier import classify

router = APIRouter(prefix="/moderation", tags=["Moderation"])


@router.get("/offensive-check")
async def offensive_check(text: str = Query(..., min_length=1)):
    """
    Devuelve si *text* es ofensivo (modelo «maria»).
    """
    is_off, label = classify(text)
    return {"text": text, "is_offensive": is_off, "label": label}
