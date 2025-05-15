from typing import Dict, List

from pydantic import BaseModel


class RecommendedUser(BaseModel):
    id: str
    bio: str
    score: float

class RecommendUsersResponse(BaseModel):
    cause: Dict[str, str]  # un dict con id, title, desc
    recommended_users: List[RecommendedUser]