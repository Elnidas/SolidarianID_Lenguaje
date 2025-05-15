from pydantic import BaseModel
from typing import Optional

class User(BaseModel):
    solidarianId: str
    firstName: str
    lastName: str
    email: str
    bio: Optional[str] = None
