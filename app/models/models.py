# Modelo para el evento
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
class CommunityJoined(BaseModel):
    id: str
    name: str
    totalMembers: int
    causes: List[str]

class CommunitiesJoinedResponseModel(BaseModel):
    status: str
    statusCode: int
    message: str
    timestamp: datetime
    data: List[CommunityJoined]

class Event(BaseModel):
    id: str
    name: str
    description: str
    date: datetime
    street: str
    city: str
    state: str
    zipCode: str
    createdAt: datetime
    updatedAt: datetime

# Modelo para el objetivo de una acción
class ActionObjective(BaseModel):
    id: str
    description: str
    currentValue: Optional[int]  # puede venir nulo
    targetValue: Optional[int]   # puede venir nulo
    createdAt: datetime
    updatedAt: datetime

# Modelo para los detalles de una acción (envueltos en "props")
class ActionProps(BaseModel):
    id: str
    title: str
    description: str
    createdAt: datetime
    updatedAt: datetime
    objectives: List[ActionObjective]
    events: List[Event]

# Modelo de acción que contiene la propiedad "props"
class Action(BaseModel):
    props: ActionProps

# Modelo para la causa
class Cause(BaseModel):
    id: str
    title: str
    description: str
    startDate: datetime
    endDate: datetime
    objectives: List[str]
    isVerified: bool
    # Hacemos "actions" opcional y con valor por defecto como lista vacía
    actions: Optional[List["Action"]] = []
    createdAt: datetime
    # Hacemos "updatedAt" opcional, ya que en algunos casos no se envía
    updatedAt: Optional[datetime] = None
    # Agregamos "community" como opcional para mapear el campo que se envía
    community: Optional[dict] = None

# Modelo para el owner (propietario)
class Owner(BaseModel):
    id: str
    firstName: str
    lastName: str
    email: str

# Modelo para cada comunidad
class Community(BaseModel):
    id: str
    name: str
    description: str
    status: str
    approvedAt: Optional[datetime]
    createdAt: datetime
    updatedAt: datetime
    owner: Owner
    causes: List[Cause]

# Modelo para la respuesta completa
class ResponseModel(BaseModel):
    status: str
    statusCode: int
    message: str
    timestamp: datetime
    data: List[Community]


class CauseResponseModel(BaseModel):
    status: str
    statusCode: int
    message: str
    timestamp: datetime
    data: List[Cause]

class Support(BaseModel):
    causeId: str
    userSolidarianId: str
    createdAt: datetime

class SupportsResponseModel(BaseModel):
    status: str
    statusCode: int
    message: str
    timestamp: datetime
    data: List[Support]




def transformar_supports(response: dict) -> SupportsResponseModel:
    supports_transformados = [support.get("props", support) for support in response.get("data", [])]
    response["data"] = supports_transformados
    return SupportsResponseModel(**response)