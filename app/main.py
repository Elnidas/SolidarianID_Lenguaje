from pathlib import Path

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware

from app.routes import  recommendation, offensiveClasifier, debug

app = FastAPI(title="SolidarianID API", version="1.0")

app.include_router(recommendation.router)
app.include_router(offensiveClasifier.router)
app.include_router(debug.router)
BASE_DIR = Path(__file__).resolve().parent  # apunta a app/
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
origins = [
    "http://localhost:3006",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.on_event("startup")
# async def startup():
#     await login("juan.perez@example.com", "Password123")