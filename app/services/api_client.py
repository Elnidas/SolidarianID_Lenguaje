import httpx
from app.config import API_BASE_URL

# Variable global para almacenar el token
# TOKEN = None
TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIzNzRiMWY4Ni0zOWM4LTRiNDktOGY5MC1iNDVjMjUzYjRiNjciLCJlbWFpbCI6Imp1YW4ucGVyZXpAZXhhbXBsZS5jb20iLCJyb2xlIjoidXNlciIsImlhdCI6MTc0NzIzNTIxNX0.g9E0WtQv_EIntpjCURfuulbB9CqMaid9QY91ijY_ymo'

async def login(email: str, password: str):
    """
    Realiza el login en la API y almacena el token de autenticación.
    """
    global TOKEN
    async with httpx.AsyncClient() as client:

        response = await client.post(f"{API_BASE_URL}/auth/login", json={"email": email, "password": password})
        response.raise_for_status()
        TOKEN = response.json().get("token")
        print(TOKEN)

async def fetch_data(endpoint: str,params: dict = None):
    """
    Hace una petición GET al servidor REST local, incluyendo el token si existe.
    """
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}/{endpoint}", headers=headers,params=params)
        response.raise_for_status()
        return response.json()
