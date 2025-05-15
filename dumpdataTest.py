import asyncio
import httpx
import json

API_BASE_URL = "http://localhost:3003"
LOGIN_URL = f"{API_BASE_URL}/auth/login"
COMMUNITIES_URL = f"{API_BASE_URL}/communities"
CAUSES_URL = f"{API_BASE_URL}/causes"
EMAIL = "juan.perez@example.com"
PASSWORD = "Password123"
TOKEN = None
OWNER_ID = "94175190-0642-492f-afb8-5dc750785f43"


# Función para hacer login y obtener el token
async def login():
    global TOKEN
    async with httpx.AsyncClient() as client:
        response = await client.post(LOGIN_URL, json={"email": EMAIL, "password": PASSWORD})
        response.raise_for_status()
        TOKEN = response.json().get("token")
        print(f"🔑 Token obtenido: {TOKEN}")


# Lista de comunidades con nombres y descripciones variadas
communities = [
    {"name": "Solidarios Unidos",
     "description": "Un grupo comprometido con la ayuda mutua y la colaboración en diversas causas sociales. Aquí trabajamos juntos para construir una sociedad más justa, ayudando a personas en situación vulnerable y promoviendo proyectos solidarios en distintas áreas como educación, salud y medioambiente."},
    {"name": "Apoyo a Familias Vulnerables",
     "description": "Nuestra misión es apoyar a familias en riesgo de exclusión social, brindando ayuda económica, asistencia psicológica y recursos para mejorar sus condiciones de vida. Creemos en la importancia de la comunidad para hacer la diferencia y trabajamos activamente en iniciativas locales."},
]

with open("causasfinal.json", "r", encoding="utf-8") as f:
    causas = json.load(f)


# Función para crear comunidades en la API
async def create_communities():
    await login()  # Hacer login antes de enviar solicitudes
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"}
    success_count = 0
    failed_count = 0

    async with httpx.AsyncClient() as client:
        for community in communities:
            data = {
                "name": community["name"],
                "description": community["description"],
                "ownerId": "5739a3f2-9441-422e-977f-e1e81f2467c4"
            }
            try:
                response = await client.post(COMMUNITIES_URL, json=data, headers=headers)
                if response.status_code == 201:
                    print(f"✅ Comunidad creada: {community['name']}")
                    success_count += 1
                else:
                    print(f"❌ Error creando comunidad: {community['name']} - {response.status_code}")
                    failed_count += 1
            except httpx.RequestError as e:
                print(f"🚨 Error en la conexión: {e}")
                failed_count += 1

            await asyncio.sleep(0.5)  # Pequeño retraso para evitar saturar la API

    print(f"\n📊 Resumen: {success_count} comunidades creadas, {failed_count} errores.")


async def create_causes():
    await login()  # Hacer login antes de enviar solicitudes
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"}
    success_count = 0
    failed_count = 0

    async with httpx.AsyncClient() as client:
        for causa in causas:
            try:
                response = await client.post(CAUSES_URL, json=causa, headers=headers)
                if response.status_code == 201:
                    print(f"✅ Causa creada: {causa['title']}")
                    success_count += 1
                else:
                    print(f"❌ Error creando causa: {causa['title']} - {response.status_code}")
                    failed_count += 1
            except httpx.RequestError as e:
                print(f"🚨 Error en la conexión: {e}")
                failed_count += 1

            await asyncio.sleep(0.5)  # Pequeño retraso para evitar saturar la API

    print(f"\n📊 Resumen: {success_count} causas creadas, {failed_count} errores.")


# Ejecutar el script asincrónico
if __name__ == "__main__":
    asyncio.run(create_causes())
