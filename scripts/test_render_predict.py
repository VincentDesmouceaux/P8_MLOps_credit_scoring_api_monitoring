import json
import os

import httpx


API_URL = (
    "https://p8-mlops-credit-scoring-api-monitoring.onrender.com/predict"
)

api_key = os.getenv("API_KEY")

if not api_key:
    raise RuntimeError(
        "La variable d'environnement API_KEY n'est pas définie."
    )


with open("models/feature_names.json", "r", encoding="utf-8") as file:
    feature_names = json.load(file)


features = {
    feature_name: 0.0
    for feature_name in feature_names
}


response = httpx.post(
    API_URL,
    json={
        "features": features
    },
    headers={
        "X-API-Key": api_key
    },
    timeout=60.0,
)


print("Status code :", response.status_code)
print("Réponse API :", response.json())