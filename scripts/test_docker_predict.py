import json

import httpx


API_URL = "http://127.0.0.1:7860/predict"


with open("models/feature_names.json", "r", encoding="utf-8") as file:
    feature_names = json.load(file)


features = {
    feature_name: 0.0
    for feature_name in feature_names
}


payload = {
    "features": features
}


response = httpx.post(
    API_URL,
    json=payload,
    timeout=30.0,
)


print("Status code :", response.status_code)
print("Réponse API :", response.json())