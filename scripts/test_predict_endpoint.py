from fastapi.testclient import TestClient

from app.main import app
from app.services.model_service import model_service


client = TestClient(app)

# Récupération automatique des 656 features attendues
features = {
    feature_name: 0.0
    for feature_name in model_service.feature_names
}

response = client.post(
    "/predict",
    json={"features": features},
)

print("Status code :", response.status_code)
print("Réponse API :", response.json())