import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(
    PROJECT_ROOT / ".env",
    override=False,
)

API_URL = "http://127.0.0.1:7860/predict"

API_KEY = os.getenv("API_KEY")

FEATURE_NAMES_PATH = (
    PROJECT_ROOT
    / "models"
    / "feature_names.json"
)


if not API_KEY:
    raise RuntimeError(
        "La variable d'environnement API_KEY "
        "n'est pas définie."
    )


# -------------------------------------------------------------------
# Chargement des features
# -------------------------------------------------------------------

if not FEATURE_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {FEATURE_NAMES_PATH}"
    )


with open(
    FEATURE_NAMES_PATH,
    "r",
    encoding="utf-8",
) as file:
    feature_names = json.load(file)


if len(feature_names) != 656:
    raise RuntimeError(
        "Le fichier feature_names.json doit contenir "
        f"656 features. Reçu : {len(feature_names)}."
    )


# -------------------------------------------------------------------
# Payload de test
# -------------------------------------------------------------------

features = {
    feature_name: 0.0
    for feature_name in feature_names
}


payload = {
    "features": features,
}


# -------------------------------------------------------------------
# Appel du conteneur Docker
# -------------------------------------------------------------------

response = httpx.post(
    API_URL,
    json=payload,
    headers={
        "X-API-Key": API_KEY,
    },
    timeout=30.0,
)


# -------------------------------------------------------------------
# Résultat
# -------------------------------------------------------------------

print(
    "\n=== TEST API DOCKER ===\n"
)

print(
    "URL :",
    API_URL,
)

print(
    "Features envoyées :",
    len(features),
)

print(
    "Status code :",
    response.status_code,
)


try:
    response_payload = response.json()

except ValueError:
    print(
        "Réponse non JSON :",
        response.text,
    )

    raise RuntimeError(
        "L'API Docker n'a pas retourné "
        "une réponse JSON valide."
    )


print(
    "Réponse API :",
    response_payload,
)


if response.status_code != 200:
    raise RuntimeError(
        "Le test Docker a échoué. "
        f"Status={response.status_code}"
    )


print(
    "\nTest Docker réussi."
)