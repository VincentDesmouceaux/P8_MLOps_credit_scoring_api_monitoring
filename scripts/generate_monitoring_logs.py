import json
import os
import random
import time
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

API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")

FEATURE_NAMES_PATH = (
    PROJECT_ROOT
    / "models"
    / "feature_names.json"
)

NUMBER_VALID_REQUESTS = 20
NUMBER_INVALID_REQUESTS = 5

REQUEST_TIMEOUT_SECONDS = 60.0


# -------------------------------------------------------------------
# Validation de la configuration
# -------------------------------------------------------------------

if not API_URL:
    raise RuntimeError(
        "La variable d'environnement API_URL "
        "n'est pas définie."
    )

if not API_KEY:
    raise RuntimeError(
        "La variable d'environnement API_KEY "
        "n'est pas définie."
    )

PREDICT_URL = (
    API_URL.rstrip("/")
    + "/predict"
)


# -------------------------------------------------------------------
# Chargement des features
# -------------------------------------------------------------------

if not FEATURE_NAMES_PATH.exists():
    raise FileNotFoundError(
        "Fichier de features introuvable : "
        f"{FEATURE_NAMES_PATH}"
    )

with open(
    FEATURE_NAMES_PATH,
    "r",
    encoding="utf-8",
) as file:
    feature_names = json.load(file)

if not isinstance(
    feature_names,
    list,
):
    raise TypeError(
        "feature_names.json doit contenir "
        "une liste."
    )

if len(feature_names) != 656:
    raise RuntimeError(
        "Le modèle doit utiliser exactement "
        f"656 features. Reçu : {len(feature_names)}."
    )


# -------------------------------------------------------------------
# Construction des features
# -------------------------------------------------------------------

def build_valid_features() -> dict[str, float]:
    """
    Construit un payload contenant exactement
    les 656 features attendues.

    Les valeurs sont volontairement simulées afin
    de générer du trafic pour le monitoring opérationnel.

    Ce script ne doit pas être utilisé pour simuler
    un Data Drift réaliste : pour cela, utiliser les
    scripts basés sur les données P6.
    """
    return {
        feature_name: random.uniform(
            0.0,
            1.0,
        )
        for feature_name in feature_names
    }


# -------------------------------------------------------------------
# Requête valide
# -------------------------------------------------------------------

def send_valid_request(
    client: httpx.Client,
    request_number: int,
) -> None:
    features = build_valid_features()

    response = client.post(
        PREDICT_URL,
        json={
            "features": features,
        },
        headers={
            "X-API-Key": API_KEY,
        },
    )

    print(
        f"[VALID {request_number:02d}] "
        f"status={response.status_code}"
    )

    if response.status_code == 200:
        data = response.json()

        print(
            "    "
            f"score={data['probability_default']:.4f} | "
            f"prediction={data['prediction']} | "
            f"label={data['prediction_label']}"
        )

        return

    print(
        "    "
        f"response={response.text}"
    )


# -------------------------------------------------------------------
# Requête invalide
# -------------------------------------------------------------------

def send_invalid_request(
    client: httpx.Client,
    request_number: int,
) -> None:
    """
    Envoie volontairement un payload incomplet.

    L'objectif est de produire une erreur métier 422
    afin de vérifier la journalisation des anomalies.
    """
    response = client.post(
        PREDICT_URL,
        json={
            "features": {},
        },
        headers={
            "X-API-Key": API_KEY,
        },
    )

    print(
        f"[ERROR {request_number:02d}] "
        f"status={response.status_code}"
    )

    print(
        "    "
        f"response={response.text}"
    )


# -------------------------------------------------------------------
# Programme principal
# -------------------------------------------------------------------

def main() -> None:
    print(
        "\n=== GENERATION DE TRAFIC MONITORING ===\n"
    )

    print(
        "API :",
        PREDICT_URL,
    )

    print(
        "Features :",
        len(feature_names),
    )

    print(
        "Requêtes valides prévues :",
        NUMBER_VALID_REQUESTS,
    )

    print(
        "Requêtes invalides prévues :",
        NUMBER_INVALID_REQUESTS,
    )

    random.seed(42)

    timeout = httpx.Timeout(
        REQUEST_TIMEOUT_SECONDS
    )

    with httpx.Client(
        timeout=timeout
    ) as client:

        # -----------------------------------------------------------
        # Trafic valide
        # -----------------------------------------------------------

        print(
            "\n=== Génération des prédictions valides ===\n"
        )

        for index in range(
            1,
            NUMBER_VALID_REQUESTS + 1,
        ):
            send_valid_request(
                client=client,
                request_number=index,
            )

            time.sleep(
                0.2
            )

        # -----------------------------------------------------------
        # Trafic invalide
        # -----------------------------------------------------------

        print(
            "\n=== Génération des erreurs 422 ===\n"
        )

        for index in range(
            1,
            NUMBER_INVALID_REQUESTS + 1,
        ):
            send_invalid_request(
                client=client,
                request_number=index,
            )

            time.sleep(
                0.2
            )

    # ----------------------------------------------------------------
    # Résumé
    # ----------------------------------------------------------------

    print(
        "\n=== Génération terminée ==="
    )

    print(
        f"{NUMBER_VALID_REQUESTS} "
        "requêtes valides envoyées."
    )

    print(
        f"{NUMBER_INVALID_REQUESTS} "
        "erreurs 422 envoyées."
    )


if __name__ == "__main__":
    main()