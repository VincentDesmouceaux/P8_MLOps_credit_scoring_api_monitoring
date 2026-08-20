import json
import os
import random
import time

import httpx


API_URL = (
    "https://p8-mlops-credit-scoring-api-monitoring.onrender.com/predict"
)

api_key = os.getenv("API_KEY")

if not api_key:
    raise RuntimeError(
        "La variable d'environnement API_KEY n'est pas définie."
    )


with open(
    "models/feature_names.json",
    "r",
    encoding="utf-8",
) as file:
    feature_names = json.load(file)


def build_valid_features():
    """
    Construit un payload contenant exactement les 656 features
    attendues par le modèle.

    Les valeurs sont légèrement variées uniquement pour simuler
    du trafic et alimenter le monitoring opérationnel.
    """
    return {
        feature_name: random.uniform(0.0, 1.0)
        for feature_name in feature_names
    }


def send_valid_request(client, request_number):
    features = build_valid_features()

    response = client.post(
        API_URL,
        json={
            "features": features,
        },
        headers={
            "X-API-Key": api_key,
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
    else:
        print(
            "    "
            f"response={response.text}"
        )


def send_invalid_request(client, request_number):
    """
    Envoie volontairement un payload incomplet afin de générer
    une erreur métier 422 enregistrée dans prediction_logs.
    """
    response = client.post(
        API_URL,
        json={
            "features": {},
        },
        headers={
            "X-API-Key": api_key,
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


def main():
    random.seed(42)

    number_valid_requests = 20
    number_invalid_requests = 5

    with httpx.Client(timeout=60.0) as client:
        print("\n=== Génération des prédictions valides ===\n")

        for index in range(1, number_valid_requests + 1):
            send_valid_request(
                client=client,
                request_number=index,
            )

            time.sleep(0.2)

        print("\n=== Génération des erreurs 422 ===\n")

        for index in range(1, number_invalid_requests + 1):
            send_invalid_request(
                client=client,
                request_number=index,
            )

            time.sleep(0.2)

    print("\n=== Génération terminée ===")
    print(
        f"{number_valid_requests} requêtes valides envoyées."
    )
    print(
        f"{number_invalid_requests} erreurs 422 envoyées."
    )


if __name__ == "__main__":
    main()