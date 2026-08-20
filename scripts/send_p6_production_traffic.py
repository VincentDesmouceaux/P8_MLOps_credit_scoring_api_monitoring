import os
from pathlib import Path

import httpx
import pandas as pd
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


if not API_URL:
    raise RuntimeError(
        "La variable d'environnement API_URL n'est pas définie."
    )

if not API_KEY:
    raise RuntimeError(
        "La variable d'environnement API_KEY n'est pas définie."
    )


PREDICT_URL = (
    API_URL.rstrip("/")
    + "/predict"
)


PRODUCTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
    / "p6_production_full.csv"
)


# -------------------------------------------------------------------
# Chargement des données
# -------------------------------------------------------------------

def load_production_data() -> pd.DataFrame:
    """
    Charge les observations réalistes issues du jeu test P6.
    """
    if not PRODUCTION_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : "
            f"{PRODUCTION_FILE}"
        )

    dataframe = pd.read_csv(
        PRODUCTION_FILE
    )

    if dataframe.empty:
        raise RuntimeError(
            "Le dataset de production P6 est vide."
        )

    return dataframe


# -------------------------------------------------------------------
# Conversion d'une ligne P6 vers JSON
# -------------------------------------------------------------------

def build_features_payload(
    row: pd.Series,
) -> dict[str, float | None]:
    """
    Convertit une observation pandas en dictionnaire JSON-compatible.

    Les NaN pandas sont transformés en None afin qu'ils soient
    sérialisés en null dans la requête JSON.

    L'API reconstruira ensuite ces valeurs manquantes pour XGBoost.
    """
    return {
        column: (
            None
            if pd.isna(value)
            else float(value)
        )
        for column, value in row.items()
    }


# -------------------------------------------------------------------
# Envoi d'une observation
# -------------------------------------------------------------------

def send_prediction(
    features: dict[str, float | None],
) -> httpx.Response:
    """
    Envoie une observation P6 à l'API Render.
    """
    return httpx.post(
        PREDICT_URL,
        json={
            "features": features,
        },
        headers={
            "X-API-Key": API_KEY,
        },
        timeout=60.0,
    )


# -------------------------------------------------------------------
# Programme principal
# -------------------------------------------------------------------

def main() -> None:
    dataframe = load_production_data()

    print(
        "\n=== TEST TRAFIC P6 -> API P8 ===\n"
    )

    print(
        "Dataset disponible :",
        dataframe.shape,
    )

    print(
        "URL appelée :",
        PREDICT_URL,
    )

    # ---------------------------------------------------------------
    # Une seule observation pour commencer
    # ---------------------------------------------------------------

    row = dataframe.iloc[0]

    missing_values_count = int(
        row.isna().sum()
    )

    features = build_features_payload(
        row
    )

    print(
        "Features envoyées :",
        len(features),
    )

    print(
        "Valeurs manquantes converties en null :",
        missing_values_count,
    )

    if len(features) != 656:
        raise RuntimeError(
            "Le payload ne contient pas exactement "
            "les 656 features attendues."
        )

    # ---------------------------------------------------------------
    # Appel API
    # ---------------------------------------------------------------

    try:
        response = send_prediction(
            features
        )

    except httpx.TimeoutException as error:
        raise RuntimeError(
            "Timeout lors de l'appel à l'API Render."
        ) from error

    except httpx.HTTPError as error:
        raise RuntimeError(
            "Erreur HTTP lors de l'appel à l'API Render : "
            f"{error}"
        ) from error

    print(
        "Status code :",
        response.status_code,
    )

    # ---------------------------------------------------------------
    # Lecture de la réponse
    # ---------------------------------------------------------------

    try:
        payload = response.json()

    except ValueError:
        print(
            "Réponse non JSON :",
            response.text,
        )

        raise RuntimeError(
            "L'API n'a pas retourné une réponse JSON valide."
        )

    print(
        "Réponse API :",
        payload,
    )

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    if response.status_code != 200:
        raise RuntimeError(
            "La première observation P6 "
            "n'a pas été acceptée par l'API. "
            f"Status={response.status_code}, "
            f"réponse={payload}"
        )

    print(
        "\nPremière observation P6 envoyée avec succès."
    )

    request_id = payload.get(
        "request_id"
    )

    if request_id:
        print(
            "request_id :",
            request_id,
        )

        print(
            "\nLa prédiction peut maintenant être "
            "retrouvée précisément dans Supabase."
        )

    else:
        print(
            "\nATTENTION : request_id absent de la réponse."
        )

        print(
            "La version Render n'intègre probablement "
            "pas encore la dernière version de l'API."
        )


if __name__ == "__main__":
    main()