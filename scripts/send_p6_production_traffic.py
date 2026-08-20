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


EXPECTED_FEATURE_COUNT = 656

SAMPLE_SIZE = 100

REQUEST_TIMEOUT_SECONDS = 60.0


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

    if len(dataframe.columns) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "Le dataset ne contient pas exactement "
            f"{EXPECTED_FEATURE_COUNT} features. "
            f"Reçu : {len(dataframe.columns)}."
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

    Les valeurs NaN sont converties en None afin d'être
    sérialisées en null dans la requête JSON.

    L'API reconstruira ensuite ces valeurs manquantes
    sous forme de np.nan pour XGBoost.
    """
    features = {
        column: (
            None
            if pd.isna(value)
            else float(value)
        )
        for column, value in row.items()
    }

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "Le payload ne contient pas exactement "
            f"{EXPECTED_FEATURE_COUNT} features."
        )

    return features


# -------------------------------------------------------------------
# Envoi d'une observation
# -------------------------------------------------------------------

def send_prediction(
    client: httpx.Client,
    features: dict[str, float | None],
) -> httpx.Response:
    """
    Envoie une observation P6 à l'API Render.
    """
    return client.post(
        PREDICT_URL,
        json={
            "features": features,
        },
        headers={
            "X-API-Key": API_KEY,
        },
    )


# -------------------------------------------------------------------
# Programme principal
# -------------------------------------------------------------------

def main() -> None:
    dataframe = load_production_data()

    print(
        "\n=== TRAFIC P6 -> API P8 ===\n"
    )

    print(
        "Dataset disponible :",
        dataframe.shape,
    )

    print(
        "URL appelée :",
        PREDICT_URL,
    )

    sample_size = min(
        SAMPLE_SIZE,
        len(dataframe),
    )

    sample_df = dataframe.head(
        sample_size
    )

    successful_requests = 0
    failed_requests = 0

    request_ids: list[str] = []

    total_missing_values = 0

    # ---------------------------------------------------------------
    # Client HTTP réutilisé pour les 100 requêtes
    # ---------------------------------------------------------------

    timeout = httpx.Timeout(
        REQUEST_TIMEOUT_SECONDS
    )

    with httpx.Client(
        timeout=timeout
    ) as client:

        for index, (_, row) in enumerate(
            sample_df.iterrows(),
            start=1,
        ):
            missing_values_count = int(
                row.isna().sum()
            )

            total_missing_values += (
                missing_values_count
            )

            features = build_features_payload(
                row
            )

            try:
                response = send_prediction(
                    client=client,
                    features=features,
                )

            except httpx.TimeoutException:
                failed_requests += 1

                print(
                    f"[{index}/{sample_size}] "
                    "TIMEOUT"
                )

                continue

            except httpx.HTTPError as error:
                failed_requests += 1

                print(
                    f"[{index}/{sample_size}] "
                    f"ERREUR HTTP : {error}"
                )

                continue

            # -------------------------------------------------------
            # Lecture JSON
            # -------------------------------------------------------

            try:
                payload = response.json()

            except ValueError:
                failed_requests += 1

                print(
                    f"[{index}/{sample_size}] "
                    f"{response.status_code} | "
                    "réponse non JSON"
                )

                continue

            # -------------------------------------------------------
            # Succès
            # -------------------------------------------------------

            if response.status_code == 200:
                successful_requests += 1

                request_id = payload.get(
                    "request_id"
                )

                if request_id:
                    request_ids.append(
                        request_id
                    )

                probability_default = payload.get(
                    "probability_default"
                )

                prediction = payload.get(
                    "prediction"
                )

                if isinstance(
                    probability_default,
                    (int, float),
                ):
                    score_display = (
                        f"{probability_default:.4f}"
                    )
                else:
                    score_display = "N/A"

                print(
                    f"[{index}/{sample_size}] "
                    f"200 | "
                    f"prediction={prediction} | "
                    f"score={score_display} | "
                    f"null={missing_values_count}"
                )

                continue

            # -------------------------------------------------------
            # Erreur API
            # -------------------------------------------------------

            failed_requests += 1

            print(
                f"[{index}/{sample_size}] "
                f"{response.status_code} | "
                f"{payload}"
            )

    # ----------------------------------------------------------------
    # Résumé
    # ----------------------------------------------------------------

    print(
        "\n=== RESUME TRAFIC P6 ===\n"
    )

    print(
        "Observations prévues :",
        sample_size,
    )

    print(
        "Succès :",
        successful_requests,
    )

    print(
        "Échecs :",
        failed_requests,
    )

    print(
        "request_id récupérés :",
        len(request_ids),
    )

    print(
        "Valeurs manquantes transmises en null :",
        total_missing_values,
    )

    success_rate = (
        successful_requests / sample_size
        if sample_size > 0
        else 0.0
    )

    print(
        "Taux de succès :",
        f"{success_rate * 100:.2f} %",
    )

    if successful_requests == sample_size:
        print(
            "\nLes observations P6 ont toutes été "
            "traitées avec succès."
        )

    else:
        print(
            "\nATTENTION : certaines observations "
            "n'ont pas été acceptées par l'API."
        )

    if request_ids:
        print(
            "\nPremier request_id :",
            request_ids[0],
        )

        print(
            "Dernier request_id :",
            request_ids[-1],
        )

        print(
            "\nCes identifiants peuvent être utilisés "
            "pour retrouver les prédictions dans Supabase."
        )


if __name__ == "__main__":
    main()