import os
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from app.core.database import get_database_url


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

LABELLED_FILE = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
    / "p6_labelled_monitoring_1000.csv"
)

EXPECTED_FEATURE_COUNT = 656
TARGET_COLUMN = "TARGET"

REQUEST_TIMEOUT_SECONDS = 60.0


DATABASE_URL = get_database_url()

SQLALCHEMY_DATABASE_URL = DATABASE_URL.replace(
    "postgresql://",
    "postgresql+psycopg://",
    1,
)


# -------------------------------------------------------------------
# Chargement du dataset
# -------------------------------------------------------------------

def load_labelled_data() -> pd.DataFrame:
    """
    Charge les observations P6 labellisées utilisées
    pour le monitoring supervisé.
    """
    if not LABELLED_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {LABELLED_FILE}"
        )

    dataframe = pd.read_csv(
        LABELLED_FILE
    )

    if dataframe.empty:
        raise RuntimeError(
            "Le dataset de monitoring labellisé est vide."
        )

    if TARGET_COLUMN not in dataframe.columns:
        raise RuntimeError(
            "La colonne TARGET est absente."
        )

    feature_columns = [
        column
        for column in dataframe.columns
        if column != TARGET_COLUMN
    ]

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "Nombre de features incorrect. "
            f"Attendu={EXPECTED_FEATURE_COUNT}, "
            f"reçu={len(feature_columns)}."
        )

    return dataframe


# -------------------------------------------------------------------
# Conversion JSON
# -------------------------------------------------------------------

def build_features_payload(
    row: pd.Series,
) -> dict[str, float | None]:
    """
    Convertit les 656 features en payload JSON-compatible.

    Les NaN pandas deviennent None puis null dans JSON.
    """
    features = {}

    for column, value in row.items():
        if column == TARGET_COLUMN:
            continue

        features[column] = (
            None
            if pd.isna(value)
            else float(value)
        )

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "Le payload ne contient pas exactement "
            f"{EXPECTED_FEATURE_COUNT} features."
        )

    return features


# -------------------------------------------------------------------
# Appel API
# -------------------------------------------------------------------

def send_prediction(
    client: httpx.Client,
    features: dict[str, float | None],
) -> httpx.Response:
    """
    Envoie les features à l'API Render.
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
# Mise à jour de la vérité terrain
# -------------------------------------------------------------------

def update_actual_default(
    engine,
    request_id: str,
    actual_default: int,
) -> None:
    """
    Associe la vérité terrain TARGET à la prédiction
    enregistrée dans prediction_logs.
    """
    query = text(
        """
        UPDATE prediction_logs
        SET actual_default = :actual_default
        WHERE request_id = :request_id
        """
    )

    with engine.begin() as connection:
        result = connection.execute(
            query,
            {
                "actual_default": actual_default,
                "request_id": request_id,
            },
        )

    if result.rowcount != 1:
        raise RuntimeError(
            "Impossible d'associer TARGET au request_id "
            f"{request_id}. "
            f"Lignes modifiées : {result.rowcount}"
        )


# -------------------------------------------------------------------
# Programme principal
# -------------------------------------------------------------------

def main() -> None:
    dataframe = load_labelled_data()

    print(
        "\n=== MONITORING SUPERVISE P6 -> API P8 ===\n"
    )

    print(
        "Dataset :",
        dataframe.shape,
    )

    print(
        "TARGET = 0 :",
        int(
            (
                dataframe[TARGET_COLUMN] == 0
            ).sum()
        ),
    )

    print(
        "TARGET = 1 :",
        int(
            (
                dataframe[TARGET_COLUMN] == 1
            ).sum()
        ),
    )

    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        pool_pre_ping=True,
    )

    timeout = httpx.Timeout(
        REQUEST_TIMEOUT_SECONDS
    )

    successes = 0
    failures = 0
    labels_updated = 0

    try:
        with httpx.Client(
            timeout=timeout
        ) as client:

            for index, (_, row) in enumerate(
                dataframe.iterrows(),
                start=1,
            ):
                target = int(
                    row[TARGET_COLUMN]
                )

                features = build_features_payload(
                    row
                )

                try:
                    response = send_prediction(
                        client=client,
                        features=features,
                    )

                except httpx.HTTPError as error:
                    failures += 1

                    print(
                        f"[{index}/{len(dataframe)}] "
                        f"HTTP ERROR | {error}"
                    )

                    continue

                try:
                    payload = response.json()

                except ValueError:
                    failures += 1

                    print(
                        f"[{index}/{len(dataframe)}] "
                        "Réponse non JSON"
                    )

                    continue

                if response.status_code != 200:
                    failures += 1

                    print(
                        f"[{index}/{len(dataframe)}] "
                        f"{response.status_code} | "
                        f"{payload}"
                    )

                    continue

                request_id = payload.get(
                    "request_id"
                )

                if not request_id:
                    failures += 1

                    print(
                        f"[{index}/{len(dataframe)}] "
                        "request_id absent"
                    )

                    continue

                try:
                    update_actual_default(
                        engine=engine,
                        request_id=request_id,
                        actual_default=target,
                    )

                except Exception as error:
                    failures += 1

                    print(
                        f"[{index}/{len(dataframe)}] "
                        f"UPDATE ERROR | {error}"
                    )

                    continue

                successes += 1
                labels_updated += 1

                probability = payload.get(
                    "probability_default"
                )

                prediction = payload.get(
                    "prediction"
                )

                score_display = (
                    f"{float(probability):.4f}"
                    if probability is not None
                    else "N/A"
                )

                print(
                    f"[{index}/{len(dataframe)}] "
                    f"200 | "
                    f"TARGET={target} | "
                    f"prediction={prediction} | "
                    f"score={score_display}"
                )

    finally:
        engine.dispose()

    print(
        "\n=== RESUME MONITORING SUPERVISE ===\n"
    )

    print(
        "Observations :",
        len(dataframe),
    )

    print(
        "Prédictions + labels réussis :",
        successes,
    )

    print(
        "Échecs :",
        failures,
    )

    print(
        "actual_default renseignés :",
        labels_updated,
    )

    if successes > 0:
        success_rate = (
            successes / len(dataframe)
        )

        print(
            "Taux de succès :",
            f"{success_rate * 100:.2f} %",
        )

    if successes == len(dataframe):
        print(
            "\nLes 1000 observations ont été "
            "traitées et labellisées dans Supabase."
        )
        
if __name__ == "__main__":
    main()