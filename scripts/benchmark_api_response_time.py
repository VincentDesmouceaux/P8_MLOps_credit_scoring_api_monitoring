import json
import os
import time
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

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH = (
    REPORTS_DIR
    / "api_response_time_benchmark.json"
)

SAMPLE_SIZE = 100
WARMUP_RUNS = 5
REQUEST_TIMEOUT_SECONDS = 60.0
EXPECTED_FEATURE_COUNT = 656


# -------------------------------------------------------------------
# Chargement des données
# -------------------------------------------------------------------

def load_sample() -> pd.DataFrame:
    """
    Charge un échantillon de vraies observations P6.
    """
    if not PRODUCTION_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {PRODUCTION_FILE}"
        )

    dataframe = pd.read_csv(
        PRODUCTION_FILE,
        nrows=SAMPLE_SIZE + WARMUP_RUNS,
    )

    if dataframe.empty:
        raise RuntimeError(
            "Le dataset de benchmark est vide."
        )

    if len(dataframe.columns) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "Le dataset ne contient pas exactement "
            f"{EXPECTED_FEATURE_COUNT} features."
        )

    return dataframe


# -------------------------------------------------------------------
# Conversion JSON
# -------------------------------------------------------------------

def build_features_payload(
    row: pd.Series,
) -> dict[str, float | None]:
    """
    Convertit une ligne pandas en payload JSON-compatible.
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
# Appel API
# -------------------------------------------------------------------

def send_prediction(
    client: httpx.Client,
    features: dict[str, float | None],
) -> httpx.Response:
    """
    Envoie une observation à l'API Render.
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
# Résumé statistique
# -------------------------------------------------------------------

def summarize_latencies(
    latencies_ms: list[float],
) -> dict:
    """
    Calcule les statistiques de temps de réponse HTTP.
    """
    series = pd.Series(
        latencies_ms
    )

    return {
        "mean_ms": round(
            float(series.mean()),
            3,
        ),
        "median_ms": round(
            float(series.median()),
            3,
        ),
        "p90_ms": round(
            float(series.quantile(0.90)),
            3,
        ),
        "p95_ms": round(
            float(series.quantile(0.95)),
            3,
        ),
        "p99_ms": round(
            float(series.quantile(0.99)),
            3,
        ),
        "min_ms": round(
            float(series.min()),
            3,
        ),
        "max_ms": round(
            float(series.max()),
            3,
        ),
        "std_ms": round(
            float(series.std()),
            3,
        ),
    }


# -------------------------------------------------------------------
# Programme principal
# -------------------------------------------------------------------

def main() -> None:
    print(
        "\n=== BENCHMARK API RENDER ===\n"
    )

    dataframe = load_sample()

    print(
        "Dataset :",
        dataframe.shape,
    )

    print(
        "URL :",
        PREDICT_URL,
    )

    timeout = httpx.Timeout(
        REQUEST_TIMEOUT_SECONDS
    )

    response_times_ms = []
    status_codes = []
    request_ids = []

    successful_requests = 0
    failed_requests = 0

    with httpx.Client(
        timeout=timeout
    ) as client:

        # -----------------------------------------------------------
        # Warm-up
        # -----------------------------------------------------------

        print(
            f"Warm-up : {WARMUP_RUNS} appels..."
        )

        for _, row in dataframe.head(
            WARMUP_RUNS
        ).iterrows():
            features = build_features_payload(
                row
            )

            response = send_prediction(
                client=client,
                features=features,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    "Échec pendant le warm-up. "
                    f"Status={response.status_code}, "
                    f"réponse={response.text}"
                )

        print(
            "Warm-up terminé."
        )

        # -----------------------------------------------------------
        # Benchmark
        # -----------------------------------------------------------

        benchmark_df = dataframe.iloc[
            WARMUP_RUNS:
            WARMUP_RUNS + SAMPLE_SIZE
        ]

        for index, (_, row) in enumerate(
            benchmark_df.iterrows(),
            start=1,
        ):
            features = build_features_payload(
                row
            )

            start_time = time.perf_counter()

            try:
                response = send_prediction(
                    client=client,
                    features=features,
                )

            except httpx.HTTPError as error:
                failed_requests += 1

                print(
                    f"[{index}/{SAMPLE_SIZE}] "
                    f"Erreur HTTP : {error}"
                )

                continue

            response_time_ms = (
                time.perf_counter()
                - start_time
            ) * 1000

            status_codes.append(
                response.status_code
            )

            if response.status_code != 200:
                failed_requests += 1

                print(
                    f"[{index}/{SAMPLE_SIZE}] "
                    f"{response.status_code}"
                )

                continue

            successful_requests += 1

            response_times_ms.append(
                response_time_ms
            )

            try:
                payload = response.json()

            except ValueError:
                payload = {}

            request_id = payload.get(
                "request_id"
            )

            if request_id:
                request_ids.append(
                    request_id
                )

            if (
                index == 1
                or index % 10 == 0
                or index == SAMPLE_SIZE
            ):
                print(
                    f"[{index}/{SAMPLE_SIZE}] "
                    f"200 | "
                    f"response_time="
                    f"{response_time_ms:.2f} ms"
                )

    # ----------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------

    if not response_times_ms:
        raise RuntimeError(
            "Aucun appel réussi pendant le benchmark."
        )

    latency_summary = summarize_latencies(
        response_times_ms
    )

    success_rate = (
        successful_requests
        / SAMPLE_SIZE
    )

    # ----------------------------------------------------------------
    # Rapport
    # ----------------------------------------------------------------

    report = {
        "sample_size": SAMPLE_SIZE,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "success_rate": round(
            success_rate,
            4,
        ),
        "success_percentage": round(
            success_rate * 100,
            2,
        ),
        "response_time_ms": latency_summary,
        "request_ids_collected": len(
            request_ids
        ),
        "api_url": API_URL,
        "execution_context": (
            "Client local vers API FastAPI déployée sur Render."
        ),
        "inference_pipeline": (
            "NumPy float32 + "
            "XGBoost Booster.inplace_predict"
        ),
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # ----------------------------------------------------------------
    # Affichage
    # ----------------------------------------------------------------

    print(
        "\n=== RESULTATS API RENDER ===\n"
    )

    print(
        "Appels prévus :",
        SAMPLE_SIZE,
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
        "Taux de succès :",
        f"{success_rate * 100:.2f} %",
    )

    print(
        "\nTemps de réponse moyen :",
        f"{latency_summary['mean_ms']:.3f} ms",
    )

    print(
        "Temps médian :",
        f"{latency_summary['median_ms']:.3f} ms",
    )

    print(
        "p90 :",
        f"{latency_summary['p90_ms']:.3f} ms",
    )

    print(
        "p95 :",
        f"{latency_summary['p95_ms']:.3f} ms",
    )

    print(
        "p99 :",
        f"{latency_summary['p99_ms']:.3f} ms",
    )

    print(
        "Minimum :",
        f"{latency_summary['min_ms']:.3f} ms",
    )

    print(
        "Maximum :",
        f"{latency_summary['max_ms']:.3f} ms",
    )

    print(
        "\nRapport :",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()