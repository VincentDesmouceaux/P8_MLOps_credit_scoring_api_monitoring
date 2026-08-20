import cProfile
import io
import json
import pstats
import time
from pathlib import Path

import pandas as pd

from app.services.model_service import model_service


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

PROFILE_TEXT_PATH = (
    REPORTS_DIR
    / "model_inference_profile.txt"
)

PROFILE_SUMMARY_PATH = (
    REPORTS_DIR
    / "model_inference_profile_summary.json"
)

SAMPLE_SIZE = 100
WARMUP_RUNS = 10


# -------------------------------------------------------------------
# Chargement des données
# -------------------------------------------------------------------

def load_sample() -> pd.DataFrame:
    """
    Charge un petit échantillon de données P6 réelles
    pour profiler l'inférence localement.
    """
    if not PRODUCTION_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {PRODUCTION_FILE}"
        )

    dataframe = pd.read_csv(
        PRODUCTION_FILE,
        nrows=SAMPLE_SIZE,
    )

    if dataframe.empty:
        raise RuntimeError(
            "Le dataset de profiling est vide."
        )

    if len(dataframe.columns) != len(
        model_service.feature_names
    ):
        raise RuntimeError(
            "Le nombre de features du dataset "
            "ne correspond pas au modèle."
        )

    return dataframe


# -------------------------------------------------------------------
# Conversion
# -------------------------------------------------------------------

def row_to_features(
    row: pd.Series,
) -> dict[str, float | None]:
    """
    Convertit une ligne pandas vers le format attendu
    par ModelService.
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
# Warm-up
# -------------------------------------------------------------------

def warmup(
    dataframe: pd.DataFrame,
) -> None:
    """
    Exécute quelques prédictions avant la mesure afin
    de réduire l'impact du chargement initial et des caches.
    """
    print(
        f"Warm-up : {WARMUP_RUNS} prédictions..."
    )

    for _, row in dataframe.head(
        WARMUP_RUNS
    ).iterrows():
        features = row_to_features(
            row
        )

        model_service.predict_proba(
            features
        )


# -------------------------------------------------------------------
# Benchmark simple
# -------------------------------------------------------------------

def benchmark_inference(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Mesure le temps d'inférence pour chaque observation.
    """
    latencies_ms = []

    for _, row in dataframe.iterrows():
        features = row_to_features(
            row
        )

        start_time = time.perf_counter()

        model_service.predict_proba(
            features
        )

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        latencies_ms.append(
            latency_ms
        )

    latency_series = pd.Series(
        latencies_ms
    )

    return {
        "observations": len(latencies_ms),
        "mean_latency_ms": round(
            float(latency_series.mean()),
            3,
        ),
        "median_latency_ms": round(
            float(latency_series.median()),
            3,
        ),
        "p90_latency_ms": round(
            float(
                latency_series.quantile(
                    0.90
                )
            ),
            3,
        ),
        "p95_latency_ms": round(
            float(
                latency_series.quantile(
                    0.95
                )
            ),
            3,
        ),
        "p99_latency_ms": round(
            float(
                latency_series.quantile(
                    0.99
                )
            ),
            3,
        ),
        "min_latency_ms": round(
            float(latency_series.min()),
            3,
        ),
        "max_latency_ms": round(
            float(latency_series.max()),
            3,
        ),
    }


# -------------------------------------------------------------------
# cProfile
# -------------------------------------------------------------------

def run_cprofile(
    dataframe: pd.DataFrame,
) -> str:
    """
    Profile le pipeline complet de prédiction locale.

    Le rapport est trié par temps cumulé afin d'identifier
    les fonctions consommant le plus de temps.
    """
    profiler = cProfile.Profile()

    profiler.enable()

    for _, row in dataframe.iterrows():
        features = row_to_features(
            row
        )

        model_service.predict_proba(
            features
        )

    profiler.disable()

    stream = io.StringIO()

    stats = pstats.Stats(
        profiler,
        stream=stream,
    )

    stats.strip_dirs()
    stats.sort_stats(
        "cumulative"
    )
    stats.print_stats(
        40
    )

    return stream.getvalue()


# -------------------------------------------------------------------
# Sauvegarde
# -------------------------------------------------------------------

def save_results(
    summary: dict,
    profile_text: str,
) -> None:
    """
    Sauvegarde les résultats du profiling.
    """
    with open(
        PROFILE_SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    with open(
        PROFILE_TEXT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            profile_text
        )


# -------------------------------------------------------------------
# Affichage
# -------------------------------------------------------------------

def display_summary(
    summary: dict,
) -> None:
    print(
        "\n=== PROFILING INFERENCE MODELE ===\n"
    )

    print(
        "Observations :",
        summary["observations"],
    )

    print(
        "Latence moyenne :",
        f"{summary['mean_latency_ms']:.3f} ms",
    )

    print(
        "Latence médiane :",
        f"{summary['median_latency_ms']:.3f} ms",
    )

    print(
        "Latence p90 :",
        f"{summary['p90_latency_ms']:.3f} ms",
    )

    print(
        "Latence p95 :",
        f"{summary['p95_latency_ms']:.3f} ms",
    )

    print(
        "Latence p99 :",
        f"{summary['p99_latency_ms']:.3f} ms",
    )

    print(
        "Latence minimale :",
        f"{summary['min_latency_ms']:.3f} ms",
    )

    print(
        "Latence maximale :",
        f"{summary['max_latency_ms']:.3f} ms",
    )

    print(
        "\nRésumé JSON :",
        PROFILE_SUMMARY_PATH,
    )

    print(
        "Rapport cProfile :",
        PROFILE_TEXT_PATH,
    )


# -------------------------------------------------------------------
# Programme principal
# -------------------------------------------------------------------

def main() -> None:
    print(
        "\n=== PREPARATION DU PROFILING ===\n"
    )

    dataframe = load_sample()

    print(
        "Dataset :",
        dataframe.shape,
    )

    print(
        "Chargement du modèle..."
    )

    model_service.load()

    print(
        "Modèle chargé."
    )

    warmup(
        dataframe
    )

    summary = benchmark_inference(
        dataframe
    )

    profile_text = run_cprofile(
        dataframe
    )

    save_results(
        summary=summary,
        profile_text=profile_text,
    )

    display_summary(
        summary
    )

    print(
        "\n=== TOP cProfile ===\n"
    )

    print(
        profile_text
    )


if __name__ == "__main__":
    main()