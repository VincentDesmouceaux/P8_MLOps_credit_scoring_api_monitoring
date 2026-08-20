import json
import time
from pathlib import Path

import numpy as np
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

REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH = (
    REPORTS_DIR
    / "xgboost_numpy_benchmark.json"
)

SAMPLE_SIZE = 500
WARMUP_RUNS = 20


# -------------------------------------------------------------------
# Chargement
# -------------------------------------------------------------------

def load_sample() -> pd.DataFrame:
    """
    Charge un échantillon P6 dans l'ordre exact
    des features attendu par le modèle.
    """
    dataframe = pd.read_csv(
        PRODUCTION_FILE,
        nrows=SAMPLE_SIZE,
    )

    expected_features = (
        model_service.feature_names
    )

    if (
        dataframe.columns.tolist()
        != expected_features
    ):
        raise RuntimeError(
            "Les features ne sont pas dans "
            "l'ordre attendu par le modèle."
        )

    return dataframe


# -------------------------------------------------------------------
# Conversion
# -------------------------------------------------------------------

def row_to_features(
    row: pd.Series,
) -> dict[str, float | None]:
    """
    Format utilisé actuellement par ModelService.
    """
    return {
        column: (
            None
            if pd.isna(value)
            else float(value)
        )
        for column, value in row.items()
    }


def row_to_numpy(
    row: pd.Series,
) -> np.ndarray:
    """
    Transforme directement une observation P6
    en tableau NumPy 2D.

    Les valeurs manquantes restent sous forme np.nan.
    """
    values = row.to_numpy(
        dtype=np.float32,
        copy=True,
    )

    return values.reshape(
        1,
        -1,
    )


# -------------------------------------------------------------------
# Prédiction V1
# -------------------------------------------------------------------

def predict_current(
    row: pd.Series,
) -> float:
    """
    Pipeline actuel :
    dict -> DataFrame -> predict_proba.
    """
    features = row_to_features(
        row
    )

    return model_service.predict_proba(
        features
    )


# -------------------------------------------------------------------
# Prédiction V2
# -------------------------------------------------------------------

def predict_numpy(
    row: pd.Series,
) -> float:
    """
    Pipeline candidat :
    NumPy -> XGBoost Booster.inplace_predict.
    """
    input_array = row_to_numpy(
        row
    )

    booster = (
        model_service.model.get_booster()
    )

    prediction = booster.inplace_predict(
        input_array
    )

    return float(
        prediction[0]
    )


# -------------------------------------------------------------------
# Benchmark
# -------------------------------------------------------------------

def benchmark(
    dataframe: pd.DataFrame,
    predict_function,
) -> tuple[list[float], list[float]]:
    """
    Retourne les prédictions et les latences.
    """
    predictions = []
    latencies = []

    for _, row in dataframe.iterrows():
        start = time.perf_counter()

        prediction = predict_function(
            row
        )

        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000

        predictions.append(
            prediction
        )

        latencies.append(
            latency_ms
        )

    return (
        predictions,
        latencies,
    )


def summarize(
    latencies: list[float],
) -> dict:
    """
    Calcule les statistiques principales.
    """
    series = pd.Series(
        latencies
    )

    return {
        "mean_ms": round(
            float(series.mean()),
            4,
        ),
        "median_ms": round(
            float(series.median()),
            4,
        ),
        "p90_ms": round(
            float(series.quantile(0.90)),
            4,
        ),
        "p95_ms": round(
            float(series.quantile(0.95)),
            4,
        ),
        "p99_ms": round(
            float(series.quantile(0.99)),
            4,
        ),
        "min_ms": round(
            float(series.min()),
            4,
        ),
        "max_ms": round(
            float(series.max()),
            4,
        ),
    }


# -------------------------------------------------------------------
# Programme principal
# -------------------------------------------------------------------

def main() -> None:
    print(
        "\n=== BENCHMARK XGBOOST : "
        "PANDAS VS NUMPY ===\n"
    )

    dataframe = load_sample()

    print(
        "Observations :",
        len(dataframe),
    )

    print(
        "Features :",
        len(dataframe.columns),
    )

    # ---------------------------------------------------------------
    # Chargement modèle
    # ---------------------------------------------------------------

    model_service.load()

    # ---------------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------------

    warmup_df = dataframe.head(
        WARMUP_RUNS
    )

    for _, row in warmup_df.iterrows():
        predict_current(
            row
        )

        predict_numpy(
            row
        )

    print(
        "Warm-up terminé."
    )

    # ---------------------------------------------------------------
    # V1
    # ---------------------------------------------------------------

    print(
        "\nBenchmark pipeline actuel..."
    )

    (
        current_predictions,
        current_latencies,
    ) = benchmark(
        dataframe,
        predict_current,
    )

    current_summary = summarize(
        current_latencies
    )

    # ---------------------------------------------------------------
    # V2
    # ---------------------------------------------------------------

    print(
        "Benchmark pipeline NumPy..."
    )

    (
        numpy_predictions,
        numpy_latencies,
    ) = benchmark(
        dataframe,
        predict_numpy,
    )

    numpy_summary = summarize(
        numpy_latencies
    )

    # ---------------------------------------------------------------
    # Vérification des prédictions
    # ---------------------------------------------------------------

    current_array = np.asarray(
        current_predictions
    )

    numpy_array = np.asarray(
        numpy_predictions
    )

    absolute_differences = np.abs(
        current_array
        - numpy_array
    )

    max_difference = float(
        absolute_differences.max()
    )

    mean_difference = float(
        absolute_differences.mean()
    )

    predictions_identical = bool(
        np.allclose(
            current_array,
            numpy_array,
            rtol=1e-6,
            atol=1e-7,
        )
    )

    # ---------------------------------------------------------------
    # Gain
    # ---------------------------------------------------------------

    current_mean = current_summary[
        "mean_ms"
    ]

    numpy_mean = numpy_summary[
        "mean_ms"
    ]

    speedup = (
        current_mean / numpy_mean
        if numpy_mean > 0
        else 0
    )

    improvement_percentage = (
        (
            current_mean
            - numpy_mean
        )
        / current_mean
        * 100
        if current_mean > 0
        else 0
    )

    # ---------------------------------------------------------------
    # Résultats
    # ---------------------------------------------------------------

    report = {
        "observations": len(
            dataframe
        ),
        "features": len(
            dataframe.columns
        ),
        "current_pipeline": current_summary,
        "numpy_pipeline": numpy_summary,
        "speedup": round(
            speedup,
            3,
        ),
        "improvement_percentage": round(
            improvement_percentage,
            2,
        ),
        "predictions_identical": (
            predictions_identical
        ),
        "max_probability_difference": (
            max_difference
        ),
        "mean_probability_difference": (
            mean_difference
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

    # ---------------------------------------------------------------
    # Affichage
    # ---------------------------------------------------------------

    print(
        "\n=== PIPELINE ACTUEL ===\n"
    )

    print(
        json.dumps(
            current_summary,
            indent=4,
        )
    )

    print(
        "\n=== PIPELINE NUMPY ===\n"
    )

    print(
        json.dumps(
            numpy_summary,
            indent=4,
        )
    )

    print(
        "\n=== COMPARAISON ===\n"
    )

    print(
        "Prédictions équivalentes :",
        predictions_identical,
    )

    print(
        "Différence maximale :",
        max_difference,
    )

    print(
        "Différence moyenne :",
        mean_difference,
    )

    print(
        "Speedup :",
        f"x{speedup:.2f}",
    )

    print(
        "Amélioration moyenne :",
        f"{improvement_percentage:.2f} %",
    )

    print(
        "\nRapport :",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()