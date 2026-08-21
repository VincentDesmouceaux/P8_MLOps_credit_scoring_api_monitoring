import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
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

ONNX_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "onnx"
    / "credit_scoring_model.onnx"
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
    / "onnx_runtime_benchmark.json"
)

SAMPLE_SIZE = 500
WARMUP_RUNS = 20

EXPECTED_FEATURE_COUNT = 656


# -------------------------------------------------------------------
# Chargement des données
# -------------------------------------------------------------------

def load_sample() -> pd.DataFrame:
    """
    Charge un échantillon P6 dans l'ordre exact
    des features attendu par le modèle.
    """
    if not PRODUCTION_FILE.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {PRODUCTION_FILE}"
        )

    dataframe = pd.read_csv(
        PRODUCTION_FILE,
        nrows=SAMPLE_SIZE,
    )

    if dataframe.empty:
        raise RuntimeError(
            "Le dataset de benchmark est vide."
        )

    if (
        dataframe.columns.tolist()
        != model_service.feature_names
    ):
        raise RuntimeError(
            "Les features du dataset ne sont pas "
            "dans l'ordre attendu par le modèle."
        )

    return dataframe


# -------------------------------------------------------------------
# Conversion NumPy
# -------------------------------------------------------------------

def row_to_numpy(
    row: pd.Series,
) -> np.ndarray:
    """
    Convertit une observation P6 en float32.

    Les NaN sont conservés afin de reproduire le comportement
    du modèle XGBoost original.
    """
    values = row.to_numpy(
        dtype=np.float32,
        copy=True,
    )

    if values.shape[0] != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "Nombre de features incorrect."
        )

    return values.reshape(
        1,
        -1,
    )


# -------------------------------------------------------------------
# Session ONNX
# -------------------------------------------------------------------

def create_onnx_session() -> ort.InferenceSession:
    """
    Initialise ONNX Runtime en CPU.

    L'utilisation CPU est volontaire afin de comparer les deux
    moteurs dans un contexte proche de l'environnement Render.
    """
    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle ONNX introuvable : "
            f"{ONNX_MODEL_PATH}"
        )

    session_options = ort.SessionOptions()

    return ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        sess_options=session_options,
        providers=[
            "CPUExecutionProvider",
        ],
    )


# -------------------------------------------------------------------
# Prédiction XGBoost
# -------------------------------------------------------------------

def predict_xgboost(
    row: pd.Series,
) -> float:
    """
    Pipeline optimisé actuellement retenu en production.
    """
    input_array = row_to_numpy(
        row
    )

    if model_service.booster is None:
        raise RuntimeError(
            "Booster XGBoost non chargé."
        )

    prediction = (
        model_service.booster.inplace_predict(
            input_array
        )
    )

    return float(
        prediction[0]
    )


# -------------------------------------------------------------------
# Prédiction ONNX
# -------------------------------------------------------------------

def inspect_onnx_outputs(
    session: ort.InferenceSession,
) -> None:
    """
    Affiche les entrées/sorties du modèle ONNX.

    Cela permet de vérifier le format réellement produit
    par le convertisseur avant d'interpréter les probabilités.
    """
    print(
        "\n=== STRUCTURE ONNX ===\n"
    )

    print(
        "Inputs :"
    )

    for input_info in session.get_inputs():
        print(
            "-",
            input_info.name,
            input_info.shape,
            input_info.type,
        )

    print(
        "\nOutputs :"
    )

    for output_info in session.get_outputs():
        print(
            "-",
            output_info.name,
            output_info.shape,
            output_info.type,
        )


def extract_onnx_probability(
    outputs: list,
) -> float:
    """
    Extrait la probabilité de la classe positive.

    Les convertisseurs XGBoost ONNX peuvent produire
    différents formats :
    - tableau de probabilités ;
    - liste de dictionnaires ;
    - sortie label + probabilities.
    """
    if not outputs:
        raise RuntimeError(
            "Aucune sortie ONNX."
        )

    # ---------------------------------------------------------------
    # Cherche d'abord une sortie de probabilités exploitable
    # ---------------------------------------------------------------

    for output in outputs:
        # Cas ndarray classique
        if isinstance(
            output,
            np.ndarray,
        ):
            if (
                output.ndim == 2
                and output.shape[0] == 1
                and output.shape[1] >= 2
            ):
                return float(
                    output[0][1]
                )

        # Cas ZipMap : [{0: p0, 1: p1}]
        if isinstance(
            output,
            list,
        ):
            if (
                len(output) == 1
                and isinstance(
                    output[0],
                    dict,
                )
            ):
                probabilities = output[0]

                if 1 in probabilities:
                    return float(
                        probabilities[1]
                    )

                if "1" in probabilities:
                    return float(
                        probabilities["1"]
                    )

    raise RuntimeError(
        "Impossible d'identifier la probabilité "
        "de la classe positive dans les sorties ONNX."
    )


def predict_onnx(
    session: ort.InferenceSession,
    input_name: str,
    row: pd.Series,
) -> float:
    """
    Exécute une prédiction avec ONNX Runtime.
    """
    input_array = row_to_numpy(
        row
    )

    outputs = session.run(
        None,
        {
            input_name: input_array,
        },
    )

    return extract_onnx_probability(
        outputs
    )


# -------------------------------------------------------------------
# Benchmark
# -------------------------------------------------------------------

def benchmark(
    dataframe: pd.DataFrame,
    predict_function,
) -> tuple[np.ndarray, list[float]]:
    """
    Exécute le benchmark d'une fonction de prédiction.
    """
    probabilities = []
    latencies = []

    for _, row in dataframe.iterrows():
        start_time = time.perf_counter()

        probability = predict_function(
            row
        )

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        probabilities.append(
            probability
        )

        latencies.append(
            latency_ms
        )

    return (
        np.asarray(
            probabilities,
            dtype=float,
        ),
        latencies,
    )


def summarize_latencies(
    latencies: list[float],
) -> dict:
    """
    Résume les performances temporelles.
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
        "\n=== BENCHMARK XGBOOST VS ONNX RUNTIME ===\n"
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
    # XGBoost
    # ---------------------------------------------------------------

    model_service.load()

    # ---------------------------------------------------------------
    # ONNX
    # ---------------------------------------------------------------

    session = create_onnx_session()

    inspect_onnx_outputs(
        session
    )

    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    # ---------------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------------

    print(
        f"\nWarm-up : {WARMUP_RUNS} observations..."
    )

    warmup_df = dataframe.head(
        WARMUP_RUNS
    )

    for _, row in warmup_df.iterrows():
        predict_xgboost(
            row
        )

        predict_onnx(
            session,
            input_name,
            row,
        )

    print(
        "Warm-up terminé."
    )

    # ---------------------------------------------------------------
    # Benchmark XGBoost
    # ---------------------------------------------------------------

    print(
        "\nBenchmark XGBoost..."
    )

    xgb_probabilities, xgb_latencies = benchmark(
        dataframe,
        predict_xgboost,
    )

    # ---------------------------------------------------------------
    # Benchmark ONNX
    # ---------------------------------------------------------------

    print(
        "Benchmark ONNX Runtime..."
    )

    onnx_probabilities, onnx_latencies = benchmark(
        dataframe,
        lambda row: predict_onnx(
            session,
            input_name,
            row,
        ),
    )

    xgb_summary = summarize_latencies(
        xgb_latencies
    )

    onnx_summary = summarize_latencies(
        onnx_latencies
    )

    # ---------------------------------------------------------------
    # Comparaison numérique
    # ---------------------------------------------------------------

    absolute_differences = np.abs(
        xgb_probabilities
        - onnx_probabilities
    )

    max_difference = float(
        absolute_differences.max()
    )

    mean_difference = float(
        absolute_differences.mean()
    )

    probabilities_equivalent = bool(
        np.allclose(
            xgb_probabilities,
            onnx_probabilities,
            rtol=1e-5,
            atol=1e-6,
        )
    )

    # ---------------------------------------------------------------
    # Classes métier
    # ---------------------------------------------------------------

    threshold = 0.45

    xgb_predictions = (
        xgb_probabilities
        >= threshold
    ).astype(int)

    onnx_predictions = (
        onnx_probabilities
        >= threshold
    ).astype(int)

    predictions_identical = bool(
        np.array_equal(
            xgb_predictions,
            onnx_predictions,
        )
    )

    # ---------------------------------------------------------------
    # Gain/perte
    # ---------------------------------------------------------------

    xgb_mean = xgb_summary[
        "mean_ms"
    ]

    onnx_mean = onnx_summary[
        "mean_ms"
    ]

    onnx_speedup = (
        xgb_mean / onnx_mean
        if onnx_mean > 0
        else 0.0
    )

    improvement_percentage = (
        (
            xgb_mean
            - onnx_mean
        )
        / xgb_mean
        * 100
        if xgb_mean > 0
        else 0.0
    )

    # ---------------------------------------------------------------
    # Rapport
    # ---------------------------------------------------------------

    report = {
        "observations": len(
            dataframe
        ),
        "features": len(
            dataframe.columns
        ),
        "xgboost": xgb_summary,
        "onnx_runtime": onnx_summary,
        "onnx_speedup_vs_xgboost": round(
            onnx_speedup,
            3,
        ),
        "improvement_percentage": round(
            improvement_percentage,
            2,
        ),
        "probabilities_equivalent": (
            probabilities_equivalent
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
        "decision_threshold": threshold,
        "onnx_provider": (
            session
            .get_providers()
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
        "\n=== XGBOOST NATIF ===\n"
    )

    print(
        json.dumps(
            xgb_summary,
            indent=4,
        )
    )

    print(
        "\n=== ONNX RUNTIME ===\n"
    )

    print(
        json.dumps(
            onnx_summary,
            indent=4,
        )
    )

    print(
        "\n=== COMPARAISON ===\n"
    )

    print(
        "Probabilités équivalentes :",
        probabilities_equivalent,
    )

    print(
        "Prédictions identiques :",
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
        "Speedup ONNX vs XGBoost :",
        f"x{onnx_speedup:.2f}",
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