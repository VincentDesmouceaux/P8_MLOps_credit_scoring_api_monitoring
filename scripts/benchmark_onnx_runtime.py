import json
import time
from pathlib import Path

import mlflow.xgboost
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

XGBOOST_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "credit_scoring_model"
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
DECISION_THRESHOLD = 0.45


# -------------------------------------------------------------------
# Chargement des données
# -------------------------------------------------------------------

def load_sample() -> pd.DataFrame:
    """
    Charge un échantillon de données P6.

    Les colonnes doivent correspondre exactement
    aux 656 features attendues par le modèle.
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

    if len(dataframe.columns) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "Le dataset ne contient pas exactement "
            f"{EXPECTED_FEATURE_COUNT} features. "
            f"Reçu : {len(dataframe.columns)}."
        )

    if (
        dataframe.columns.tolist()
        != model_service.feature_names
    ):
        raise RuntimeError(
            "Les features du dataset ne correspondent "
            "pas exactement à l'ordre attendu par le modèle."
        )

    return dataframe


# -------------------------------------------------------------------
# Conversion NumPy
# -------------------------------------------------------------------

def row_to_numpy(
    row: pd.Series,
) -> np.ndarray:
    """
    Convertit une observation pandas en tableau NumPy float32
    de shape (1, 656).

    Les NaN sont conservés afin de reproduire exactement
    le comportement attendu par XGBoost et ONNX Runtime.
    """
    values = row.to_numpy(
        dtype=np.float32,
        copy=True,
    )

    if values.shape != (
        EXPECTED_FEATURE_COUNT,
    ):
        raise RuntimeError(
            "Shape inattendue après conversion NumPy : "
            f"{values.shape}"
        )

    return values.reshape(
        1,
        EXPECTED_FEATURE_COUNT,
    )


# -------------------------------------------------------------------
# Chargement XGBoost de référence
# -------------------------------------------------------------------

def load_reference_xgboost():
    """
    Charge le modèle XGBoost original.

    Ce modèle n'est utilisé que comme référence
    pour comparer ONNX Runtime au modèle source.
    """
    if not XGBOOST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Modèle XGBoost de référence introuvable : "
            f"{XGBOOST_MODEL_PATH}"
        )

    model = mlflow.xgboost.load_model(
        str(XGBOOST_MODEL_PATH)
    )

    if model is None:
        raise RuntimeError(
            "Le modèle XGBoost de référence "
            "n'a pas pu être chargé."
        )

    try:
        booster = model.get_booster()

    except AttributeError as error:
        raise RuntimeError(
            "Impossible de récupérer le Booster "
            "du modèle XGBoost de référence."
        ) from error

    if booster is None:
        raise RuntimeError(
            "Booster XGBoost indisponible."
        )

    return booster


# -------------------------------------------------------------------
# Session ONNX de production
# -------------------------------------------------------------------

def get_production_onnx_session(
) -> tuple[
    ort.InferenceSession,
    str,
    str,
]:
    """
    Retourne exactement la session ONNX utilisée
    par ModelService en production.

    Le benchmark mesure donc la configuration réelle :
    - CPUExecutionProvider ;
    - ORT_SEQUENTIAL ;
    - intra_op_num_threads = 1 ;
    - inter_op_num_threads = 1.
    """
    model_service.load()

    if model_service.session is None:
        raise RuntimeError(
            "Session ONNX Runtime non chargée."
        )

    if model_service.input_name is None:
        raise RuntimeError(
            "Nom d'entrée ONNX indisponible."
        )

    if (
        model_service.probabilities_output_name
        is None
    ):
        raise RuntimeError(
            "Nom de sortie probabilities indisponible."
        )

    return (
        model_service.session,
        model_service.input_name,
        model_service.probabilities_output_name,
    )


# -------------------------------------------------------------------
# Inspection ONNX
# -------------------------------------------------------------------

def inspect_onnx_session(
    session: ort.InferenceSession,
) -> None:
    """
    Affiche les caractéristiques principales
    de la session ONNX Runtime.
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

    print(
        "\nProviders actifs :",
        session.get_providers(),
    )

    print(
        "Mode d'exécution : ORT_SEQUENTIAL"
    )

    print(
        "Threads intra-op : 1"
    )

    print(
        "Threads inter-op : 1"
    )


# -------------------------------------------------------------------
# Prédiction XGBoost
# -------------------------------------------------------------------

def predict_xgboost(
    booster,
    row: pd.Series,
) -> float:
    """
    Exécute une prédiction avec le Booster
    XGBoost de référence.
    """
    input_array = row_to_numpy(
        row
    )

    predictions = booster.inplace_predict(
        input_array
    )

    if len(predictions) != 1:
        raise RuntimeError(
            "Format inattendu retourné "
            "par XGBoost."
        )

    probability = float(
        predictions[0]
    )

    if not np.isfinite(
        probability
    ):
        raise RuntimeError(
            "La probabilité XGBoost "
            "n'est pas finie."
        )

    if not (
        0.0
        <= probability
        <= 1.0
    ):
        raise RuntimeError(
            "La probabilité XGBoost "
            "n'est pas comprise entre 0 et 1."
        )

    return probability


# -------------------------------------------------------------------
# Prédiction ONNX
# -------------------------------------------------------------------

def predict_onnx(
    session: ort.InferenceSession,
    input_name: str,
    probabilities_output_name: str,
    row: pd.Series,
) -> float:
    """
    Exécute une prédiction avec la session
    ONNX Runtime utilisée en production.
    """
    input_array = row_to_numpy(
        row
    )

    outputs = session.run(
        [
            probabilities_output_name,
        ],
        {
            input_name: input_array,
        },
    )

    if len(outputs) != 1:
        raise RuntimeError(
            "Nombre de sorties ONNX inattendu."
        )

    probabilities = outputs[0]

    if not isinstance(
        probabilities,
        np.ndarray,
    ):
        raise RuntimeError(
            "La sortie probabilities "
            "n'est pas un tableau NumPy."
        )

    if probabilities.shape != (
        1,
        2,
    ):
        raise RuntimeError(
            "Shape ONNX inattendue : "
            f"{probabilities.shape}"
        )

    probability = float(
        probabilities[0][1]
    )

    if not np.isfinite(
        probability
    ):
        raise RuntimeError(
            "La probabilité ONNX "
            "n'est pas finie."
        )

    if not (
        0.0
        <= probability
        <= 1.0
    ):
        raise RuntimeError(
            "La probabilité ONNX "
            "n'est pas comprise entre 0 et 1."
        )

    return probability


# -------------------------------------------------------------------
# Benchmark générique
# -------------------------------------------------------------------

def benchmark(
    dataframe: pd.DataFrame,
    predict_function,
) -> tuple[
    np.ndarray,
    list[float],
]:
    """
    Exécute un benchmark observation par observation.

    Le temps mesuré inclut :
    - conversion pandas -> NumPy ;
    - inférence ;
    - extraction de la probabilité.
    """
    probabilities = []
    latencies = []

    for _, row in dataframe.iterrows():
        start_time = (
            time.perf_counter()
        )

        probability = (
            predict_function(
                row
            )
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


# -------------------------------------------------------------------
# Statistiques
# -------------------------------------------------------------------

def summarize_latencies(
    latencies: list[float],
) -> dict:
    """
    Calcule les principales statistiques
    de latence du benchmark.
    """
    if not latencies:
        raise RuntimeError(
            "Aucune latence disponible."
        )

    series = pd.Series(
        latencies,
        dtype=float,
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
            float(
                series.quantile(
                    0.90
                )
            ),
            4,
        ),
        "p95_ms": round(
            float(
                series.quantile(
                    0.95
                )
            ),
            4,
        ),
        "p99_ms": round(
            float(
                series.quantile(
                    0.99
                )
            ),
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
        "std_ms": round(
            float(series.std()),
            4,
        ),
    }


# -------------------------------------------------------------------
# Programme principal
# -------------------------------------------------------------------

def main() -> None:
    print(
        "\n=== BENCHMARK XGBOOST VS "
        "ONNX RUNTIME MONO-THREAD ===\n"
    )

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------

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
    # XGBoost de référence
    # ---------------------------------------------------------------

    print(
        "\nChargement du modèle "
        "XGBoost de référence..."
    )

    xgboost_booster = (
        load_reference_xgboost()
    )

    print(
        "Modèle XGBoost de référence chargé."
    )

    # ---------------------------------------------------------------
    # ONNX de production
    # ---------------------------------------------------------------

    (
        onnx_session,
        input_name,
        probabilities_output_name,
    ) = get_production_onnx_session()

    inspect_onnx_session(
        onnx_session
    )

    # ---------------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------------

    print(
        f"\nWarm-up : "
        f"{WARMUP_RUNS} observations..."
    )

    warmup_df = dataframe.head(
        WARMUP_RUNS
    )

    for _, row in warmup_df.iterrows():
        predict_xgboost(
            xgboost_booster,
            row,
        )

        predict_onnx(
            onnx_session,
            input_name,
            probabilities_output_name,
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

    (
        xgb_probabilities,
        xgb_latencies,
    ) = benchmark(
        dataframe,
        lambda row: predict_xgboost(
            xgboost_booster,
            row,
        ),
    )

    # ---------------------------------------------------------------
    # Benchmark ONNX
    # ---------------------------------------------------------------

    print(
        "Benchmark ONNX Runtime mono-thread..."
    )

    (
        onnx_probabilities,
        onnx_latencies,
    ) = benchmark(
        dataframe,
        lambda row: predict_onnx(
            onnx_session,
            input_name,
            probabilities_output_name,
            row,
        ),
    )

    # ---------------------------------------------------------------
    # Résumés
    # ---------------------------------------------------------------

    xgb_summary = summarize_latencies(
        xgb_latencies
    )

    onnx_summary = summarize_latencies(
        onnx_latencies
    )

    # ---------------------------------------------------------------
    # Comparaison des probabilités
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
    # Comparaison des décisions métier
    # ---------------------------------------------------------------

    xgb_predictions = (
        xgb_probabilities
        >= DECISION_THRESHOLD
    ).astype(int)

    onnx_predictions = (
        onnx_probabilities
        >= DECISION_THRESHOLD
    ).astype(int)

    predictions_identical = bool(
        np.array_equal(
            xgb_predictions,
            onnx_predictions,
        )
    )

    differing_predictions = int(
        np.sum(
            xgb_predictions
            != onnx_predictions
        )
    )

    # ---------------------------------------------------------------
    # Calcul du gain
    # ---------------------------------------------------------------

    xgb_mean = xgb_summary[
        "mean_ms"
    ]

    onnx_mean = onnx_summary[
        "mean_ms"
    ]

    if onnx_mean > 0:
        speedup = (
            xgb_mean
            / onnx_mean
        )
    else:
        speedup = 0.0

    if xgb_mean > 0:
        improvement_percentage = (
            (
                xgb_mean
                - onnx_mean
            )
            / xgb_mean
            * 100
        )
    else:
        improvement_percentage = 0.0

    # ---------------------------------------------------------------
    # Rapport
    # ---------------------------------------------------------------

    report = {
        "observations": int(
            len(dataframe)
        ),
        "features": int(
            len(dataframe.columns)
        ),
        "decision_threshold": (
            DECISION_THRESHOLD
        ),
        "xgboost_reference": (
            xgb_summary
        ),
        "onnx_runtime_production": (
            onnx_summary
        ),
        "onnx_configuration": {
            "providers": (
                onnx_session.get_providers()
            ),
            "execution_mode": (
                "ORT_SEQUENTIAL"
            ),
            "intra_op_num_threads": 1,
            "inter_op_num_threads": 1,
        },
        "onnx_speedup_vs_xgboost": round(
            speedup,
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
        "differing_predictions": (
            differing_predictions
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
    # Affichage XGBoost
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

    # ---------------------------------------------------------------
    # Affichage ONNX
    # ---------------------------------------------------------------

    print(
        "\n=== ONNX RUNTIME "
        "MONO-THREAD ===\n"
    )

    print(
        json.dumps(
            onnx_summary,
            indent=4,
        )
    )

    # ---------------------------------------------------------------
    # Comparaison
    # ---------------------------------------------------------------

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
        "Prédictions différentes :",
        differing_predictions,
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
        f"x{speedup:.2f}",
    )

    print(
        "Amélioration moyenne :",
        f"{improvement_percentage:.2f} %",
    )

    # ---------------------------------------------------------------
    # Configuration ONNX
    # ---------------------------------------------------------------

    print(
        "\n=== CONFIGURATION ONNX ===\n"
    )

    print(
        "Provider :",
        onnx_session.get_providers(),
    )

    print(
        "Execution mode : ORT_SEQUENTIAL"
    )

    print(
        "Threads intra-op : 1"
    )

    print(
        "Threads inter-op : 1"
    )

    print(
        "\nRapport :",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()