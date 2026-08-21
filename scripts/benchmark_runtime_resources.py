import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import psutil

from app.services.model_service import model_service


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

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = (
    REPORTS_DIR
    / "runtime_resource_comparison.json"
)

SAMPLE_SIZE = 1000
WARMUP_RUNS = 50


def load_sample() -> pd.DataFrame:
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
            "L'ordre des features ne correspond pas au modèle."
        )

    return dataframe


def dataframe_to_numpy(
    dataframe: pd.DataFrame,
) -> np.ndarray:
    return dataframe.to_numpy(
        dtype=np.float32,
        copy=True,
    )


def create_onnx_session() -> ort.InferenceSession:
    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle ONNX introuvable : {ONNX_MODEL_PATH}"
        )

    return ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        providers=[
            "CPUExecutionProvider",
        ],
    )


def measure_runtime(
    *,
    name: str,
    predict_function,
) -> dict:
    process = psutil.Process(
        os.getpid()
    )

    memory_before = (
        process.memory_info().rss
        / 1024
        / 1024
    )

    cpu_before = process.cpu_times()

    start = time.perf_counter()

    predict_function()

    elapsed = (
        time.perf_counter()
        - start
    )

    cpu_after = process.cpu_times()

    memory_after = (
        process.memory_info().rss
        / 1024
        / 1024
    )

    cpu_seconds = (
        cpu_after.user
        - cpu_before.user
        + cpu_after.system
        - cpu_before.system
    )

    cpu_percent_estimate = (
        cpu_seconds
        / elapsed
        * 100
        if elapsed > 0
        else 0.0
    )

    return {
        "runtime": name,
        "elapsed_seconds": round(
            elapsed,
            6,
        ),
        "mean_latency_ms": round(
            elapsed
            / SAMPLE_SIZE
            * 1000,
            6,
        ),
        "cpu_time_seconds": round(
            cpu_seconds,
            6,
        ),
        "cpu_utilization_estimate_percent": round(
            cpu_percent_estimate,
            2,
        ),
        "memory_before_mb": round(
            memory_before,
            2,
        ),
        "memory_after_mb": round(
            memory_after,
            2,
        ),
        "memory_delta_mb": round(
            memory_after - memory_before,
            2,
        ),
    }


def main() -> None:
    print(
        "\n=== COMPARAISON CPU / MEMOIRE ===\n"
    )

    dataframe = load_sample()

    input_array = dataframe_to_numpy(
        dataframe
    )

    print(
        "Observations :",
        len(dataframe),
    )

    print(
        "Features :",
        input_array.shape[1],
    )

    model_service.load()

    if model_service.booster is None:
        raise RuntimeError(
            "Booster XGBoost non chargé."
        )

    onnx_session = create_onnx_session()

    input_name = (
        onnx_session
        .get_inputs()[0]
        .name
    )

    # Warm-up
    for _ in range(WARMUP_RUNS):
        model_service.booster.inplace_predict(
            input_array[:1]
        )

        onnx_session.run(
            None,
            {
                input_name: input_array[:1],
            },
        )

    def run_xgboost():
        for row in input_array:
            model_service.booster.inplace_predict(
                row.reshape(1, -1)
            )

    def run_onnx():
        for row in input_array:
            onnx_session.run(
                None,
                {
                    input_name: row.reshape(
                        1,
                        -1,
                    ),
                },
            )

    print(
        "\nMesure XGBoost..."
    )

    xgboost_result = measure_runtime(
        name="xgboost_inplace_predict",
        predict_function=run_xgboost,
    )

    print(
        "Mesure ONNX Runtime..."
    )

    onnx_result = measure_runtime(
        name="onnxruntime_cpu",
        predict_function=run_onnx,
    )

    report = {
        "observations": SAMPLE_SIZE,
        "features": input_array.shape[1],
        "hardware": {
            "logical_cpu_count": psutil.cpu_count(
                logical=True
            ),
            "physical_cpu_count": psutil.cpu_count(
                logical=False
            ),
            "gpu_used": False,
        },
        "xgboost": xgboost_result,
        "onnx_runtime": onnx_result,
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

    print(
        "\n=== XGBOOST ==="
    )

    print(
        json.dumps(
            xgboost_result,
            indent=4,
        )
    )

    print(
        "\n=== ONNX RUNTIME ==="
    )

    print(
        json.dumps(
            onnx_result,
            indent=4,
        )
    )

    print(
        "\nGPU utilisé : False"
    )

    print(
        "\nRapport :",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()