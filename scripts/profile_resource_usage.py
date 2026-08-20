import json
import os
import time
from pathlib import Path

import pandas as pd
import psutil

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

REPORT_PATH = (
    REPORTS_DIR
    / "resource_usage_profile.json"
)

SAMPLE_SIZE = 200
WARMUP_RUNS = 20


# -------------------------------------------------------------------
# Chargement
# -------------------------------------------------------------------

def load_sample() -> pd.DataFrame:
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
            "Le nombre de features ne correspond "
            "pas au modèle."
        )

    return dataframe


# -------------------------------------------------------------------
# Conversion
# -------------------------------------------------------------------

def row_to_features(
    row: pd.Series,
) -> dict[str, float | None]:
    return {
        column: (
            None
            if pd.isna(value)
            else float(value)
        )
        for column, value in row.items()
    }


# -------------------------------------------------------------------
# Profiling ressources
# -------------------------------------------------------------------

def measure_resource_usage(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Mesure le temps d'inférence, l'utilisation CPU
    du processus et la mémoire RSS.
    """
    process = psutil.Process(
        os.getpid()
    )

    # ---------------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------------

    for _, row in dataframe.head(
        WARMUP_RUNS
    ).iterrows():
        model_service.predict_proba(
            row_to_features(
                row
            )
        )

    memory_before_mb = (
        process.memory_info().rss
        / 1024
        / 1024
    )

    cpu_times_before = (
        process.cpu_times()
    )

    start_time = time.perf_counter()

    # ---------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------

    for _, row in dataframe.iterrows():
        model_service.predict_proba(
            row_to_features(
                row
            )
        )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    cpu_times_after = (
        process.cpu_times()
    )

    memory_after_mb = (
        process.memory_info().rss
        / 1024
        / 1024
    )

    cpu_seconds = (
        (
            cpu_times_after.user
            - cpu_times_before.user
        )
        +
        (
            cpu_times_after.system
            - cpu_times_before.system
        )
    )

    cpu_utilization_estimate = (
        cpu_seconds
        / elapsed_seconds
        * 100
        if elapsed_seconds > 0
        else 0.0
    )

    mean_latency_ms = (
        elapsed_seconds
        / len(dataframe)
        * 1000
    )

    return {
        "observations": int(
            len(dataframe)
        ),
        "elapsed_seconds": round(
            elapsed_seconds,
            4,
        ),
        "mean_latency_ms": round(
            mean_latency_ms,
            4,
        ),
        "cpu_time_seconds": round(
            cpu_seconds,
            4,
        ),
        "cpu_utilization_estimate_percent": round(
            cpu_utilization_estimate,
            2,
        ),
        "memory_before_mb": round(
            memory_before_mb,
            2,
        ),
        "memory_after_mb": round(
            memory_after_mb,
            2,
        ),
        "memory_delta_mb": round(
            memory_after_mb
            - memory_before_mb,
            2,
        ),
        "logical_cpu_count": (
            psutil.cpu_count(
                logical=True
            )
        ),
        "physical_cpu_count": (
            psutil.cpu_count(
                logical=False
            )
        ),
        "gpu_used": False,
        "hardware_context": (
            "CPU inference. GPU not used."
        ),
    }


# -------------------------------------------------------------------
# Programme principal
# -------------------------------------------------------------------

def main() -> None:
    print(
        "\n=== PROFILING CPU / MEMOIRE ===\n"
    )

    dataframe = load_sample()

    print(
        "Observations :",
        len(dataframe),
    )

    model_service.load()

    report = measure_resource_usage(
        dataframe
    )

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
        "Temps total :",
        f"{report['elapsed_seconds']:.4f} s",
    )

    print(
        "Latence moyenne :",
        f"{report['mean_latency_ms']:.4f} ms",
    )

    print(
        "CPU estimé :",
        (
            f"{report['cpu_utilization_estimate_percent']:.2f} %"
        ),
    )

    print(
        "Mémoire avant :",
        f"{report['memory_before_mb']:.2f} MB",
    )

    print(
        "Mémoire après :",
        f"{report['memory_after_mb']:.2f} MB",
    )

    print(
        "Delta mémoire :",
        f"{report['memory_delta_mb']:.2f} MB",
    )

    print(
        "CPU logiques :",
        report["logical_cpu_count"],
    )

    print(
        "GPU utilisé :",
        report["gpu_used"],
    )

    print(
        "\nRapport :",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()