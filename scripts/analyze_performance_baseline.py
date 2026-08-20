import json
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from app.core.database import get_database_url


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BASELINE_REPORT_PATH = (
    REPORTS_DIR
    / "performance_baseline.json"
)


DATABASE_URL = get_database_url()

SQLALCHEMY_DATABASE_URL = DATABASE_URL.replace(
    "postgresql://",
    "postgresql+psycopg://",
    1,
)


QUERY = """
SELECT
    created_at,
    latency_ms,
    probability_default,
    prediction,
    actual_default,
    status_code
FROM prediction_logs
WHERE
    status_code = 200
    AND actual_default IS NOT NULL
    AND latency_ms IS NOT NULL
ORDER BY created_at;
"""


# -------------------------------------------------------------------
# Chargement des données
# -------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Charge uniquement les prédictions P6 supervisées
    utilisées pour établir la baseline de performance.
    """
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        pool_pre_ping=True,
    )

    try:
        with engine.connect() as connection:
            dataframe = pd.read_sql_query(
                QUERY,
                connection,
            )

    finally:
        engine.dispose()

    if dataframe.empty:
        raise RuntimeError(
            "Aucune prédiction supervisée disponible."
        )

    return dataframe


# -------------------------------------------------------------------
# Calcul de la baseline
# -------------------------------------------------------------------

def compute_baseline(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Calcule les principales statistiques de latence
    observées sur les prédictions supervisées P6.
    """
    latency = dataframe[
        "latency_ms"
    ].astype(float)

    return {
        "observations": int(
            len(dataframe)
        ),
        "mean_latency_ms": round(
            float(latency.mean()),
            3,
        ),
        "median_latency_ms": round(
            float(latency.median()),
            3,
        ),
        "p90_latency_ms": round(
            float(
                latency.quantile(0.90)
            ),
            3,
        ),
        "p95_latency_ms": round(
            float(
                latency.quantile(0.95)
            ),
            3,
        ),
        "p99_latency_ms": round(
            float(
                latency.quantile(0.99)
            ),
            3,
        ),
        "min_latency_ms": round(
            float(latency.min()),
            3,
        ),
        "max_latency_ms": round(
            float(latency.max()),
            3,
        ),
        "std_latency_ms": round(
            float(latency.std()),
            3,
        ),
    }


# -------------------------------------------------------------------
# Sauvegarde
# -------------------------------------------------------------------

def save_report(
    report: dict,
) -> None:
    """
    Sauvegarde la baseline pour permettre une comparaison
    rigoureuse avec les futures versions optimisées.
    """
    with open(
        BASELINE_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )


# -------------------------------------------------------------------
# Affichage
# -------------------------------------------------------------------

def display_report(
    report: dict,
) -> None:
    print(
        "\n=== BASELINE PERFORMANCE P8 ===\n"
    )

    print(
        "Observations :",
        report["observations"],
    )

    print(
        "Latence moyenne :",
        f"{report['mean_latency_ms']:.3f} ms",
    )

    print(
        "Latence médiane :",
        f"{report['median_latency_ms']:.3f} ms",
    )

    print(
        "Latence p90 :",
        f"{report['p90_latency_ms']:.3f} ms",
    )

    print(
        "Latence p95 :",
        f"{report['p95_latency_ms']:.3f} ms",
    )

    print(
        "Latence p99 :",
        f"{report['p99_latency_ms']:.3f} ms",
    )

    print(
        "Latence minimale :",
        f"{report['min_latency_ms']:.3f} ms",
    )

    print(
        "Latence maximale :",
        f"{report['max_latency_ms']:.3f} ms",
    )

    print(
        "Écart-type :",
        f"{report['std_latency_ms']:.3f} ms",
    )

    print(
        "\nRapport :",
        BASELINE_REPORT_PATH,
    )


# -------------------------------------------------------------------
# Programme principal
# -------------------------------------------------------------------

def main() -> None:
    dataframe = load_data()

    report = compute_baseline(
        dataframe
    )

    save_report(
        report
    )

    display_report(
        report
    )


if __name__ == "__main__":
    main()