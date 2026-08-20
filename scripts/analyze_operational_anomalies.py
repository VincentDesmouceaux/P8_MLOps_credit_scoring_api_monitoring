import json
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from app.core.database import get_database_url


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

DATABASE_URL = get_database_url()

SQLALCHEMY_DATABASE_URL = DATABASE_URL.replace(
    "postgresql://",
    "postgresql+psycopg://",
    1,
)


REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ANOMALY_REPORT_PATH = (
    REPORTS_DIR / "operational_anomalies.json"
)


# -------------------------------------------------------------------
# Seuils de monitoring
# -------------------------------------------------------------------

ERROR_RATE_WARNING_THRESHOLD = 0.10
ERROR_RATE_CRITICAL_THRESHOLD = 0.20

LATENCY_P95_WARNING_MS = 150.0
LATENCY_P95_CRITICAL_MS = 250.0


# -------------------------------------------------------------------
# Requête SQL
# -------------------------------------------------------------------

QUERY = """
SELECT
    created_at,
    latency_ms,
    status_code,
    probability_default,
    prediction,
    prediction_label,
    error_message
FROM prediction_logs
ORDER BY created_at;
"""


# -------------------------------------------------------------------
# Chargement des logs
# -------------------------------------------------------------------

def load_monitoring_data() -> pd.DataFrame:
    """
    Charge les logs de production depuis Supabase/PostgreSQL.
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
            "Aucune donnée de monitoring disponible."
        )

    return dataframe


# -------------------------------------------------------------------
# Analyse du taux d'erreur
# -------------------------------------------------------------------

def compute_error_metrics(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Calcule le volume d'appels, le taux de succès,
    le taux d'erreur et le niveau d'alerte associé.
    """
    total_requests = len(
        dataframe
    )

    successful_requests = int(
        (
            dataframe["status_code"] == 200
        ).sum()
    )

    error_requests = int(
        (
            dataframe["status_code"] != 200
        ).sum()
    )

    success_rate = (
        successful_requests / total_requests
        if total_requests > 0
        else 0.0
    )

    error_rate = (
        error_requests / total_requests
        if total_requests > 0
        else 0.0
    )

    if error_rate >= ERROR_RATE_CRITICAL_THRESHOLD:
        error_status = "critical"

    elif error_rate >= ERROR_RATE_WARNING_THRESHOLD:
        error_status = "warning"

    else:
        error_status = "normal"

    return {
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "error_requests": error_requests,
        "success_rate": round(
            success_rate,
            4,
        ),
        "error_rate": round(
            error_rate,
            4,
        ),
        "success_percentage": round(
            success_rate * 100,
            2,
        ),
        "error_percentage": round(
            error_rate * 100,
            2,
        ),
        "error_status": error_status,
    }


# -------------------------------------------------------------------
# Analyse de la latence
# -------------------------------------------------------------------

def compute_latency_metrics(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Calcule les principales métriques de latence
    et détecte une éventuelle anomalie sur le p95.
    """
    latency = dataframe[
        "latency_ms"
    ].dropna()

    if latency.empty:
        raise RuntimeError(
            "Aucune mesure de latence disponible."
        )

    mean_latency = float(
        latency.mean()
    )

    median_latency = float(
        latency.median()
    )

    p95_latency = float(
        latency.quantile(
            0.95
        )
    )

    max_latency = float(
        latency.max()
    )

    if p95_latency >= LATENCY_P95_CRITICAL_MS:
        latency_status = "critical"

    elif p95_latency >= LATENCY_P95_WARNING_MS:
        latency_status = "warning"

    else:
        latency_status = "normal"

    return {
        "mean_latency_ms": round(
            mean_latency,
            2,
        ),
        "median_latency_ms": round(
            median_latency,
            2,
        ),
        "p95_latency_ms": round(
            p95_latency,
            2,
        ),
        "max_latency_ms": round(
            max_latency,
            2,
        ),
        "latency_status": latency_status,
    }


# -------------------------------------------------------------------
# Distribution des codes HTTP
# -------------------------------------------------------------------

def compute_status_code_distribution(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Retourne la distribution des codes HTTP observés.
    """
    counts = (
        dataframe["status_code"]
        .value_counts()
        .sort_index()
    )

    return {
        str(status_code): int(count)
        for status_code, count in counts.items()
    }


# -------------------------------------------------------------------
# Construction du rapport
# -------------------------------------------------------------------

def build_operational_report(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Construit le rapport global d'anomalies opérationnelles.
    """
    error_metrics = compute_error_metrics(
        dataframe
    )

    latency_metrics = compute_latency_metrics(
        dataframe
    )

    status_codes = compute_status_code_distribution(
        dataframe
    )

    anomalies_detected = (
        error_metrics["error_status"] != "normal"
        or latency_metrics["latency_status"] != "normal"
    )

    return {
        "error_metrics": error_metrics,
        "latency_metrics": latency_metrics,
        "status_codes": status_codes,
        "thresholds": {
            "error_rate_warning": (
                ERROR_RATE_WARNING_THRESHOLD
            ),
            "error_rate_critical": (
                ERROR_RATE_CRITICAL_THRESHOLD
            ),
            "latency_p95_warning_ms": (
                LATENCY_P95_WARNING_MS
            ),
            "latency_p95_critical_ms": (
                LATENCY_P95_CRITICAL_MS
            ),
        },
        "anomalies_detected": anomalies_detected,
        "data_context": (
            "Trafic de test simulé pour le PoC."
        ),
    }


# -------------------------------------------------------------------
# Sauvegarde
# -------------------------------------------------------------------

def save_operational_report(
    report: dict,
) -> None:
    """
    Sauvegarde le rapport opérationnel au format JSON.
    """
    with open(
        ANOMALY_REPORT_PATH,
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

def display_operational_report(
    report: dict,
) -> None:
    """
    Affiche une synthèse lisible dans le terminal.
    """
    error_metrics = report[
        "error_metrics"
    ]

    latency_metrics = report[
        "latency_metrics"
    ]

    print(
        "\n=== ANOMALIES OPERATIONNELLES ===\n"
    )

    print(
        "Nombre total d'appels :",
        error_metrics["total_requests"],
    )

    print(
        "Appels réussis :",
        error_metrics["successful_requests"],
    )

    print(
        "Appels en erreur :",
        error_metrics["error_requests"],
    )

    print(
        "Taux de succès :",
        f"{error_metrics['success_percentage']:.2f} %",
    )

    print(
        "Taux d'erreur :",
        f"{error_metrics['error_percentage']:.2f} %",
    )

    print(
        "État taux d'erreur :",
        error_metrics["error_status"].upper(),
    )

    print(
        "\n--- LATENCE ---"
    )

    print(
        "Latence moyenne :",
        f"{latency_metrics['mean_latency_ms']:.2f} ms",
    )

    print(
        "Latence médiane :",
        f"{latency_metrics['median_latency_ms']:.2f} ms",
    )

    print(
        "Latence p95 :",
        f"{latency_metrics['p95_latency_ms']:.2f} ms",
    )

    print(
        "Latence maximale :",
        f"{latency_metrics['max_latency_ms']:.2f} ms",
    )

    print(
        "État latence :",
        latency_metrics["latency_status"].upper(),
    )

    print(
        "\n--- CODES HTTP ---"
    )

    print(
        report["status_codes"]
    )

    print(
        "\nAnomalie opérationnelle détectée :",
        (
            "OUI"
            if report["anomalies_detected"]
            else "NON"
        ),
    )

    print(
        "\nRapport JSON :",
        ANOMALY_REPORT_PATH,
    )


# -------------------------------------------------------------------
# Programme principal
# -------------------------------------------------------------------

def main() -> None:
    dataframe = load_monitoring_data()

    report = build_operational_report(
        dataframe
    )

    save_operational_report(
        report
    )

    display_operational_report(
        report
    )


if __name__ == "__main__":
    main()