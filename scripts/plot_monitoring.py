from pathlib import Path

import matplotlib.pyplot as plt
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


QUERY = """
SELECT
    created_at,
    probability_default,
    prediction,
    prediction_label,
    latency_ms,
    status_code
FROM prediction_logs
ORDER BY created_at;
"""


# -------------------------------------------------------------------
# Chargement des données
# -------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Charge les données de monitoring depuis Supabase/PostgreSQL.
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
# Distribution des scores
# -------------------------------------------------------------------

def plot_score_distribution(
    dataframe: pd.DataFrame,
) -> None:
    """
    Génère l'histogramme des probabilités de défaut
    pour les prédictions ayant retourné HTTP 200.
    """
    scores = dataframe.loc[
        dataframe["status_code"] == 200,
        "probability_default",
    ].dropna()

    if scores.empty:
        raise RuntimeError(
            "Aucun score de prédiction disponible."
        )

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        scores,
        bins=10,
    )

    plt.axvline(
        0.45,
        linestyle="--",
        label="Seuil de décision = 0.45",
    )

    plt.xlabel(
        "Probabilité de défaut"
    )

    plt.ylabel(
        "Nombre de prédictions"
    )

    plt.title(
        "Distribution des scores de prédiction"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "score_distribution.png",
        dpi=150,
    )

    plt.close()


# -------------------------------------------------------------------
# Latence API
# -------------------------------------------------------------------

def plot_latency(
    dataframe: pd.DataFrame,
) -> None:
    """
    Génère l'évolution de la latence des appels API.
    """
    latency = dataframe[
        "latency_ms"
    ].dropna()

    if latency.empty:
        raise RuntimeError(
            "Aucune mesure de latence disponible."
        )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        range(
            1,
            len(latency) + 1,
        ),
        latency,
        marker="o",
    )

    plt.xlabel(
        "Numéro de requête"
    )

    plt.ylabel(
        "Latence (ms)"
    )

    plt.title(
        "Latence des appels API"
    )

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "api_latency.png",
        dpi=150,
    )

    plt.close()


# -------------------------------------------------------------------
# Codes HTTP
# -------------------------------------------------------------------

def plot_status_codes(
    dataframe: pd.DataFrame,
) -> None:
    """
    Génère la répartition des codes HTTP observés.
    """
    status_counts = (
        dataframe["status_code"]
        .value_counts()
        .sort_index()
    )

    if status_counts.empty:
        raise RuntimeError(
            "Aucun code HTTP disponible."
        )

    plt.figure(
        figsize=(8, 5)
    )

    status_counts.plot(
        kind="bar"
    )

    plt.xlabel(
        "Code HTTP"
    )

    plt.ylabel(
        "Nombre d'appels"
    )

    plt.title(
        "Répartition des codes HTTP"
    )

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "http_status_codes.png",
        dpi=150,
    )

    plt.close()


# -------------------------------------------------------------------
# Programme principal
# -------------------------------------------------------------------

def main() -> None:
    dataframe = load_data()

    print(
        "Données chargées :",
        dataframe.shape,
    )

    plot_score_distribution(
        dataframe
    )

    plot_latency(
        dataframe
    )

    plot_status_codes(
        dataframe
    )

    print(
        "\nGraphiques générés dans reports/"
    )

    print(
        "- reports/score_distribution.png"
    )

    print(
        "- reports/api_latency.png"
    )

    print(
        "- reports/http_status_codes.png"
    )


if __name__ == "__main__":
    main()