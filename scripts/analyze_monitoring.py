import os

import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "La variable d'environnement DATABASE_URL n'est pas définie."
    )

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "La variable d'environnement DATABASE_URL n'est pas définie."
    )

SQLALCHEMY_DATABASE_URL = DATABASE_URL.replace(
    "postgresql://",
    "postgresql+psycopg://",
    1,
)

QUERY = """
SELECT
    request_id,
    created_at,
    probability_default,
    prediction,
    prediction_label,
    threshold,
    latency_ms,
    status_code,
    error_message,
    model_name,
    model_version,
    actual_default
FROM prediction_logs
ORDER BY created_at;
"""

def load_monitoring_data() -> pd.DataFrame:
    """
    Charge les logs de production depuis Supabase PostgreSQL.
    """
    engine = create_engine (SQLALCHEMY_DATABASE_URL)

    with engine.connect() as connection:
        dataframe = pd.read_sql_query(
            QUERY,
            connection,
        )

    return dataframe

def display_operational_metrics(dataframe: pd.DataFrame) -> None:
    """
    Calcule les principales métriques opérationnelles de l'API.
    """
    total_requests = len(dataframe)

    if total_requests == 0:
        print("Aucune donnée de monitoring disponible.")
        return

    successful_requests = (
        dataframe["status_code"] == 200
    ).sum()

    error_requests = (
        dataframe["status_code"] != 200
    ).sum()

    success_rate = (
        successful_requests / total_requests
    ) * 100

    error_rate = (
        error_requests / total_requests
    ) * 100

    print("\n=== MONITORING API ===\n")

    print(f"Nombre total d'appels : {total_requests}")
    print(f"Appels réussis : {successful_requests}")
    print(f"Appels en erreur : {error_requests}")
    print(f"Taux de succès : {success_rate:.2f} %")
    print(f"Taux d'erreur : {error_rate:.2f} %")

    print("\n--- Codes HTTP ---")

    print(
        dataframe["status_code"]
        .value_counts()
        .sort_index()
    )

    print("\n--- Latence ---")

    latency = dataframe["latency_ms"].dropna()

    print(
        f"Latence moyenne : "
        f"{latency.mean():.2f} ms"
    )

    print(
        f"Latence médiane : "
        f"{latency.median():.2f} ms"
    )

    print(
        f"Latence p95 : "
        f"{latency.quantile(0.95):.2f} ms"
    )

    print(
        f"Latence maximale : "
        f"{latency.max():.2f} ms"
    )


def display_prediction_metrics(
    dataframe: pd.DataFrame,
) -> None:
    """
    Analyse la distribution des scores et des classes prédites.
    """
    successful_predictions = dataframe[
        dataframe["status_code"] == 200
    ].copy()

    if successful_predictions.empty:
        print(
            "\nAucune prédiction réussie disponible."
        )
        return

    scores = successful_predictions[
        "probability_default"
    ].dropna()

    print("\n=== SCORES DE PREDICTION ===\n")

    print(
        f"Score moyen : "
        f"{scores.mean():.4f}"
    )

    print(
        f"Score médian : "
        f"{scores.median():.4f}"
    )

    print(
        f"Score minimum : "
        f"{scores.min():.4f}"
    )

    print(
        f"Score maximum : "
        f"{scores.max():.4f}"
    )

    print("\n--- Classes prédites ---")

    print(
        successful_predictions[
            "prediction_label"
        ].value_counts()
    )


def display_supervised_metrics(
    dataframe: pd.DataFrame,
) -> None:
    """
    Calcule les performances du modèle lorsque la vérité terrain
    actual_default est disponible.
    """
    labelled_data = dataframe[
        (dataframe["status_code"] == 200)
        & dataframe["actual_default"].notna()
        & dataframe["prediction"].notna()
    ].copy()

    if labelled_data.empty:
        print(
            "\n=== PERFORMANCE DU MODELE ===\n"
        )
        print(
            "Aucune vérité terrain disponible pour le moment."
        )
        print(
            "Accuracy, precision, recall, F1 et matrice "
            "de confusion ne peuvent pas encore être calculés."
        )
        return

    y_true = labelled_data[
        "actual_default"
    ].astype(int)

    y_pred = labelled_data[
        "prediction"
    ].astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    print("\n=== PERFORMANCE DU MODELE ===\n")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\n--- Matrice de confusion ---")

    print(f"TP : {tp}")
    print(f"TN : {tn}")
    print(f"FP : {fp}")
    print(f"FN : {fn}")


def main():
    dataframe = load_monitoring_data()

    display_operational_metrics(dataframe)
    display_prediction_metrics(dataframe)
    display_supervised_metrics(dataframe)


if __name__ == "__main__":
    main()