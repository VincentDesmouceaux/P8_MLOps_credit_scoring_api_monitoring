import json
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sqlalchemy import create_engine

from app.core.database import get_database_url


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="P8 MLOps Monitoring",
    page_icon="📊",
    layout="wide",
)


DATABASE_URL = get_database_url()

SQLALCHEMY_DATABASE_URL = DATABASE_URL.replace(
    "postgresql://",
    "postgresql+psycopg://",
    1,
)


REPORTS_DIR = Path("reports")

OPERATIONAL_REPORT_PATH = (
    REPORTS_DIR / "operational_anomalies.json"
)

DRIFT_SUMMARY_PATH = (
    REPORTS_DIR / "data_drift_summary.json"
)

DRIFT_FEATURES_PATH = (
    REPORTS_DIR / "data_drift_features.csv"
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


# -------------------------------------------------------------------
# Chargement des données
# -------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_production_data() -> pd.DataFrame:
    """
    Charge les données de monitoring depuis Supabase/PostgreSQL.

    Le cache est rafraîchi toutes les 60 secondes.
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

    return dataframe


@st.cache_data(ttl=60)
def load_json_report(
    path: Path,
) -> dict:
    """
    Charge un rapport JSON généré par les scripts de monitoring.
    """
    if not path.exists():
        return {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


@st.cache_data(ttl=60)
def load_drift_features() -> pd.DataFrame:
    """
    Charge le détail du Data Drift feature par feature.
    """
    if not DRIFT_FEATURES_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        DRIFT_FEATURES_PATH
    )


# -------------------------------------------------------------------
# Données
# -------------------------------------------------------------------

production_df = load_production_data()

operational_report = load_json_report(
    OPERATIONAL_REPORT_PATH
)

drift_summary = load_json_report(
    DRIFT_SUMMARY_PATH
)

drift_features_df = load_drift_features()


# -------------------------------------------------------------------
# En-tête
# -------------------------------------------------------------------

st.title(
    "P8 - Credit Scoring Monitoring"
)

st.caption(
    "Monitoring opérationnel, performance du modèle "
    "et détection du Data Drift."
)


if production_df.empty:
    st.error(
        "Aucune donnée de production disponible."
    )
    st.stop()


# -------------------------------------------------------------------
# Vue générale
# -------------------------------------------------------------------

st.header(
    "1. Vue générale de la production"
)


total_requests = len(
    production_df
)

successful_requests = int(
    (
        production_df["status_code"] == 200
    ).sum()
)

error_requests = int(
    (
        production_df["status_code"] != 200
    ).sum()
)

error_rate = (
    error_requests / total_requests
    if total_requests > 0
    else 0
)


successful_df = production_df[
    production_df["status_code"] == 200
].copy()


latency = production_df[
    "latency_ms"
].dropna()


mean_latency = (
    latency.mean()
    if not latency.empty
    else 0
)

p95_latency = (
    latency.quantile(0.95)
    if not latency.empty
    else 0
)


col1, col2, col3, col4 = st.columns(
    4
)

col1.metric(
    "Appels API",
    total_requests,
)

col2.metric(
    "Succès",
    successful_requests,
)

col3.metric(
    "Taux d'erreur",
    f"{error_rate * 100:.2f} %",
)

col4.metric(
    "Latence p95",
    f"{p95_latency:.2f} ms",
)


# -------------------------------------------------------------------
# Alertes opérationnelles
# -------------------------------------------------------------------

st.subheader(
    "État opérationnel"
)


if operational_report:
    error_metrics = operational_report.get(
        "error_metrics",
        {},
    )

    latency_metrics = operational_report.get(
        "latency_metrics",
        {},
    )

    error_status = error_metrics.get(
        "error_status",
        "unknown",
    )

    latency_status = latency_metrics.get(
        "latency_status",
        "unknown",
    )


    status_col1, status_col2 = st.columns(
        2
    )

    status_col1.metric(
        "État taux d'erreur",
        error_status.upper(),
    )

    status_col2.metric(
        "État latence",
        latency_status.upper(),
    )


    if operational_report.get(
        "anomalies_detected",
        False,
    ):
        st.error(
            "Une ou plusieurs anomalies opérationnelles "
            "ont été détectées."
        )

    else:
        st.success(
            "Aucune anomalie opérationnelle détectée."
        )


    with st.expander(
        "Voir les seuils d'alerte"
    ):
        st.json(
            operational_report.get(
                "thresholds",
                {},
            )
        )

else:
    st.warning(
        "Le rapport d'anomalies opérationnelles "
        "n'a pas encore été généré."
    )


# -------------------------------------------------------------------
# Codes HTTP
# -------------------------------------------------------------------

st.header(
    "2. Fiabilité de l'API"
)


status_counts = (
    production_df["status_code"]
    .value_counts()
    .sort_index()
    .rename_axis("status_code")
    .reset_index(name="count")
)

st.subheader(
    "Distribution des codes HTTP"
)

st.bar_chart(
    status_counts,
    x="status_code",
    y="count",
)


# -------------------------------------------------------------------
# Latence
# -------------------------------------------------------------------

st.subheader(
    "Évolution de la latence"
)


latency_chart_df = (
    production_df[
        [
            "created_at",
            "latency_ms",
        ]
    ]
    .dropna()
    .set_index(
        "created_at"
    )
)

st.line_chart(
    latency_chart_df
)


lat_col1, lat_col2, lat_col3, lat_col4 = st.columns(
    4
)

lat_col1.metric(
    "Latence moyenne",
    f"{mean_latency:.2f} ms",
)

lat_col2.metric(
    "Latence médiane",
    (
        f"{latency.median():.2f} ms"
        if not latency.empty
        else "N/A"
    ),
)

lat_col3.metric(
    "Latence p95",
    f"{p95_latency:.2f} ms",
)

lat_col4.metric(
    "Latence maximale",
    (
        f"{latency.max():.2f} ms"
        if not latency.empty
        else "N/A"
    ),
)


# -------------------------------------------------------------------
# Scores du modèle
# -------------------------------------------------------------------

st.header(
    "3. Distribution des prédictions"
)


scores = successful_df[
    "probability_default"
].dropna()


if not scores.empty:
    score_col1, score_col2, score_col3 = st.columns(
        3
    )

    score_col1.metric(
        "Score moyen",
        f"{scores.mean():.4f}",
    )

    score_col2.metric(
        "Score médian",
        f"{scores.median():.4f}",
    )

    score_col3.metric(
        "Seuil de décision",
        "0.45",
    )


    st.subheader(
        "Distribution des scores de défaut"
    )


    score_bins = pd.cut(
        scores,
        bins=10,
    )

    score_distribution = (
        score_bins
        .value_counts()
        .sort_index()
        .rename_axis(
            "intervalle"
        )
        .reset_index(
            name="nombre"
        )
    )

    score_distribution[
        "intervalle"
    ] = score_distribution[
        "intervalle"
    ].astype(str)


    st.bar_chart(
        score_distribution,
        x="intervalle",
        y="nombre",
    )


# -------------------------------------------------------------------
# Classes prédites
# -------------------------------------------------------------------

st.subheader(
    "Répartition des décisions"
)


prediction_counts = (
    successful_df[
        "prediction_label"
    ]
    .dropna()
    .value_counts()
    .rename_axis(
        "prediction"
    )
    .reset_index(
        name="nombre"
    )
)


if not prediction_counts.empty:
    st.bar_chart(
        prediction_counts,
        x="prediction",
        y="nombre",
    )


# -------------------------------------------------------------------
# Data Drift
# -------------------------------------------------------------------

st.header(
    "4. Data Drift"
)


if drift_summary:
    drift_col1, drift_col2, drift_col3, drift_col4 = st.columns(
        4
    )

    drift_col1.metric(
        "Features analysées",
        drift_summary.get(
            "features_total",
            0,
        ),
    )

    drift_col2.metric(
        "Features en drift",
        drift_summary.get(
            "features_drifted",
            0,
        ),
    )

    drift_col3.metric(
        "Features stables",
        drift_summary.get(
            "features_stable",
            0,
        ),
    )

    drift_col4.metric(
        "Part en drift",
        (
            f"{drift_summary.get('drift_percentage', 0):.2f} %"
        ),
    )


    if drift_summary.get(
        "dataset_drift_detected",
        False,
    ):
        st.error(
            "Dataset Drift détecté."
        )

    else:
        st.success(
            "Aucun Dataset Drift détecté."
        )


    st.caption(
        "Référence : "
        f"{drift_summary.get('reference_rows', 0)} observations | "
        "Production : "
        f"{drift_summary.get('production_rows', 0)} observations"
    )

else:
    st.warning(
        "La synthèse Evidently n'a pas encore été générée."
    )


# -------------------------------------------------------------------
# Top features en drift
# -------------------------------------------------------------------

if not drift_features_df.empty:
    st.subheader(
        "Features présentant le plus fort drift"
    )

    top_drift_features = (
        drift_features_df[
            drift_features_df[
                "drift_detected"
            ]
        ]
        .sort_values(
            by="drift_score",
            ascending=False,
        )
        .head(15)
    )


    st.dataframe(
        top_drift_features[
            [
                "feature",
                "drift_score",
                "threshold",
                "method",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


    st.bar_chart(
        top_drift_features,
        x="feature",
        y="drift_score",
    )


# -------------------------------------------------------------------
# Performance supervisée
# -------------------------------------------------------------------

st.header(
    "5. Performance du modèle"
)


labelled_df = production_df[
    (
        production_df["status_code"] == 200
    )
    & production_df[
        "actual_default"
    ].notna()
    & production_df[
        "prediction"
    ].notna()
].copy()


if labelled_df.empty:
    st.info(
        "Aucune vérité terrain disponible pour le moment. "
        "Les métriques Accuracy, Precision, Recall, F1, "
        "TP, TN, FP et FN seront calculées lorsque "
        "`actual_default` sera renseigné."
    )

else:
    y_true = labelled_df[
        "actual_default"
    ].astype(int)

    y_pred = labelled_df[
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
        labels=[
            0,
            1,
        ],
    ).ravel()


    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(
        4
    )

    perf_col1.metric(
        "Accuracy",
        f"{accuracy:.3f}",
    )

    perf_col2.metric(
        "Precision",
        f"{precision:.3f}",
    )

    perf_col3.metric(
        "Recall",
        f"{recall:.3f}",
    )

    perf_col4.metric(
        "F1-score",
        f"{f1:.3f}",
    )


    confusion_df = pd.DataFrame(
        {
            "Métrique": [
                "True Negative",
                "False Positive",
                "False Negative",
                "True Positive",
            ],
            "Valeur": [
                int(tn),
                int(fp),
                int(fn),
                int(tp),
            ],
        }
    )


    st.subheader(
        "Matrice de confusion"
    )

    st.dataframe(
        confusion_df,
        use_container_width=True,
        hide_index=True,
    )


# -------------------------------------------------------------------
# Points de vigilance
# -------------------------------------------------------------------

st.header(
    "6. Points de vigilance"
)


st.warning(
    "Le trafic actuel correspond principalement à un PoC "
    "et contient des requêtes volontairement invalides utilisées "
    "pour tester la détection d'anomalies."
)


st.warning(
    "Les données utilisées pour simuler certaines prédictions "
    "ne reproduisent pas les distributions réelles du P6. "
    "Le niveau de Data Drift observé ne doit donc pas être "
    "interprété comme une dérive réelle du modèle."
)


st.info(
    "En production réelle, les seuils de latence et de taux "
    "d'erreur devront être recalibrés à partir d'une période "
    "de fonctionnement stable."
)


st.info(
    "Les données stockées doivent rester pseudonymisées, "
    "avec une politique de rétention adaptée aux contraintes "
    "RGPD et au coût du stockage."
)