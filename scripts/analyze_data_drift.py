import json
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from evidently import Report
from evidently.presets import DataDriftPreset

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


REFERENCE_FILE = Path(
    "data/reference/reference_features_10000.csv"
)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DRIFT_REPORT_PATH = (
    REPORTS_DIR / "data_drift_report.html"
)

DRIFT_SUMMARY_PATH = (
    REPORTS_DIR / "data_drift_summary.json"
)

DRIFT_FEATURES_PATH = (
    REPORTS_DIR / "data_drift_features.csv"
)


# -------------------------------------------------------------------
# Requête SQL
# -------------------------------------------------------------------

QUERY = """
SELECT
    input_features,
    status_code
FROM prediction_logs
WHERE status_code = 200
ORDER BY created_at;
"""


# -------------------------------------------------------------------
# Chargement des données
# -------------------------------------------------------------------

def load_reference_data() -> pd.DataFrame:
    """
    Charge les données de référence du modèle champion P6.

    Ces données constituent la baseline utilisée pour comparer
    les distributions observées en production.
    """
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Fichier de référence introuvable : "
            f"{REFERENCE_FILE}"
        )

    dataframe = pd.read_csv(
        REFERENCE_FILE
    )

    if dataframe.empty:
        raise RuntimeError(
            "Le dataset de référence est vide."
        )

    return dataframe


def load_production_data() -> pd.DataFrame:
    """
    Reconstruit les inputs de production à partir des appels
    réussis stockés dans PostgreSQL/Supabase.
    """
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        pool_pre_ping=True,
    )

    try:
        with engine.connect() as connection:
            logs_df = pd.read_sql_query(
                QUERY,
                connection,
            )
    finally:
        engine.dispose()

    if logs_df.empty:
        raise RuntimeError(
            "Aucune prédiction valide disponible "
            "en production."
        )

    production_records = []

    for value in logs_df["input_features"]:
        if isinstance(value, dict):
            record = value

        elif isinstance(value, str):
            record = json.loads(
                value
            )

        else:
            raise TypeError(
                "Format inattendu dans input_features : "
                f"{type(value)}"
            )

        production_records.append(
            record
        )

    production_df = pd.DataFrame(
        production_records
    )

    if production_df.empty:
        raise RuntimeError(
            "Impossible de reconstruire les données "
            "de production."
        )

    return production_df


# -------------------------------------------------------------------
# Contrôle de compatibilité
# -------------------------------------------------------------------

def check_feature_compatibility(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Vérifie que les datasets de référence et de production
    contiennent exactement les mêmes features.
    """
    missing_features = [
        feature
        for feature in reference_df.columns
        if feature not in production_df.columns
    ]

    extra_features = [
        feature
        for feature in production_df.columns
        if feature not in reference_df.columns
    ]

    return (
        missing_features,
        extra_features,
    )


def align_production_columns(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Réordonne les colonnes de production dans le même ordre
    que celles utilisées par les données de référence.
    """
    return production_df[
        reference_df.columns
    ].copy()


# -------------------------------------------------------------------
# Evidently
# -------------------------------------------------------------------

def generate_drift_report(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
):
    """
    Exécute l'analyse de Data Drift avec Evidently
    et sauvegarde le rapport HTML complet.
    """
    report = Report(
        [
            DataDriftPreset(),
        ]
    )

    snapshot = report.run(
        reference_data=reference_df,
        current_data=production_df,
    )

    snapshot.save_html(
        str(DRIFT_REPORT_PATH)
    )

    return snapshot


def snapshot_to_dict(
    snapshot,
) -> dict:
    """
    Convertit un Snapshot Evidently en dictionnaire Python.

    La fonction reste compatible avec Evidently 0.7.21.
    """
    if hasattr(
        snapshot,
        "dict",
    ):
        try:
            return snapshot.dict()
        except Exception:
            pass

    if hasattr(
        snapshot,
        "model_dump",
    ):
        try:
            return snapshot.model_dump()
        except Exception:
            pass

    raise RuntimeError(
        "Impossible de convertir le snapshot Evidently "
        "en dictionnaire."
    )


# -------------------------------------------------------------------
# Synthèse globale du drift
# -------------------------------------------------------------------

def extract_global_drift_summary(
    snapshot,
    reference_rows: int,
    production_rows: int,
    features_total: int,
) -> dict:
    """
    Extrait la métrique globale DriftedColumnsCount.

    Elle fournit :
    - le nombre de features détectées en drift ;
    - la proportion de features détectées en drift ;
    - le statut global du dataset.
    """
    snapshot_dict = snapshot_to_dict(
        snapshot
    )

    metrics = snapshot_dict.get(
        "metrics",
        [],
    )

    for metric in metrics:
        if not isinstance(
            metric,
            dict,
        ):
            continue

        metric_name = str(
            metric.get(
                "metric_name",
                "",
            )
        )

        if not metric_name.startswith(
            "DriftedColumnsCount"
        ):
            continue

        value = metric.get(
            "value",
            {},
        )

        if not isinstance(
            value,
            dict,
        ):
            continue

        drifted_count = int(
            value.get(
                "count",
                0,
            )
        )

        drift_share = float(
            value.get(
                "share",
                0.0,
            )
        )

        config = metric.get(
            "config",
            {},
        )

        drift_threshold = 0.5

        if isinstance(
            config,
            dict,
        ):
            drift_threshold = float(
                config.get(
                    "drift_share",
                    drift_threshold,
                )
            )

        return {
            "reference_rows": reference_rows,
            "production_rows": production_rows,
            "features_total": features_total,
            "features_drifted": drifted_count,
            "features_stable": (
                features_total
                - drifted_count
            ),
            "drift_share": round(
                drift_share,
                4,
            ),
            "drift_percentage": round(
                drift_share * 100,
                2,
            ),
            "dataset_drift_detected": (
                drift_share
                >= drift_threshold
            ),
            "dataset_drift_threshold": (
                drift_threshold
            ),
            "production_data_warning": (
                production_rows < 100
            ),
            "production_data_type": (
                "simulated"
            ),
        }

    raise RuntimeError(
        "Métrique DriftedColumnsCount "
        "introuvable dans Evidently."
    )


def save_drift_summary(
    summary: dict,
) -> None:
    """
    Sauvegarde la synthèse globale du drift en JSON.
    """
    with open(
        DRIFT_SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )


# -------------------------------------------------------------------
# Drift feature par feature
# -------------------------------------------------------------------

def extract_feature_drift_details(
    snapshot,
) -> pd.DataFrame:
    """
    Extrait les métriques ValueDrift calculées
    par Evidently pour chaque feature.

    Pour chaque variable :
    - score de drift ;
    - seuil de décision ;
    - statut drift/non-drift ;
    - méthode statistique.
    """
    snapshot_dict = snapshot_to_dict(
        snapshot
    )

    metrics = snapshot_dict.get(
        "metrics",
        [],
    )

    records = []

    for metric in metrics:
        if not isinstance(
            metric,
            dict,
        ):
            continue

        config = metric.get(
            "config",
            {},
        )

        if not isinstance(
            config,
            dict,
        ):
            continue

        metric_type = str(
            config.get(
                "type",
                "",
            )
        )

        if "ValueDrift" not in metric_type:
            continue

        feature_name = config.get(
            "column"
        )

        method = config.get(
            "method"
        )

        threshold = config.get(
            "threshold"
        )

        drift_score = metric.get(
            "value"
        )

        if (
            feature_name is None
            or threshold is None
            or not isinstance(
                drift_score,
                (int, float),
            )
        ):
            continue

        threshold = float(
            threshold
        )

        drift_score = float(
            drift_score
        )

        records.append(
            {
                "feature": feature_name,
                "drift_score": drift_score,
                "threshold": threshold,
                "drift_detected": (
                    drift_score
                    >= threshold
                ),
                "method": method,
            }
        )

    dataframe = pd.DataFrame(
        records
    )

    if dataframe.empty:
        raise RuntimeError(
            "Aucune métrique ValueDrift "
            "n'a été trouvée dans Evidently."
        )

    dataframe = (
        dataframe
        .sort_values(
            by="drift_score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return dataframe


def save_feature_drift_details(
    dataframe: pd.DataFrame,
) -> None:
    """
    Sauvegarde le détail du drift feature par feature
    au format CSV.
    """
    dataframe.to_csv(
        DRIFT_FEATURES_PATH,
        index=False,
    )


# -------------------------------------------------------------------
# Affichage synthèse
# -------------------------------------------------------------------

def display_drift_summary(
    summary: dict,
) -> None:
    """
    Affiche la synthèse globale du Data Drift.
    """
    print(
        "\n=== SYNTHESE DATA DRIFT ===\n"
    )

    print(
        "Features analysées :",
        summary["features_total"],
    )

    print(
        "Features en drift :",
        summary["features_drifted"],
    )

    print(
        "Features stables :",
        summary["features_stable"],
    )

    print(
        "Pourcentage de drift :",
        f"{summary['drift_percentage']:.2f} %",
    )

    print(
        "Seuil dataset drift :",
        (
            f"{summary['dataset_drift_threshold'] * 100:.0f} %"
        ),
    )

    print(
        "Dataset drift détecté :",
        (
            "OUI"
            if summary[
                "dataset_drift_detected"
            ]
            else "NON"
        ),
    )

    print(
        "Observations référence :",
        summary["reference_rows"],
    )

    print(
        "Observations production :",
        summary["production_rows"],
    )

    print(
        "Synthèse JSON :",
        DRIFT_SUMMARY_PATH,
    )


def display_feature_drift(
    dataframe: pd.DataFrame,
) -> None:
    """
    Affiche une synthèse du drift feature par feature.
    """
    drifted_count = int(
        dataframe[
            "drift_detected"
        ].sum()
    )

    stable_count = int(
        (
            ~dataframe[
                "drift_detected"
            ]
        ).sum()
    )

    print(
        "\n=== DRIFT PAR FEATURE ===\n"
    )

    print(
        "Features détaillées :",
        len(dataframe),
    )

    print(
        "Features en drift :",
        drifted_count,
    )

    print(
        "Features stables :",
        stable_count,
    )

    print(
        "\nTop 10 des scores de drift :\n"
    )

    print(
        dataframe[
            [
                "feature",
                "drift_score",
                "threshold",
                "drift_detected",
                "method",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        "\nDétail CSV :",
        DRIFT_FEATURES_PATH,
    )


# -------------------------------------------------------------------
# Programme principal
# -------------------------------------------------------------------

def main() -> None:
    reference_df = load_reference_data()
    production_df = load_production_data()

    print(
        "\n=== DATA DRIFT - PREPARATION ===\n"
    )

    print(
        "Référence :",
        reference_df.shape,
    )

    print(
        "Production :",
        production_df.shape,
    )

    print(
        "Features référence :",
        len(reference_df.columns),
    )

    print(
        "Features production :",
        len(production_df.columns),
    )

    (
        missing_features,
        extra_features,
    ) = check_feature_compatibility(
        reference_df,
        production_df,
    )

    print(
        "Features manquantes en production :",
        len(missing_features),
    )

    print(
        "Features supplémentaires en production :",
        len(extra_features),
    )

    if (
        missing_features
        or extra_features
    ):
        if missing_features:
            print(
                "Exemples manquants :",
                missing_features[:10],
            )

        if extra_features:
            print(
                "Exemples supplémentaires :",
                extra_features[:10],
            )

        raise RuntimeError(
            "Les datasets ne sont pas compatibles "
            "pour l'analyse du drift."
        )

    production_df = align_production_columns(
        reference_df,
        production_df,
    )

    same_order = (
        list(
            production_df.columns
        )
        == list(
            reference_df.columns
        )
    )

    print(
        "Ordre aligné sur la référence :",
        same_order,
    )

    if not same_order:
        raise RuntimeError(
            "L'ordre des features n'a pas pu être aligné."
        )

    print(
        "\nLes données sont compatibles "
        "pour l'analyse du drift."
    )

    # ---------------------------------------------------------------
    # Evidently
    # ---------------------------------------------------------------

    print(
        "\n=== EVIDENTLY - DATA DRIFT ===\n"
    )

    print(
        "Génération du rapport Evidently..."
    )

    snapshot = generate_drift_report(
        reference_df=reference_df,
        production_df=production_df,
    )

    print(
        "Rapport HTML :",
        DRIFT_REPORT_PATH,
    )

    # ---------------------------------------------------------------
    # Synthèse globale
    # ---------------------------------------------------------------

    drift_summary = extract_global_drift_summary(
        snapshot=snapshot,
        reference_rows=len(
            reference_df
        ),
        production_rows=len(
            production_df
        ),
        features_total=len(
            reference_df.columns
        ),
    )

    save_drift_summary(
        drift_summary
    )

    display_drift_summary(
        drift_summary
    )

    # ---------------------------------------------------------------
    # Détail par feature
    # ---------------------------------------------------------------

    feature_drift_df = extract_feature_drift_details(
        snapshot
    )

    save_feature_drift_details(
        feature_drift_df
    )

    display_feature_drift(
        feature_drift_df
    )

    # ---------------------------------------------------------------
    # Contrôle de cohérence
    # ---------------------------------------------------------------

    feature_drift_count = int(
        feature_drift_df[
            "drift_detected"
        ].sum()
    )

    global_drift_count = (
        drift_summary[
            "features_drifted"
        ]
    )

    if (
        feature_drift_count
        != global_drift_count
    ):
        print(
            "\nATTENTION : le nombre de features "
            "calculé individuellement diffère "
            "de DriftedColumnsCount."
        )

        print(
            "Global Evidently :",
            global_drift_count,
        )

        print(
            "Calcul individuel :",
            feature_drift_count,
        )

    else:
        print(
            "\nContrôle de cohérence : OK"
        )

        print(
            "Le nombre de features en drift "
            "correspond à DriftedColumnsCount."
        )

    # ---------------------------------------------------------------
    # Interprétation
    # ---------------------------------------------------------------

    print(
        "\n=== INTERPRETATION ===\n"
    )

    if drift_summary[
        "production_data_warning"
    ]:
        print(
            "ATTENTION : le volume de production "
            "est encore faible."
        )

    print(
        "Les données actuelles proviennent "
        "d'un trafic de test simulé."
    )

    print(
        "Les valeurs artificielles entre 0 et 1 "
        "ne reproduisent pas les distributions "
        "des données P6."
    )

    print(
        "Un drift important est donc attendu."
    )

    print(
        "Cette analyse valide le fonctionnement "
        "technique du pipeline Evidently, "
        "mais ne démontre pas une dérive réelle "
        "du modèle en production."
    )


if __name__ == "__main__":
    main()