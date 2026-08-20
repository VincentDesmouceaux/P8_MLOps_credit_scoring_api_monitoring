import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

P6_ROOT = (
    PROJECT_ROOT.parent
    / "P6_MLOps_credit_scoring"
)

P6_TRAIN_FILE = (
    P6_ROOT
    / "data"
    / "processed"
    / "train_modeling.csv"
)

P6_TEST_FILE = (
    P6_ROOT
    / "data"
    / "processed"
    / "test_modeling.csv"
)

FEATURE_NAMES_FILE = (
    PROJECT_ROOT
    / "models"
    / "feature_names.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
)

REFERENCE_OUTPUT = (
    OUTPUT_DIR
    / "p6_reference_full.csv"
)

PRODUCTION_OUTPUT = (
    OUTPUT_DIR
    / "p6_production_full.csv"
)

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"


# -------------------------------------------------------------------
# Nettoyage des noms de features
# -------------------------------------------------------------------

def clean_feature_names(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduit exactement le nettoyage utilisé pendant
    l'entraînement XGBoost du P6.
    """
    dataframe = dataframe.copy()

    cleaned_columns = []
    counts = Counter()

    for column in dataframe.columns:
        clean_column = re.sub(
            r"[^A-Za-z0-9_]+",
            "_",
            str(column),
        )

        clean_column = re.sub(
            r"_+",
            "_",
            clean_column,
        ).strip("_")

        if not clean_column:
            clean_column = "feature"

        counts[clean_column] += 1

        if counts[clean_column] > 1:
            clean_column = (
                f"{clean_column}_"
                f"{counts[clean_column]}"
            )

        cleaned_columns.append(
            clean_column
        )

    dataframe.columns = cleaned_columns

    return dataframe


# -------------------------------------------------------------------
# Features attendues par le modèle
# -------------------------------------------------------------------

def load_model_features() -> list[str]:
    if not FEATURE_NAMES_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : "
            f"{FEATURE_NAMES_FILE}"
        )

    with open(
        FEATURE_NAMES_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# -------------------------------------------------------------------
# Référence P6
# -------------------------------------------------------------------

def build_reference_dataset(
    model_features: list[str],
) -> pd.DataFrame:
    """
    Prépare les 307 507 observations labellisées du P6
    dans l'espace exact des 656 features du modèle.
    """
    if not P6_TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Dataset P6 introuvable : "
            f"{P6_TRAIN_FILE}"
        )

    dataframe = pd.read_csv(
        P6_TRAIN_FILE
    )

    dataframe = dataframe.drop(
        columns=[
            TARGET_COLUMN,
            ID_COLUMN,
        ],
        errors="ignore",
    )

    dataframe = clean_feature_names(
        dataframe
    )

    validate_features(
        dataframe=dataframe,
        model_features=model_features,
        dataset_name="référence P6",
    )

    return dataframe[
        model_features
    ].copy()


# -------------------------------------------------------------------
# Production simulée réaliste
# -------------------------------------------------------------------

def build_production_dataset(
    model_features: list[str],
) -> pd.DataFrame:
    """
    Prépare les 48 744 observations du jeu test P6
    dans l'espace exact des 656 features du modèle.
    """
    if not P6_TEST_FILE.exists():
        raise FileNotFoundError(
            f"Dataset P6 introuvable : "
            f"{P6_TEST_FILE}"
        )

    dataframe = pd.read_csv(
        P6_TEST_FILE
    )

    dataframe = dataframe.drop(
        columns=[
            ID_COLUMN,
        ],
        errors="ignore",
    )

    dataframe = clean_feature_names(
        dataframe
    )

    validate_features(
        dataframe=dataframe,
        model_features=model_features,
        dataset_name="production simulée P6",
    )

    return dataframe[
        model_features
    ].copy()


# -------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------

def validate_features(
    *,
    dataframe: pd.DataFrame,
    model_features: list[str],
    dataset_name: str,
) -> None:
    missing_features = [
        feature
        for feature in model_features
        if feature not in dataframe.columns
    ]

    extra_features = [
        feature
        for feature in dataframe.columns
        if feature not in model_features
    ]

    if missing_features or extra_features:
        raise RuntimeError(
            f"Incompatibilité des features pour "
            f"{dataset_name}. "
            f"Manquantes={len(missing_features)}, "
            f"supplémentaires={len(extra_features)}."
        )

    if list(dataframe.columns) != model_features:
        dataframe = dataframe[
            model_features
        ]

    print(
        f"{dataset_name} : "
        f"{len(dataframe)} lignes, "
        f"{len(model_features)} features."
    )


# -------------------------------------------------------------------
# Programme principal
# -------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_features = load_model_features()

    print(
        "\n=== PREPARATION DES DONNEES P6 POUR P8 ===\n"
    )

    print(
        "Features attendues par le modèle :",
        len(model_features),
    )

    reference_df = build_reference_dataset(
        model_features
    )

    production_df = build_production_dataset(
        model_features
    )

    print(
        "\n=== VERIFICATION ===\n"
    )

    print(
        "Référence :",
        reference_df.shape,
    )

    print(
        "Production simulée :",
        production_df.shape,
    )

    print(
        "Même espace de features :",
        list(reference_df.columns)
        == list(production_df.columns)
        == model_features,
    )

    print(
        "\nExport de la production simulée..."
    )

    production_df.to_csv(
        PRODUCTION_OUTPUT,
        index=False,
    )

    print(
        "Production :",
        PRODUCTION_OUTPUT,
    )

    print(
        "\nNOTE : la référence complète contient "
        "307 507 × 656 valeurs."
    )

    print(
        "Nous ne l'exportons pas encore en CSV afin "
        "d'éviter de dupliquer inutilement un très gros fichier."
    )


if __name__ == "__main__":
    main()