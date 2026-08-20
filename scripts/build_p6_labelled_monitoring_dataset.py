import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

P6_ROOT = (
    PROJECT_ROOT.parent
    / "P6_MLOps_credit_scoring"
)

SOURCE_FILE = (
    P6_ROOT
    / "data"
    / "processed"
    / "train_modeling.csv"
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

OUTPUT_FILE = (
    OUTPUT_DIR
    / "p6_labelled_monitoring_1000.csv"
)

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"

SAMPLE_SIZE = 1000
RANDOM_STATE = 42


def clean_feature_names(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
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


def main() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {SOURCE_FILE}"
        )

    with open(
        FEATURE_NAMES_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        model_features = json.load(file)

    dataframe = pd.read_csv(
        SOURCE_FILE
    )

    y = dataframe[
        TARGET_COLUMN
    ].astype(int)

    X = dataframe.drop(
        columns=[
            TARGET_COLUMN,
            ID_COLUMN,
        ],
        errors="ignore",
    )

    X = clean_feature_names(
        X
    )

    missing_features = [
        feature
        for feature in model_features
        if feature not in X.columns
    ]

    extra_features = [
        feature
        for feature in X.columns
        if feature not in model_features
    ]

    if missing_features or extra_features:
        raise RuntimeError(
            "Les features du dataset labellisé "
            "ne correspondent pas au modèle."
        )

    X = X[
        model_features
    ]

    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_sample = X_sample.reset_index(
        drop=True
    )

    y_sample = y_sample.reset_index(
        drop=True
    )

    monitoring_df = X_sample.copy()

    monitoring_df[
        TARGET_COLUMN
    ] = y_sample

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    monitoring_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\n=== DATASET MONITORING LABELLISE ===\n"
    )

    print(
        "Shape :",
        monitoring_df.shape,
    )

    print(
        "TARGET = 0 :",
        int(
            (
                monitoring_df[TARGET_COLUMN] == 0
            ).sum()
        ),
    )

    print(
        "TARGET = 1 :",
        int(
            (
                monitoring_df[TARGET_COLUMN] == 1
            ).sum()
        ),
    )

    print(
        "Taux de défaut :",
        round(
            monitoring_df[
                TARGET_COLUMN
            ].mean(),
            4,
        ),
    )

    print(
        "Fichier :",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()