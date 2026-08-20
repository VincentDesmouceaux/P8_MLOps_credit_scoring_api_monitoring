import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from app.services.model_service import model_service


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABELLED_FILE = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
    / "p6_labelled_monitoring_1000.csv"
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
    / "optimized_model_validation.json"
)

TARGET_COLUMN = "TARGET"

DECISION_THRESHOLD = 0.45

FN_COST = 10
FP_COST = 1


# -------------------------------------------------------------------
# Chargement
# -------------------------------------------------------------------

def load_labelled_data() -> pd.DataFrame:
    """
    Charge les 1000 observations P6 labellisées.
    """
    if not LABELLED_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : "
            f"{LABELLED_FILE}"
        )

    dataframe = pd.read_csv(
        LABELLED_FILE
    )

    if dataframe.empty:
        raise RuntimeError(
            "Le dataset labellisé est vide."
        )

    if TARGET_COLUMN not in dataframe.columns:
        raise RuntimeError(
            "La colonne TARGET est absente."
        )

    feature_columns = [
        column
        for column in dataframe.columns
        if column != TARGET_COLUMN
    ]

    if feature_columns != model_service.feature_names:
        raise RuntimeError(
            "Les features du dataset ne correspondent "
            "pas exactement aux features du modèle."
        )

    return dataframe


# -------------------------------------------------------------------
# Pipeline V1
# -------------------------------------------------------------------

def row_to_features(
    row: pd.Series,
) -> dict[str, float | None]:
    """
    Convertit une ligne pandas vers le format attendu
    par le pipeline actuel ModelService.
    """
    return {
        column: (
            None
            if pd.isna(value)
            else float(value)
        )
        for column, value in row.items()
        if column != TARGET_COLUMN
    }


def predict_v1(
    row: pd.Series,
) -> float:
    """
    Pipeline actuel :
    dict -> DataFrame pandas -> predict_proba.
    """
    features = row_to_features(
        row
    )

    return model_service.predict_proba(
        features
    )


# -------------------------------------------------------------------
# Pipeline V2
# -------------------------------------------------------------------

def row_to_numpy(
    row: pd.Series,
) -> np.ndarray:
    """
    Convertit directement les 656 features
    vers un tableau NumPy float32.
    """
    feature_row = row[
        model_service.feature_names
    ]

    values = feature_row.to_numpy(
        dtype=np.float32,
        copy=True,
    )

    return values.reshape(
        1,
        -1,
    )


def predict_v2(
    row: pd.Series,
) -> float:
    """
    Pipeline optimisé :
    NumPy -> XGBoost Booster.inplace_predict.
    """
    input_array = row_to_numpy(
        row
    )

    booster = (
        model_service.model.get_booster()
    )

    prediction = booster.inplace_predict(
        input_array
    )

    return float(
        prediction[0]
    )


# -------------------------------------------------------------------
# Métriques
# -------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """
    Calcule les métriques supervisées
    avec le seuil métier de 0.45.
    """
    y_pred = (
        probabilities
        >= DECISION_THRESHOLD
    ).astype(int)

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

    business_cost = (
        FN_COST * int(fn)
        + FP_COST * int(fp)
    )

    return {
        "accuracy": round(
            float(accuracy),
            6,
        ),
        "precision": round(
            float(precision),
            6,
        ),
        "recall": round(
            float(recall),
            6,
        ),
        "f1_score": round(
            float(f1),
            6,
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "business_cost": int(
            business_cost
        ),
    }


# -------------------------------------------------------------------
# Comparaison
# -------------------------------------------------------------------

def compare_metrics(
    metrics_v1: dict,
    metrics_v2: dict,
) -> bool:
    """
    Vérifie que toutes les métriques sont identiques.
    """
    return (
        metrics_v1
        == metrics_v2
    )


# -------------------------------------------------------------------
# Programme principal
# -------------------------------------------------------------------

def main() -> None:
    print(
        "\n=== VALIDATION MODELE OPTIMISE ===\n"
    )

    dataframe = load_labelled_data()

    print(
        "Observations :",
        len(dataframe),
    )

    print(
        "Features :",
        len(
            model_service.feature_names
        ),
    )

    print(
        "TARGET = 0 :",
        int(
            (
                dataframe[TARGET_COLUMN] == 0
            ).sum()
        ),
    )

    print(
        "TARGET = 1 :",
        int(
            (
                dataframe[TARGET_COLUMN] == 1
            ).sum()
        ),
    )

    # ---------------------------------------------------------------
    # Chargement modèle
    # ---------------------------------------------------------------

    model_service.load()

    # ---------------------------------------------------------------
    # Prédictions
    # ---------------------------------------------------------------

    probabilities_v1 = []
    probabilities_v2 = []

    print(
        "\nCalcul des prédictions V1..."
    )

    for _, row in dataframe.iterrows():
        probabilities_v1.append(
            predict_v1(
                row
            )
        )

    print(
        "Calcul des prédictions V2..."
    )

    for _, row in dataframe.iterrows():
        probabilities_v2.append(
            predict_v2(
                row
            )
        )

    probabilities_v1 = np.asarray(
        probabilities_v1,
        dtype=float,
    )

    probabilities_v2 = np.asarray(
        probabilities_v2,
        dtype=float,
    )

    y_true = dataframe[
        TARGET_COLUMN
    ].astype(int).to_numpy()

    # ---------------------------------------------------------------
    # Comparaison probabilités
    # ---------------------------------------------------------------

    absolute_differences = np.abs(
        probabilities_v1
        - probabilities_v2
    )

    max_difference = float(
        absolute_differences.max()
    )

    mean_difference = float(
        absolute_differences.mean()
    )

    probabilities_identical = bool(
        np.allclose(
            probabilities_v1,
            probabilities_v2,
            rtol=1e-6,
            atol=1e-7,
        )
    )

    # ---------------------------------------------------------------
    # Comparaison classes
    # ---------------------------------------------------------------

    predictions_v1 = (
        probabilities_v1
        >= DECISION_THRESHOLD
    ).astype(int)

    predictions_v2 = (
        probabilities_v2
        >= DECISION_THRESHOLD
    ).astype(int)

    predictions_identical = bool(
        np.array_equal(
            predictions_v1,
            predictions_v2,
        )
    )

    # ---------------------------------------------------------------
    # Métriques
    # ---------------------------------------------------------------

    metrics_v1 = compute_metrics(
        y_true=y_true,
        probabilities=probabilities_v1,
    )

    metrics_v2 = compute_metrics(
        y_true=y_true,
        probabilities=probabilities_v2,
    )

    metrics_identical = compare_metrics(
        metrics_v1,
        metrics_v2,
    )

    # ---------------------------------------------------------------
    # Rapport
    # ---------------------------------------------------------------

    report = {
        "observations": int(
            len(dataframe)
        ),
        "decision_threshold": (
            DECISION_THRESHOLD
        ),
        "probabilities_identical": (
            probabilities_identical
        ),
        "predictions_identical": (
            predictions_identical
        ),
        "metrics_identical": (
            metrics_identical
        ),
        "max_probability_difference": (
            max_difference
        ),
        "mean_probability_difference": (
            mean_difference
        ),
        "v1_current_pipeline": (
            metrics_v1
        ),
        "v2_numpy_pipeline": (
            metrics_v2
        ),
        "regression_detected": not (
            probabilities_identical
            and predictions_identical
            and metrics_identical
        ),
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

    # ---------------------------------------------------------------
    # Affichage
    # ---------------------------------------------------------------

    print(
        "\n=== V1 - PIPELINE ACTUEL ===\n"
    )

    print(
        json.dumps(
            metrics_v1,
            indent=4,
        )
    )

    print(
        "\n=== V2 - PIPELINE NUMPY ===\n"
    )

    print(
        json.dumps(
            metrics_v2,
            indent=4,
        )
    )

    print(
        "\n=== COMPARAISON ===\n"
    )

    print(
        "Probabilités identiques :",
        probabilities_identical,
    )

    print(
        "Prédictions identiques :",
        predictions_identical,
    )

    print(
        "Métriques identiques :",
        metrics_identical,
    )

    print(
        "Différence maximale :",
        max_difference,
    )

    print(
        "Différence moyenne :",
        mean_difference,
    )

    print(
        "Régression détectée :",
        report[
            "regression_detected"
        ],
    )

    print(
        "\nRapport :",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()