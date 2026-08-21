import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
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

ONNX_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "onnx"
    / "credit_scoring_model.onnx"
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
    / "onnx_model_validation.json"
)

TARGET_COLUMN = "TARGET"

DECISION_THRESHOLD = 0.45

FN_COST = 10
FP_COST = 1


# -------------------------------------------------------------------
# Chargement des données
# -------------------------------------------------------------------

def load_labelled_data() -> pd.DataFrame:
    if not LABELLED_FILE.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {LABELLED_FILE}"
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
            "Les features ne correspondent pas "
            "exactement au modèle."
        )

    return dataframe


# -------------------------------------------------------------------
# Session ONNX
# -------------------------------------------------------------------

def create_onnx_session() -> ort.InferenceSession:
    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle ONNX introuvable : "
            f"{ONNX_MODEL_PATH}"
        )

    return ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        providers=[
            "CPUExecutionProvider",
        ],
    )


# -------------------------------------------------------------------
# Conversion NumPy
# -------------------------------------------------------------------

def row_to_numpy(
    row: pd.Series,
) -> np.ndarray:
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


# -------------------------------------------------------------------
# XGBoost
# -------------------------------------------------------------------

def predict_xgboost(
    row: pd.Series,
) -> float:
    if model_service.booster is None:
        raise RuntimeError(
            "Booster XGBoost non chargé."
        )

    input_array = row_to_numpy(
        row
    )

    prediction = (
        model_service.booster.inplace_predict(
            input_array
        )
    )

    return float(
        prediction[0]
    )


# -------------------------------------------------------------------
# ONNX
# -------------------------------------------------------------------

def predict_onnx(
    session: ort.InferenceSession,
    input_name: str,
    row: pd.Series,
) -> float:
    input_array = row_to_numpy(
        row
    )

    outputs = session.run(
        None,
        {
            input_name: input_array,
        },
    )

    if len(outputs) < 2:
        raise RuntimeError(
            "Sorties ONNX inattendues."
        )

    probabilities = outputs[1]

    if not isinstance(
        probabilities,
        np.ndarray,
    ):
        raise RuntimeError(
            "La sortie probabilities n'est pas "
            "un tableau NumPy."
        )

    if probabilities.shape != (
        1,
        2,
    ):
        raise RuntimeError(
            "Shape ONNX inattendue : "
            f"{probabilities.shape}"
        )

    return float(
        probabilities[0][1]
    )


# -------------------------------------------------------------------
# Métriques
# -------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
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
# Programme principal
# -------------------------------------------------------------------

def main() -> None:
    print(
        "\n=== VALIDATION ONNX SUR DONNEES LABELLISEES ===\n"
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

    model_service.load()

    session = create_onnx_session()

    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    probabilities_xgb = []
    probabilities_onnx = []

    print(
        "\nPrédictions XGBoost..."
    )

    for _, row in dataframe.iterrows():
        probabilities_xgb.append(
            predict_xgboost(
                row
            )
        )

    print(
        "Prédictions ONNX..."
    )

    for _, row in dataframe.iterrows():
        probabilities_onnx.append(
            predict_onnx(
                session,
                input_name,
                row,
            )
        )

    probabilities_xgb = np.asarray(
        probabilities_xgb,
        dtype=float,
    )

    probabilities_onnx = np.asarray(
        probabilities_onnx,
        dtype=float,
    )

    y_true = dataframe[
        TARGET_COLUMN
    ].astype(int).to_numpy()

    # ---------------------------------------------------------------
    # Comparaison numérique
    # ---------------------------------------------------------------

    absolute_differences = np.abs(
        probabilities_xgb
        - probabilities_onnx
    )

    max_difference = float(
        absolute_differences.max()
    )

    mean_difference = float(
        absolute_differences.mean()
    )

    probabilities_equivalent = bool(
        np.allclose(
            probabilities_xgb,
            probabilities_onnx,
            rtol=1e-5,
            atol=1e-6,
        )
    )

    # ---------------------------------------------------------------
    # Classes
    # ---------------------------------------------------------------

    predictions_xgb = (
        probabilities_xgb
        >= DECISION_THRESHOLD
    ).astype(int)

    predictions_onnx = (
        probabilities_onnx
        >= DECISION_THRESHOLD
    ).astype(int)

    predictions_identical = bool(
        np.array_equal(
            predictions_xgb,
            predictions_onnx,
        )
    )

    # ---------------------------------------------------------------
    # Métriques
    # ---------------------------------------------------------------

    metrics_xgb = compute_metrics(
        y_true,
        probabilities_xgb,
    )

    metrics_onnx = compute_metrics(
        y_true,
        probabilities_onnx,
    )

    metrics_identical = (
        metrics_xgb
        == metrics_onnx
    )

    regression_detected = not (
        probabilities_equivalent
        and predictions_identical
        and metrics_identical
    )

    # ---------------------------------------------------------------
    # Rapport
    # ---------------------------------------------------------------

    report = {
        "observations": len(
            dataframe
        ),
        "decision_threshold": (
            DECISION_THRESHOLD
        ),
        "probabilities_equivalent": (
            probabilities_equivalent
        ),
        "predictions_identical": (
            predictions_identical
        ),
        "metrics_identical": (
            metrics_identical
        ),
        "regression_detected": (
            regression_detected
        ),
        "max_probability_difference": (
            max_difference
        ),
        "mean_probability_difference": (
            mean_difference
        ),
        "xgboost": metrics_xgb,
        "onnx_runtime": metrics_onnx,
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
        "\n=== XGBOOST ===\n"
    )

    print(
        json.dumps(
            metrics_xgb,
            indent=4,
        )
    )

    print(
        "\n=== ONNX RUNTIME ===\n"
    )

    print(
        json.dumps(
            metrics_onnx,
            indent=4,
        )
    )

    print(
        "\n=== COMPARAISON ===\n"
    )

    print(
        "Probabilités équivalentes :",
        probabilities_equivalent,
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
        regression_detected,
    )

    print(
        "\nRapport :",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()