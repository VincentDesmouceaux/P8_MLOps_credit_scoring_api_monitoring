from pathlib import Path

import onnx
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

from app.services.model_service import model_service


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ONNX_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "onnx"
)

ONNX_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ONNX_MODEL_PATH = (
    ONNX_MODEL_DIR
    / "credit_scoring_model.onnx"
)

EXPECTED_FEATURE_COUNT = 656


# -------------------------------------------------------------------
# Export
# -------------------------------------------------------------------

def export_model() -> None:
    """
    Convertit le modèle XGBoost champion vers ONNX.

    onnxmltools attend des noms de features XGBoost
    suivant le format f0, f1, ..., fN.

    Le Booster utilisé uniquement pour l'export reçoit donc
    temporairement ces noms normalisés.

    Le modèle de production et feature_names.json ne sont
    pas modifiés.
    """
    print(
        "\n=== EXPORT XGBOOST -> ONNX ===\n"
    )

    model_service.load()

    if model_service.model is None:
        raise RuntimeError(
            "Le modèle XGBoost n'est pas chargé."
        )

    if model_service.booster is None:
        raise RuntimeError(
            "Le Booster XGBoost n'est pas disponible."
        )

    feature_count = len(
        model_service.feature_names
    )

    if feature_count != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "Le modèle ne contient pas exactement "
            f"{EXPECTED_FEATURE_COUNT} features. "
            f"Reçu : {feature_count}."
        )

    print(
        "Features :",
        feature_count,
    )

    # ----------------------------------------------------------------
    # Sauvegarde des noms originaux
    # ----------------------------------------------------------------

    original_feature_names = (
        model_service.booster.feature_names
    )

    normalized_feature_names = [
        f"f{index}"
        for index in range(
            EXPECTED_FEATURE_COUNT
        )
    ]

    print(
        "Normalisation temporaire des noms "
        "de features pour ONNX..."
    )

    # ----------------------------------------------------------------
    # Modification temporaire
    # ----------------------------------------------------------------

    model_service.booster.feature_names = (
        normalized_feature_names
    )

    try:
        initial_types = [
            (
                "input",
                FloatTensorType(
                    [
                        None,
                        EXPECTED_FEATURE_COUNT,
                    ]
                ),
            )
        ]

        print(
            "Conversion en cours..."
        )

        onnx_model = (
            onnxmltools.convert_xgboost(
                model_service.booster,
                initial_types=initial_types,
            )
        )

        # ------------------------------------------------------------
        # Validation ONNX
        # ------------------------------------------------------------

        onnx.checker.check_model(
            onnx_model
        )

        onnx.save_model(
            onnx_model,
            str(ONNX_MODEL_PATH),
        )

    finally:
        # ------------------------------------------------------------
        # Restauration impérative
        # ------------------------------------------------------------

        model_service.booster.feature_names = (
            original_feature_names
        )

    print(
        "Noms de features XGBoost restaurés."
    )

    print(
        "Modèle ONNX valide."
    )

    print(
        "Fichier :",
        ONNX_MODEL_PATH,
    )

    print(
        "Taille :",
        (
            f"{ONNX_MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB"
        ),
    )


if __name__ == "__main__":
    export_model()