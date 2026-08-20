import json
from pathlib import Path
from typing import Any

import mlflow.xgboost
import numpy as np


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "credit_scoring_model"
)

FEATURE_NAMES_PATH = (
    PROJECT_ROOT
    / "models"
    / "feature_names.json"
)

EXPECTED_FEATURE_COUNT = 656


# -------------------------------------------------------------------
# Service modèle
# -------------------------------------------------------------------

class ModelService:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.booster: Any | None = None
        self.loaded = False

        if not FEATURE_NAMES_PATH.exists():
            raise FileNotFoundError(
                f"Fichier de features introuvable : "
                f"{FEATURE_NAMES_PATH}"
            )

        with open(
            FEATURE_NAMES_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            self.feature_names = json.load(file)

        if not isinstance(
            self.feature_names,
            list,
        ):
            raise TypeError(
                "feature_names.json doit contenir "
                "une liste de noms de features."
            )

        if len(self.feature_names) != EXPECTED_FEATURE_COUNT:
            raise RuntimeError(
                "Le modèle P8 doit utiliser exactement "
                f"{EXPECTED_FEATURE_COUNT} features. "
                f"Reçu : {len(self.feature_names)}."
            )

        if len(
            set(self.feature_names)
        ) != len(
            self.feature_names
        ):
            raise RuntimeError(
                "Des noms de features dupliqués "
                "ont été détectés."
            )

    # ----------------------------------------------------------------
    # Chargement du modèle
    # ----------------------------------------------------------------

    def load(self) -> None:
        """
        Charge le modèle MLflow/XGBoost une seule fois.

        Le Booster natif XGBoost est extrait au chargement
        afin d'éviter les transformations pandas coûteuses
        lors de chaque prédiction.
        """
        if self.loaded:
            return

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Artefact modèle introuvable : "
                f"{MODEL_PATH}"
            )

        print(
            "Chargement du modèle..."
        )

        self.model = mlflow.xgboost.load_model(
            str(MODEL_PATH)
        )

        if self.model is None:
            raise RuntimeError(
                "Le chargement du modèle a échoué."
            )

        try:
            self.booster = self.model.get_booster()

        except AttributeError as error:
            raise RuntimeError(
                "Impossible d'extraire le Booster XGBoost "
                "du modèle chargé."
            ) from error

        if self.booster is None:
            raise RuntimeError(
                "Le Booster XGBoost n'est pas disponible."
            )

        self.loaded = True

        print(
            "Modèle chargé."
        )

        print(
            "Pipeline d'inférence : NumPy + "
            "XGBoost Booster.inplace_predict"
        )

        print(
            "Nombre de features attendues :",
            len(self.feature_names),
        )

    # ----------------------------------------------------------------
    # Validation des features
    # ----------------------------------------------------------------

    def validate_features(
        self,
        features: dict[str, float | None],
    ) -> dict[str, list[str]]:
        """
        Vérifie que les features reçues correspondent exactement
        aux features attendues par le modèle.
        """
        missing_features = [
            feature
            for feature in self.feature_names
            if feature not in features
        ]

        extra_features = [
            feature
            for feature in features
            if feature not in self.feature_names
        ]

        return {
            "missing_features": missing_features,
            "extra_features": extra_features,
        }

    # ----------------------------------------------------------------
    # Construction du tableau NumPy
    # ----------------------------------------------------------------

    def build_input_array(
        self,
        features: dict[str, float | None],
    ) -> np.ndarray:
        """
        Reconstruit une observation dans l'ordre exact
        des 656 features utilisées lors de l'entraînement.

        Les valeurs JSON null deviennent None dans FastAPI
        puis np.nan ici afin de préserver les valeurs manquantes
        pour XGBoost.

        Le tableau est construit directement en float32 afin
        d'éviter le coût de création et de transformation
        d'un DataFrame pandas.
        """
        values = np.empty(
            EXPECTED_FEATURE_COUNT,
            dtype=np.float32,
        )

        for index, feature_name in enumerate(
            self.feature_names
        ):
            value = features[
                feature_name
            ]

            if value is None:
                values[index] = np.nan
                continue

            try:
                numeric_value = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ) as error:
                raise ValueError(
                    "Valeur non numérique pour la feature "
                    f"'{feature_name}'."
                ) from error

            if np.isinf(
                numeric_value
            ):
                raise ValueError(
                    "Valeur infinie interdite pour la feature "
                    f"'{feature_name}'."
                )

            values[index] = numeric_value

        input_array = values.reshape(
            1,
            -1,
        )

        if input_array.shape != (
            1,
            EXPECTED_FEATURE_COUNT,
        ):
            raise RuntimeError(
                "Le tableau NumPy d'inférence n'a pas "
                "la forme attendue."
            )

        return input_array

    # ----------------------------------------------------------------
    # Prédiction optimisée
    # ----------------------------------------------------------------

    def predict_proba(
        self,
        features: dict[str, float | None],
    ) -> float:
        """
        Retourne la probabilité de défaut du client.

        Pipeline optimisé :
        dict -> NumPy float32 -> Booster.inplace_predict.

        Cette approche évite les transformations pandas
        identifiées comme principal goulot d'étranglement
        lors du profiling cProfile.
        """
        self.load()

        validation = self.validate_features(
            features
        )

        missing_features = validation[
            "missing_features"
        ]

        extra_features = validation[
            "extra_features"
        ]

        if missing_features:
            raise ValueError(
                f"{len(missing_features)} "
                "features manquantes. "
                f"Exemples : {missing_features[:5]}"
            )

        if extra_features:
            raise ValueError(
                f"{len(extra_features)} "
                "features inconnues. "
                f"Exemples : {extra_features[:5]}"
            )

        input_array = self.build_input_array(
            features
        )

        if self.booster is None:
            raise RuntimeError(
                "Le Booster XGBoost n'est pas chargé."
            )

        predictions = self.booster.inplace_predict(
            input_array
        )

        if len(predictions) != 1:
            raise RuntimeError(
                "Format inattendu retourné par "
                "Booster.inplace_predict()."
            )

        probability_default = float(
            predictions[0]
        )

        if not np.isfinite(
            probability_default
        ):
            raise RuntimeError(
                "La probabilité retournée par le modèle "
                "n'est pas finie."
            )

        if not 0.0 <= probability_default <= 1.0:
            raise RuntimeError(
                "La probabilité retournée par le modèle "
                "n'est pas comprise entre 0 et 1."
            )

        return probability_default


# -------------------------------------------------------------------
# Instance globale
# -------------------------------------------------------------------

model_service = ModelService()