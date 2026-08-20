import json
from pathlib import Path
from typing import Any

import mlflow.xgboost
import numpy as np
import pandas as pd


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


# -------------------------------------------------------------------
# Service modèle
# -------------------------------------------------------------------

class ModelService:
    def __init__(self) -> None:
        self.model: Any | None = None
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
            self.feature_names = json.load(
                file
            )

        if not isinstance(
            self.feature_names,
            list,
        ):
            raise TypeError(
                "feature_names.json doit contenir "
                "une liste de noms de features."
            )

        if len(self.feature_names) != 656:
            raise RuntimeError(
                "Le modèle P8 doit utiliser exactement "
                f"656 features. Reçu : "
                f"{len(self.feature_names)}."
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

        Le modèle reste ensuite en mémoire et est réutilisé
        pour toutes les prédictions.
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

        self.loaded = True

        print(
            "Modèle chargé."
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
    # Construction du DataFrame d'inférence
    # ----------------------------------------------------------------

    def build_input_dataframe(
        self,
        features: dict[str, float | None],
    ) -> pd.DataFrame:
        """
        Reconstruit une observation dans l'ordre exact
        des 656 features utilisées lors de l'entraînement.

        Les valeurs JSON null deviennent None dans FastAPI,
        puis np.nan ici afin de préserver les données
        manquantes originales pour XGBoost.
        """
        values = []

        for feature_name in self.feature_names:
            value = features[
                feature_name
            ]

            if value is None:
                values.append(
                    np.nan
                )

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

            values.append(
                numeric_value
            )

        input_df = pd.DataFrame(
            [
                values,
            ],
            columns=self.feature_names,
            dtype=float,
        )

        if input_df.shape != (
            1,
            len(self.feature_names),
        ):
            raise RuntimeError(
                "Le DataFrame d'inférence n'a pas "
                "la forme attendue."
            )

        if (
            input_df.columns.tolist()
            != self.feature_names
        ):
            raise RuntimeError(
                "L'ordre des features d'inférence "
                "ne correspond pas au modèle."
            )

        return input_df

    # ----------------------------------------------------------------
    # Prédiction
    # ----------------------------------------------------------------

    def predict_proba(
        self,
        features: dict[str, float | None],
    ) -> float:
        """
        Retourne la probabilité de défaut du client.
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

        input_df = self.build_input_dataframe(
            features
        )

        if self.model is None:
            raise RuntimeError(
                "Le modèle n'est pas chargé."
            )

        probabilities = self.model.predict_proba(
            input_df
        )

        if (
            len(probabilities) != 1
            or len(probabilities[0]) < 2
        ):
            raise RuntimeError(
                "Format inattendu retourné par predict_proba()."
            )

        probability_default = float(
            probabilities[0][1]
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