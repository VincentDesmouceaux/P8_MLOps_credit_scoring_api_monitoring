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
        """
        Initialise le service d'inférence XGBoost.

        Le modèle n'est pas chargé immédiatement afin de conserver
        un chargement explicite au démarrage de FastAPI.
        """
        self.model: Any | None = None
        self.booster: Any | None = None
        self.loaded = False

        # -----------------------------------------------------------
        # Chargement des noms de features
        # -----------------------------------------------------------

        if not FEATURE_NAMES_PATH.exists():
            raise FileNotFoundError(
                "Fichier de features introuvable : "
                f"{FEATURE_NAMES_PATH}"
            )

        with open(
            FEATURE_NAMES_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            feature_names = json.load(
                file
            )

        # -----------------------------------------------------------
        # Validation feature_names.json
        # -----------------------------------------------------------

        if not isinstance(
            feature_names,
            list,
        ):
            raise TypeError(
                "feature_names.json doit contenir "
                "une liste de noms de features."
            )

        if not all(
            isinstance(feature_name, str)
            for feature_name in feature_names
        ):
            raise TypeError(
                "Tous les noms de features doivent "
                "être des chaînes de caractères."
            )

        if len(feature_names) != EXPECTED_FEATURE_COUNT:
            raise RuntimeError(
                "Le modèle P8 doit utiliser exactement "
                f"{EXPECTED_FEATURE_COUNT} features. "
                f"Reçu : {len(feature_names)}."
            )

        if len(
            set(feature_names)
        ) != len(
            feature_names
        ):
            raise RuntimeError(
                "Des noms de features dupliqués "
                "ont été détectés."
            )

        self.feature_names: list[str] = (
            feature_names
        )

    # ----------------------------------------------------------------
    # Chargement du modèle
    # ----------------------------------------------------------------

    def load(self) -> None:
        """
        Charge le modèle MLflow/XGBoost une seule fois.

        Le Booster natif est extrait au démarrage afin d'utiliser
        directement Booster.inplace_predict() pendant l'inférence.

        Cette configuration a été retenue après comparaison avec
        ONNX Runtime car elle offre le meilleur temps de réponse
        HTTP dans l'environnement de production Render.
        """
        if self.loaded:
            return

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Artefact modèle introuvable : "
                f"{MODEL_PATH}"
            )

        print(
            "Chargement du modèle XGBoost..."
        )

        try:
            model = mlflow.xgboost.load_model(
                str(MODEL_PATH)
            )

        except Exception as error:
            raise RuntimeError(
                "Impossible de charger le modèle "
                "MLflow/XGBoost."
            ) from error

        if model is None:
            raise RuntimeError(
                "Le chargement du modèle a échoué."
            )

        try:
            booster = model.get_booster()

        except AttributeError as error:
            raise RuntimeError(
                "Impossible d'extraire le Booster "
                "XGBoost du modèle."
            ) from error

        if booster is None:
            raise RuntimeError(
                "Le Booster XGBoost "
                "n'est pas disponible."
            )

        # Affectation seulement après validation complète.
        self.model = model
        self.booster = booster
        self.loaded = True

        print(
            "Modèle XGBoost chargé."
        )

        print(
            "Pipeline d'inférence : "
            "NumPy float32 + "
            "XGBoost Booster.inplace_predict"
        )

        print(
            "Nombre de features attendues :",
            len(
                self.feature_names
            ),
        )

    # ----------------------------------------------------------------
    # Validation des features
    # ----------------------------------------------------------------

    def validate_features(
        self,
        features: dict[str, float | None],
    ) -> dict[str, list[str]]:
        """
        Vérifie que le payload contient exactement
        les 656 features attendues par le modèle.
        """
        missing_features = [
            feature_name
            for feature_name in self.feature_names
            if feature_name not in features
        ]

        extra_features = [
            feature_name
            for feature_name in features
            if feature_name not in self.feature_names
        ]

        return {
            "missing_features": missing_features,
            "extra_features": extra_features,
        }

    # ----------------------------------------------------------------
    # Construction de l'entrée NumPy
    # ----------------------------------------------------------------

    def build_input_array(
        self,
        features: dict[str, float | None],
    ) -> np.ndarray:
        """
        Reconstruit une observation dans l'ordre exact
        des 656 features utilisées lors de l'entraînement.

        Les valeurs JSON null deviennent None dans FastAPI,
        puis np.nan afin de préserver les valeurs manquantes.

        L'utilisation directe de NumPy float32 évite la création
        d'un DataFrame pandas à chaque requête, goulot
        d'étranglement identifié lors du profiling cProfile.
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
                    "Valeur non numérique pour "
                    f"la feature '{feature_name}'."
                ) from error

            if np.isinf(
                numeric_value
            ):
                raise ValueError(
                    "Valeur infinie interdite pour "
                    f"la feature '{feature_name}'."
                )

            values[index] = (
                numeric_value
            )

        input_array = values.reshape(
            1,
            EXPECTED_FEATURE_COUNT,
        )

        if input_array.shape != (
            1,
            EXPECTED_FEATURE_COUNT,
        ):
            raise RuntimeError(
                "Le tableau NumPy d'inférence "
                "n'a pas la forme attendue."
            )

        if input_array.dtype != np.float32:
            raise RuntimeError(
                "Le tableau d'inférence doit "
                "être en float32."
            )

        return input_array

    # ----------------------------------------------------------------
    # Prédiction
    # ----------------------------------------------------------------

    def predict_proba(
        self,
        features: dict[str, float | None],
    ) -> float:
        """
        Retourne la probabilité de défaut du client.

        Pipeline final retenu en production :

        dict
          -> validation
          -> NumPy float32 (1, 656)
          -> XGBoost Booster.inplace_predict()
          -> probabilité de défaut
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
                f"Exemples : "
                f"{missing_features[:5]}"
            )

        if extra_features:
            raise ValueError(
                f"{len(extra_features)} "
                "features inconnues. "
                f"Exemples : "
                f"{extra_features[:5]}"
            )

        input_array = self.build_input_array(
            features
        )

        if self.booster is None:
            raise RuntimeError(
                "Le Booster XGBoost "
                "n'est pas chargé."
            )

        try:
            predictions = (
                self.booster.inplace_predict(
                    input_array
                )
            )

        except Exception as error:
            raise RuntimeError(
                "Erreur lors de l'inférence "
                "XGBoost."
            ) from error

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
                "La probabilité retournée "
                "par XGBoost n'est pas finie."
            )

        if not (
            0.0
            <= probability_default
            <= 1.0
        ):
            raise RuntimeError(
                "La probabilité retournée "
                "par le modèle n'est pas comprise "
                "entre 0 et 1."
            )

        return probability_default


# -------------------------------------------------------------------
# Instance globale
# -------------------------------------------------------------------

model_service = ModelService()