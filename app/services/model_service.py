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
    """
    Service responsable du chargement du modèle,
    de la validation des features et de l'inférence.

    Pipeline de production :

    dict
        -> validation
        -> NumPy float32
        -> XGBoost Booster.inplace_predict
        -> probabilité de défaut
    """

    def __init__(self) -> None:
        self.model: Any | None = None
        self.booster: Any | None = None
        self.loaded = False

        self.feature_names = (
            self._load_feature_names()
        )

    # ----------------------------------------------------------------
    # Chargement des noms de features
    # ----------------------------------------------------------------

    def _load_feature_names(
        self,
    ) -> list[str]:
        """
        Charge et valide le fichier feature_names.json.
        """
        if not FEATURE_NAMES_PATH.exists():
            raise FileNotFoundError(
                "Fichier de features introuvable : "
                f"{FEATURE_NAMES_PATH}"
            )

        try:
            with open(
                FEATURE_NAMES_PATH,
                "r",
                encoding="utf-8",
            ) as file:
                feature_names = json.load(
                    file
                )

        except json.JSONDecodeError as error:
            raise RuntimeError(
                "feature_names.json "
                "n'est pas un JSON valide."
            ) from error

        if not isinstance(
            feature_names,
            list,
        ):
            raise TypeError(
                "feature_names.json doit contenir "
                "une liste de noms de features."
            )

        if not all(
            isinstance(
                feature_name,
                str,
            )
            for feature_name in feature_names
        ):
            raise TypeError(
                "Tous les noms de features doivent "
                "être des chaînes de caractères."
            )

        if len(
            feature_names
        ) != EXPECTED_FEATURE_COUNT:
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

        return feature_names

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
                "Artefact modèle introuvable : "
                f"{MODEL_PATH}"
            )

        print(
            "Chargement du modèle XGBoost..."
        )

        try:
            model = mlflow.xgboost.load_model(
                str(
                    MODEL_PATH
                )
            )

        except Exception as error:
            raise RuntimeError(
                "Impossible de charger le modèle "
                "MLflow/XGBoost."
            ) from error

        if model is None:
            raise RuntimeError(
                "Le chargement du modèle "
                "a retourné None."
            )

        try:
            booster = model.get_booster()

        except AttributeError as error:
            raise RuntimeError(
                "Impossible d'extraire le Booster "
                "XGBoost du modèle chargé."
            ) from error

        if booster is None:
            raise RuntimeError(
                "Le Booster XGBoost "
                "n'est pas disponible."
            )

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
        features: dict[
            str,
            float | None,
        ],
    ) -> dict[str, list[str]]:
        """
        Vérifie quelles features sont manquantes
        ou inconnues.
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
            "missing_features": (
                missing_features
            ),
            "extra_features": (
                extra_features
            ),
        }

    # ----------------------------------------------------------------
    # Validation du contrat des features
    # ----------------------------------------------------------------

    def validate_feature_contract(
        self,
        features: dict[
            str,
            float | None,
        ],
    ) -> None:
        """
        Vérifie que le payload correspond exactement
        au contrat attendu par le modèle.
        """
        validation = (
            self.validate_features(
                features
            )
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

    # ----------------------------------------------------------------
    # Construction du tableau NumPy
    # ----------------------------------------------------------------

    def build_input_array(
        self,
        features: dict[
            str,
            float | None,
        ],
    ) -> np.ndarray:
        """
        Reconstruit une observation dans l'ordre exact
        des 656 features utilisées lors de l'entraînement.

        Les valeurs JSON null deviennent None dans FastAPI,
        puis np.nan ici afin de préserver les valeurs manquantes.

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
                values[
                    index
                ] = np.nan

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

            values[
                index
            ] = numeric_value

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

        if (
            input_array.dtype
            != np.float32
        ):
            raise RuntimeError(
                "Le tableau NumPy d'inférence "
                "doit être en float32."
            )

        return input_array

    # ----------------------------------------------------------------
    # Validation de la probabilité
    # ----------------------------------------------------------------

    @staticmethod
    def validate_probability(
        probability: float,
    ) -> float:
        """
        Vérifie que la sortie du modèle est
        une probabilité finie entre 0 et 1.
        """
        if not np.isfinite(
            probability
        ):
            raise RuntimeError(
                "La probabilité retournée "
                "par le modèle n'est pas finie."
            )

        if not (
            0.0
            <= probability
            <= 1.0
        ):
            raise RuntimeError(
                "La probabilité retournée "
                "par le modèle n'est pas comprise "
                "entre 0 et 1."
            )

        return probability

    # ----------------------------------------------------------------
    # Prédiction optimisée
    # ----------------------------------------------------------------

    def predict_proba(
        self,
        features: dict[
            str,
            float | None,
        ],
    ) -> float:
        """
        Retourne la probabilité de défaut du client.

        Pipeline optimisé :

        dict
            -> validation du contrat
            -> NumPy float32
            -> Booster.inplace_predict
            -> validation de la probabilité
        """
        self.load()

        self.validate_feature_contract(
            features
        )

        input_array = (
            self.build_input_array(
                features
            )
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

        if len(
            predictions
        ) != 1:
            raise RuntimeError(
                "Format inattendu retourné par "
                "Booster.inplace_predict()."
            )

        probability_default = float(
            predictions[
                0
            ]
        )

        return (
            self.validate_probability(
                probability_default
            )
        )


# -------------------------------------------------------------------
# Instance globale
# -------------------------------------------------------------------

model_service = ModelService()