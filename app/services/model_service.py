import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ONNX_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "onnx"
    / "credit_scoring_model.onnx"
)

FEATURE_NAMES_PATH = (
    PROJECT_ROOT
    / "models"
    / "feature_names.json"
)

EXPECTED_FEATURE_COUNT = 656

ONNX_PROVIDER = "CPUExecutionProvider"


# -------------------------------------------------------------------
# Service modèle
# -------------------------------------------------------------------

class ModelService:
    def __init__(self) -> None:
        """
        Initialise la configuration du service d'inférence.

        Le modèle ONNX n'est pas chargé ici afin de conserver
        un chargement explicite au démarrage de FastAPI.
        """
        self.session: ort.InferenceSession | None = None

        self.input_name: str | None = None
        self.probabilities_output_name: str | None = None

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
        Charge le modèle ONNX une seule fois.

        ONNX Runtime est configuré avec CPUExecutionProvider,
        ce qui correspond à l'environnement de production Render
        et à la configuration utilisée pendant les benchmarks.
        """
        if self.loaded:
            return

        if not ONNX_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Artefact ONNX introuvable : "
                f"{ONNX_MODEL_PATH}"
            )

        print(
            "Chargement du modèle ONNX..."
        )

        # -----------------------------------------------------------
        # Configuration ONNX Runtime
        # -----------------------------------------------------------

        session_options = (
            ort.SessionOptions()
        )

        try:
            session = ort.InferenceSession(
                str(ONNX_MODEL_PATH),
                sess_options=session_options,
                providers=[
                    ONNX_PROVIDER,
                ],
            )

        except Exception as error:
            raise RuntimeError(
                "Impossible de charger le modèle "
                "avec ONNX Runtime."
            ) from error

        # -----------------------------------------------------------
        # Validation provider
        # -----------------------------------------------------------

        active_providers = (
            session.get_providers()
        )

        if ONNX_PROVIDER not in active_providers:
            raise RuntimeError(
                "Le provider ONNX Runtime attendu "
                f"'{ONNX_PROVIDER}' n'est pas actif. "
                f"Providers disponibles : "
                f"{active_providers}"
            )

        # -----------------------------------------------------------
        # Validation entrée ONNX
        # -----------------------------------------------------------

        inputs = session.get_inputs()

        if len(inputs) != 1:
            raise RuntimeError(
                "Le modèle ONNX doit exposer "
                "exactement une entrée."
            )

        input_info = inputs[0]

        if input_info.type != "tensor(float)":
            raise RuntimeError(
                "Le modèle ONNX doit accepter "
                "une entrée float32. "
                f"Type reçu : {input_info.type}."
            )

        input_shape = input_info.shape

        if len(input_shape) != 2:
            raise RuntimeError(
                "L'entrée ONNX doit être "
                "un tenseur à deux dimensions."
            )

        feature_dimension = (
            input_shape[1]
        )

        if (
            isinstance(
                feature_dimension,
                int,
            )
            and feature_dimension
            != EXPECTED_FEATURE_COUNT
        ):
            raise RuntimeError(
                "Le modèle ONNX n'attend pas "
                f"{EXPECTED_FEATURE_COUNT} features. "
                f"Shape reçue : {input_shape}."
            )

        # -----------------------------------------------------------
        # Recherche de la sortie probabilities
        # -----------------------------------------------------------

        outputs = session.get_outputs()

        probabilities_output = None

        for output_info in outputs:
            if (
                output_info.name
                == "probabilities"
            ):
                probabilities_output = (
                    output_info
                )
                break

        if probabilities_output is None:
            for output_info in outputs:
                output_shape = (
                    output_info.shape
                )

                if (
                    output_info.type
                    == "tensor(float)"
                    and len(output_shape) == 2
                ):
                    probabilities_output = (
                        output_info
                    )
                    break

        if probabilities_output is None:
            raise RuntimeError(
                "Impossible d'identifier la sortie "
                "de probabilités du modèle ONNX."
            )

        # -----------------------------------------------------------
        # Affectation uniquement après validation complète
        # -----------------------------------------------------------

        self.session = session
        self.input_name = (
            input_info.name
        )

        self.probabilities_output_name = (
            probabilities_output.name
        )

        self.loaded = True

        print(
            "Modèle ONNX chargé."
        )

        print(
            "Pipeline d'inférence : "
            "NumPy float32 + ONNX Runtime"
        )

        print(
            "Provider :",
            ONNX_PROVIDER,
        )

        print(
            "Entrée ONNX :",
            self.input_name,
        )

        print(
            "Sortie probabilités :",
            self.probabilities_output_name,
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
        les 656 features attendues.
        """
        missing_features = [
            feature_name
            for feature_name
            in self.feature_names
            if feature_name
            not in features
        ]

        extra_features = [
            feature_name
            for feature_name
            in features
            if feature_name
            not in self.feature_names
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
    # Construction de l'entrée NumPy
    # ----------------------------------------------------------------

    def build_input_array(
        self,
        features: dict[str, float | None],
    ) -> np.ndarray:
        """
        Reconstruit une observation dans l'ordre exact
        des 656 features du modèle.

        JSON null -> Python None -> np.nan.

        L'entrée finale est un tableau NumPy float32
        de shape (1, 656), format attendu par ONNX Runtime.
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

            # NaN est autorisé car il représente
            # une valeur manquante du dataset P6.
            #
            # Les valeurs infinies sont interdites.

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
    # Extraction de la probabilité
    # ----------------------------------------------------------------

    def extract_probability_default(
        self,
        probabilities: np.ndarray,
    ) -> float:
        """
        Extrait la probabilité de la classe positive
        correspondant au défaut client.

        Le modèle exporté fournit une matrice :
        [[P(classe 0), P(classe 1)]].
        """
        if not isinstance(
            probabilities,
            np.ndarray,
        ):
            raise RuntimeError(
                "La sortie probabilities du modèle "
                "ONNX n'est pas un tableau NumPy."
            )

        if probabilities.shape != (
            1,
            2,
        ):
            raise RuntimeError(
                "Shape inattendue pour la sortie "
                "probabilities : "
                f"{probabilities.shape}."
            )

        probability_default = float(
            probabilities[0][1]
        )

        if not np.isfinite(
            probability_default
        ):
            raise RuntimeError(
                "La probabilité retournée par "
                "ONNX Runtime n'est pas finie."
            )

        if not (
            0.0
            <= probability_default
            <= 1.0
        ):
            raise RuntimeError(
                "La probabilité retournée par "
                "le modèle n'est pas comprise "
                "entre 0 et 1."
            )

        return probability_default

    # ----------------------------------------------------------------
    # Prédiction ONNX Runtime
    # ----------------------------------------------------------------

    def predict_proba(
        self,
        features: dict[str, float | None],
    ) -> float:
        """
        Retourne la probabilité de défaut du client.

        Pipeline final optimisé :

        dict
          -> validation
          -> NumPy float32 (1, 656)
          -> ONNX Runtime CPU
          -> probabilité classe 1

        Le contrat public predict_proba() reste identique
        à celui des précédentes versions du service.
        """
        self.load()

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

        input_array = (
            self.build_input_array(
                features
            )
        )

        if self.session is None:
            raise RuntimeError(
                "La session ONNX Runtime "
                "n'est pas chargée."
            )

        if self.input_name is None:
            raise RuntimeError(
                "Le nom de l'entrée ONNX "
                "n'est pas disponible."
            )

        if (
            self.probabilities_output_name
            is None
        ):
            raise RuntimeError(
                "Le nom de la sortie probabilities "
                "n'est pas disponible."
            )

        try:
            outputs = self.session.run(
                [
                    self.probabilities_output_name,
                ],
                {
                    self.input_name: (
                        input_array
                    ),
                },
            )

        except Exception as error:
            raise RuntimeError(
                "Erreur lors de l'inférence "
                "ONNX Runtime."
            ) from error

        if len(outputs) != 1:
            raise RuntimeError(
                "Nombre de sorties ONNX "
                "inattendu."
            )

        return (
            self.extract_probability_default(
                outputs[0]
            )
        )


# -------------------------------------------------------------------
# Instance globale
# -------------------------------------------------------------------

model_service = ModelService()