from pathlib import Path
import json

import mlflow.xgboost
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "credit_scoring_model"
FEATURE_NAMES_PATH = PROJECT_ROOT / "models" / "feature_names.json"


class ModelService:
    def __init__(self):
        self.model = None
        self.loaded = False

        with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as file:
            self.feature_names = json.load(file)

    def load(self):
        if self.loaded:
            return

        print("Chargement du modèle...")

        self.model = mlflow.xgboost.load_model(str(MODEL_PATH))
        self.loaded = True

        print("Modèle chargé.")
        print("Nombre de features attendues :", len(self.feature_names))

    def validate_features(self, features: dict):
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

    def predict_proba(self, features: dict) -> float:
        self.load()

        validation = self.validate_features(features)

        if validation["missing_features"]:
            raise ValueError(
                f"{len(validation['missing_features'])} features manquantes."
            )

        if validation["extra_features"]:
            raise ValueError(
                f"{len(validation['extra_features'])} features inconnues."
            )

        input_df = pd.DataFrame(
            [[features[name] for name in self.feature_names]],
            columns=self.feature_names,
        )

        probability_default = self.model.predict_proba(input_df)[0][1]

        return float(probability_default)


model_service = ModelService()