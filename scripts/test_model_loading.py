from pathlib import Path

import mlflow.xgboost


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "credit_scoring_model"


def main():
    print("Chargement du modèle depuis :", MODEL_PATH)

    model = mlflow.xgboost.load_model(str(MODEL_PATH))

    print("Modèle chargé avec succès.")
    print("Type :", type(model))


if __name__ == "__main__":
    main()