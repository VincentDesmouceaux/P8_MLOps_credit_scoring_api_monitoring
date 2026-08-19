from fastapi import Depends, FastAPI, HTTPException

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.security.api_key import verify_api_key
from app.services.model_service import model_service


app = FastAPI(
    title="P8 Credit Scoring API",
    description=(
        "API de mise en production du modèle de scoring crédit "
        "issu du projet P6."
    ),
    version="0.2.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "p8-credit-scoring-api",
        "version": "0.2.0",
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": "P6_credit_scoring_default_risk_model",
        "model_version": 2,
        "model_family": "XGBoost",
        "mlflow_alias": "champion",
        "decision_threshold": 0.45,
        "n_features": len(model_service.feature_names),
        "loaded": model_service.loaded,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
)
def predict(request: PredictionRequest):
    try:
        probability_default = model_service.predict_proba(
            request.features
        )

        threshold = 0.45
        prediction = int(
            probability_default >= threshold
        )

        return {
            "probability_default": probability_default,
            "prediction": prediction,
            "prediction_label": (
                "client_risque"
                if prediction == 1
                else "client_non_risque"
            ),
            "threshold": threshold,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur interne lors de la prédiction : "
                f"{error}"
            ),
        ) from error