import os
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException

from app.monitoring.prediction_logger import save_prediction_log
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.security.api_key import verify_api_key
from app.services.model_service import model_service


MODEL_NAME = "P6_credit_scoring_default_risk_model"
MODEL_VERSION = "2"
DECISION_THRESHOLD = 0.45


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Charge le modèle une seule fois au démarrage de l'API.

    Le même modèle reste ensuite en mémoire et est réutilisé
    pour toutes les requêtes de prédiction.
    """
    model_service.load()
    yield


app = FastAPI(
    title="P8 Credit Scoring API",
    description=(
        "API de mise en production du modèle de scoring crédit "
        "issu du projet P6."
    ),
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    """
    Vérifie que l'API est disponible.
    """
    return {
        "status": "ok",
        "service": "p8-credit-scoring-api",
        "version": "0.2.0",
    }


@app.get("/model-info")
def model_info():
    """
    Retourne les informations principales du modèle déployé.
    """
    return {
        "model_name": MODEL_NAME,
        "model_version": 2,
        "model_family": "XGBoost",
        "mlflow_alias": "champion",
        "decision_threshold": DECISION_THRESHOLD,
        "n_features": len(model_service.feature_names),
        "loaded": model_service.loaded,
        "deploy_commit": os.getenv("RENDER_GIT_COMMIT", "local"),
    }


def log_prediction_safely(
    *,
    request_id,
    input_features,
    probability_default,
    prediction,
    prediction_label,
    threshold,
    latency_ms,
    status_code,
    error_message,
):
    """
    Enregistre les données de monitoring sans interrompre l'API
    si le stockage PostgreSQL/Supabase est indisponible.
    """
    try:
        save_prediction_log(
            request_id=request_id,
            input_features=input_features,
            probability_default=probability_default,
            prediction=prediction,
            prediction_label=prediction_label,
            threshold=threshold,
            latency_ms=latency_ms,
            status_code=status_code,
            error_message=error_message,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
        )

    except Exception as log_error:
        print(
            f"Erreur de monitoring pour la requête "
            f"{request_id}: {log_error}"
        )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
)
def predict(request: PredictionRequest):
    """
    Calcule la probabilité de défaut d'un client,
    retourne la décision associée et enregistre
    les informations de monitoring en base.
    """
    request_id = uuid4()
    start_time = time.perf_counter()

    try:
        probability_default = model_service.predict_proba(
            request.features
        )

        prediction = int(
            probability_default >= DECISION_THRESHOLD
        )

        prediction_label = (
            "client_risque"
            if prediction == 1
            else "client_non_risque"
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        log_prediction_safely(
            request_id=request_id,
            input_features=request.features,
            probability_default=probability_default,
            prediction=prediction,
            prediction_label=prediction_label,
            threshold=DECISION_THRESHOLD,
            latency_ms=latency_ms,
            status_code=200,
            error_message=None,
        )

        return {
            "probability_default": probability_default,
            "prediction": prediction,
            "prediction_label": prediction_label,
            "threshold": DECISION_THRESHOLD,
        }

    except ValueError as error:
        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        log_prediction_safely(
            request_id=request_id,
            input_features=request.features,
            probability_default=None,
            prediction=None,
            prediction_label=None,
            threshold=DECISION_THRESHOLD,
            latency_ms=latency_ms,
            status_code=422,
            error_message=str(error),
        )

        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        error_message = (
            "Erreur interne lors de la prédiction : "
            f"{error}"
        )

        log_prediction_safely(
            request_id=request_id,
            input_features=request.features,
            probability_default=None,
            prediction=None,
            prediction_label=None,
            threshold=DECISION_THRESHOLD,
            latency_ms=latency_ms,
            status_code=500,
            error_message=error_message,
        )

        raise HTTPException(
            status_code=500,
            detail=error_message,
        ) from error