import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
)

from app.monitoring.prediction_logger import save_prediction_log
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
)
from app.security.api_key import verify_api_key
from app.services.model_service import model_service


# -------------------------------------------------------------------
# Configuration API
# -------------------------------------------------------------------

API_TITLE = "P8 Credit Scoring API"
API_VERSION = "0.5.0"
API_SERVICE_NAME = "p8-credit-scoring-api"


# -------------------------------------------------------------------
# Configuration modèle
# -------------------------------------------------------------------

MODEL_NAME = "P6_credit_scoring_default_risk_model"
MODEL_VERSION = "2"

# Le modèle métier reste un XGBoost.
MODEL_FAMILY = "XGBoost"

# Runtime utilisé en production après optimisation.
MODEL_RUNTIME = "ONNX Runtime"

ONNX_PROVIDER = "CPUExecutionProvider"

MLFLOW_ALIAS = "champion"

DECISION_THRESHOLD = 0.45


# -------------------------------------------------------------------
# Cycle de vie FastAPI
# -------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Charge la session ONNX Runtime une seule fois
    au démarrage de l'API.

    Le modèle reste ensuite en mémoire et est réutilisé
    pour toutes les requêtes de prédiction.
    """
    model_service.load()

    yield


app = FastAPI(
    title=API_TITLE,
    description=(
        "API de mise en production du modèle de scoring crédit "
        "issu du projet P6, optimisée avec ONNX Runtime."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)


# -------------------------------------------------------------------
# Endpoints techniques
# -------------------------------------------------------------------

@app.get("/health")
def health_check() -> dict[str, str]:
    """
    Vérifie que l'API est disponible.
    """
    return {
        "status": "ok",
        "service": API_SERVICE_NAME,
        "version": API_VERSION,
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    """
    Retourne les principales informations
    du modèle actuellement déployé.
    """
    active_providers: list[str] = []

    if (
        model_service.loaded
        and model_service.session is not None
    ):
        active_providers = (
            model_service.session.get_providers()
        )

    return {
        "model_name": MODEL_NAME,
        "model_version": int(
            MODEL_VERSION
        ),
        "model_family": MODEL_FAMILY,
        "model_runtime": MODEL_RUNTIME,
        "mlflow_alias": MLFLOW_ALIAS,
        "decision_threshold": DECISION_THRESHOLD,
        "n_features": len(
            model_service.feature_names
        ),
        "loaded": model_service.loaded,
        "inference_pipeline": (
            "JSON features -> NumPy float32 -> "
            "ONNX Runtime"
        ),
        "onnx_provider": ONNX_PROVIDER,
        "active_onnx_providers": (
            active_providers
        ),
        "monitoring_persistence": (
            "FastAPI BackgroundTasks + "
            "PostgreSQL/Supabase"
        ),
        "deploy_commit": os.getenv(
            "RENDER_GIT_COMMIT",
            "local",
        ),
    }


# -------------------------------------------------------------------
# Utilitaires de mesure
# -------------------------------------------------------------------

def compute_latency_ms(
    start_time: float,
) -> float:
    """
    Calcule le temps écoulé depuis start_time
    en millisecondes.
    """
    return (
        time.perf_counter()
        - start_time
    ) * 1000


# -------------------------------------------------------------------
# Logging structuré
# -------------------------------------------------------------------

def write_structured_log(
    *,
    level: str,
    event: str,
    request_id: UUID,
    status_code: int,
    latency_ms: float,
    probability_default: float | None = None,
    prediction: int | None = None,
    prediction_label: str | None = None,
    monitoring_storage_ms: float | None = None,
    handler_total_ms: float | None = None,
    error_message: str | None = None,
) -> None:
    """
    Écrit un événement JSON structuré
    dans les logs applicatifs.

    Les 656 features complètes ne sont volontairement
    pas écrites dans les logs Render.

    Elles sont persistées séparément dans
    PostgreSQL/Supabase pour le monitoring et le Data Drift.

    Mesures principales :
    - latency_ms :
      préparation de l'entrée + inférence ONNX ;
    - handler_total_ms :
      temps passé dans le handler avant génération
      de la réponse ;
    - monitoring_storage_ms :
      durée de la persistance Supabase exécutée
      en arrière-plan.
    """
    log_entry: dict[str, Any] = {
        "level": level,
        "event": event,
        "request_id": str(
            request_id
        ),
        "status_code": status_code,
        "latency_ms": round(
            latency_ms,
            3,
        ),
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_family": MODEL_FAMILY,
        "model_runtime": MODEL_RUNTIME,
        "decision_threshold": DECISION_THRESHOLD,
    }

    if probability_default is not None:
        log_entry[
            "probability_default"
        ] = float(
            probability_default
        )

    if prediction is not None:
        log_entry[
            "prediction"
        ] = int(
            prediction
        )

    if prediction_label is not None:
        log_entry[
            "prediction_label"
        ] = prediction_label

    if monitoring_storage_ms is not None:
        log_entry[
            "monitoring_storage_ms"
        ] = round(
            monitoring_storage_ms,
            3,
        )

    if handler_total_ms is not None:
        log_entry[
            "handler_total_ms"
        ] = round(
            handler_total_ms,
            3,
        )

    if error_message is not None:
        log_entry[
            "error_message"
        ] = error_message

    print(
        json.dumps(
            log_entry,
            ensure_ascii=False,
        )
    )


# -------------------------------------------------------------------
# Persistance monitoring
# -------------------------------------------------------------------

def log_prediction_safely(
    *,
    request_id: UUID,
    input_features: dict[str, float | None],
    probability_default: float | None,
    prediction: int | None,
    prediction_label: str | None,
    threshold: float,
    latency_ms: float,
    status_code: int,
    error_message: str | None,
) -> float:
    """
    Persiste les données de monitoring
    dans PostgreSQL/Supabase.

    Cette fonction est exécutée via BackgroundTasks
    afin de sortir le stockage PostgreSQL/Supabase
    du chemin critique de la réponse HTTP.

    Retourne la durée du stockage en millisecondes.
    """
    storage_start = time.perf_counter()

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
        storage_ms = compute_latency_ms(
            storage_start
        )

        write_structured_log(
            level="error",
            event="monitoring_storage_error",
            request_id=request_id,
            status_code=status_code,
            latency_ms=latency_ms,
            monitoring_storage_ms=storage_ms,
            error_message=str(
                log_error
            ),
        )

        return storage_ms

    return compute_latency_ms(
        storage_start
    )


def persist_prediction_in_background(
    *,
    request_id: UUID,
    input_features: dict[str, float | None],
    probability_default: float | None,
    prediction: int | None,
    prediction_label: str | None,
    threshold: float,
    latency_ms: float,
    status_code: int,
    error_message: str | None,
) -> None:
    """
    Persiste les données de monitoring
    après génération de la réponse HTTP.

    Le coût du stockage reste mesuré et tracé,
    mais il ne bloque plus le client.
    """
    monitoring_storage_ms = (
        log_prediction_safely(
            request_id=request_id,
            input_features=input_features,
            probability_default=probability_default,
            prediction=prediction,
            prediction_label=prediction_label,
            threshold=threshold,
            latency_ms=latency_ms,
            status_code=status_code,
            error_message=error_message,
        )
    )

    write_structured_log(
        level="info",
        event="monitoring_background_storage",
        request_id=request_id,
        status_code=status_code,
        latency_ms=latency_ms,
        monitoring_storage_ms=monitoring_storage_ms,
    )


# -------------------------------------------------------------------
# Endpoint de prédiction
# -------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[
        Depends(verify_api_key),
    ],
)
def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
) -> PredictionResponse:
    """
    Calcule la probabilité de défaut d'un client.

    Pipeline optimisé :
    - génère un request_id ;
    - valide et reconstruit les 656 features ;
    - exécute l'inférence via ONNX Runtime ;
    - applique le seuil métier de 0.45 ;
    - mesure la latence d'inférence ;
    - planifie la persistance Supabase en arrière-plan ;
    - retourne immédiatement la prédiction.
    """
    request_id = uuid4()

    handler_start = (
        time.perf_counter()
    )

    inference_start = (
        time.perf_counter()
    )

    try:
        # -----------------------------------------------------------
        # Inférence ONNX Runtime
        # -----------------------------------------------------------

        probability_default = float(
            model_service.predict_proba(
                request.features
            )
        )

        prediction = int(
            probability_default
            >= DECISION_THRESHOLD
        )

        prediction_label = (
            "client_risque"
            if prediction == 1
            else "client_non_risque"
        )

        latency_ms = compute_latency_ms(
            inference_start
        )

        # -----------------------------------------------------------
        # Persistance monitoring en arrière-plan
        # -----------------------------------------------------------

        background_tasks.add_task(
            persist_prediction_in_background,
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

        handler_total_ms = (
            compute_latency_ms(
                handler_start
            )
        )

        # -----------------------------------------------------------
        # Log du chemin critique
        # -----------------------------------------------------------

        write_structured_log(
            level="info",
            event="prediction_success",
            request_id=request_id,
            status_code=200,
            latency_ms=latency_ms,
            probability_default=probability_default,
            prediction=prediction,
            prediction_label=prediction_label,
            handler_total_ms=handler_total_ms,
        )

        # -----------------------------------------------------------
        # Réponse API
        # -----------------------------------------------------------

        return PredictionResponse(
            request_id=request_id,
            probability_default=probability_default,
            prediction=prediction,
            prediction_label=prediction_label,
            threshold=DECISION_THRESHOLD,
        )

    # ----------------------------------------------------------------
    # Erreur métier / features invalides
    # ----------------------------------------------------------------

    except ValueError as error:
        latency_ms = compute_latency_ms(
            inference_start
        )

        error_message = str(
            error
        )

        background_tasks.add_task(
            persist_prediction_in_background,
            request_id=request_id,
            input_features=request.features,
            probability_default=None,
            prediction=None,
            prediction_label=None,
            threshold=DECISION_THRESHOLD,
            latency_ms=latency_ms,
            status_code=422,
            error_message=error_message,
        )

        handler_total_ms = (
            compute_latency_ms(
                handler_start
            )
        )

        write_structured_log(
            level="warning",
            event="prediction_validation_error",
            request_id=request_id,
            status_code=422,
            latency_ms=latency_ms,
            handler_total_ms=handler_total_ms,
            error_message=error_message,
        )

        raise HTTPException(
            status_code=422,
            detail=error_message,
        ) from error

    # ----------------------------------------------------------------
    # Préserve les HTTPException explicites
    # ----------------------------------------------------------------

    except HTTPException:
        raise

    # ----------------------------------------------------------------
    # Erreur interne inattendue
    # ----------------------------------------------------------------

    except Exception as error:
        latency_ms = compute_latency_ms(
            inference_start
        )

        error_message = (
            "Erreur interne lors de la prédiction : "
            f"{error}"
        )

        background_tasks.add_task(
            persist_prediction_in_background,
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

        handler_total_ms = (
            compute_latency_ms(
                handler_start
            )
        )

        write_structured_log(
            level="error",
            event="prediction_internal_error",
            request_id=request_id,
            status_code=500,
            latency_ms=latency_ms,
            handler_total_ms=handler_total_ms,
            error_message=error_message,
        )

        raise HTTPException(
            status_code=500,
            detail=error_message,
        ) from error