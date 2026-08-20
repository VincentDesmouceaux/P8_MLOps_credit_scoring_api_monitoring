from uuid import UUID

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: dict[str, float | None] = Field(
        ...,
        description=(
            "Dictionnaire contenant les 656 features attendues "
            "par le modèle. Les valeurs manquantes peuvent être null."
        ),
    )


class PredictionResponse(BaseModel):
    request_id: UUID
    probability_default: float
    prediction: int
    prediction_label: str
    threshold: float