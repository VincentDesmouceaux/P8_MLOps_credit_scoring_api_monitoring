from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Dictionnaire contenant les 656 features attendues par le modèle.",
    )


class PredictionResponse(BaseModel):
    probability_default: float
    prediction: int
    prediction_label: str
    threshold: float