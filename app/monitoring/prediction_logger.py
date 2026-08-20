import json
from uuid import UUID

from app.core.database import get_database_connection


def save_prediction_log(
    request_id: UUID,
    input_features: dict,
    probability_default: float | None,
    prediction: int | None,
    prediction_label: str | None,
    threshold: float | None,
    latency_ms: float,
    status_code: int,
    error_message: str | None,
    model_name: str,
    model_version: str,
) -> None:
    query = """
        INSERT INTO prediction_logs (
            request_id,
            input_features,
            probability_default,
            prediction,
            prediction_label,
            threshold,
            latency_ms,
            status_code,
            error_message,
            model_name,
            model_version
        )
        VALUES (
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
    """

    values = (
        request_id,
        json.dumps(input_features),
        probability_default,
        prediction,
        prediction_label,
        threshold,
        latency_ms,
        status_code,
        error_message,
        model_name,
        model_version,
    )

    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, values)

        connection.commit()