import math

import numpy as np
import pytest

from app.services.model_service import (
    EXPECTED_FEATURE_COUNT,
    model_service,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def build_valid_features() -> dict[str, float | None]:
    """
    Construit un payload valide contenant exactement
    les features attendues par le modèle.
    """
    return {
        feature_name: 0.0
        for feature_name in model_service.feature_names
    }


# -------------------------------------------------------------------
# Validation simple des features
# -------------------------------------------------------------------

def test_validate_features_valid_payload():
    features = build_valid_features()

    validation = model_service.validate_features(
        features
    )

    assert validation["missing_features"] == []
    assert validation["extra_features"] == []


def test_validate_features_detects_missing_feature():
    features = build_valid_features()

    feature_to_remove = model_service.feature_names[0]

    del features[feature_to_remove]

    validation = model_service.validate_features(
        features
    )

    assert feature_to_remove in validation["missing_features"]
    assert validation["extra_features"] == []


def test_validate_features_detects_extra_feature():
    features = build_valid_features()

    features["FEATURE_INCONNUE"] = 123.0

    validation = model_service.validate_features(
        features
    )

    assert validation["missing_features"] == []
    assert "FEATURE_INCONNUE" in validation["extra_features"]


# -------------------------------------------------------------------
# Validation stricte du contrat
# -------------------------------------------------------------------

def test_validate_feature_contract_accepts_valid_payload():
    features = build_valid_features()

    model_service.validate_feature_contract(
        features
    )


def test_validate_feature_contract_rejects_missing_feature():
    features = build_valid_features()

    feature_to_remove = model_service.feature_names[0]

    del features[feature_to_remove]

    with pytest.raises(
        ValueError,
        match="features manquantes",
    ):
        model_service.validate_feature_contract(
            features
        )


def test_validate_feature_contract_rejects_extra_feature():
    features = build_valid_features()

    features["FEATURE_INCONNUE"] = 123.0

    with pytest.raises(
        ValueError,
        match="features inconnues",
    ):
        model_service.validate_feature_contract(
            features
        )


# -------------------------------------------------------------------
# Construction du tableau NumPy
# -------------------------------------------------------------------

def test_build_input_array_shape_and_dtype():
    features = build_valid_features()

    input_array = model_service.build_input_array(
        features
    )

    assert input_array.shape == (
        1,
        EXPECTED_FEATURE_COUNT,
    )

    assert input_array.dtype == np.float32


def test_build_input_array_preserves_none_as_nan():
    features = build_valid_features()

    first_feature = model_service.feature_names[0]

    features[first_feature] = None

    input_array = model_service.build_input_array(
        features
    )

    assert math.isnan(
        float(
            input_array[0, 0]
        )
    )


def test_build_input_array_rejects_non_numeric_value():
    features = build_valid_features()

    first_feature = model_service.feature_names[0]

    features[first_feature] = "texte_invalide"

    with pytest.raises(
        ValueError,
        match="Valeur non numérique",
    ):
        model_service.build_input_array(
            features
        )


def test_build_input_array_rejects_positive_infinity():
    features = build_valid_features()

    first_feature = model_service.feature_names[0]

    features[first_feature] = float("inf")

    with pytest.raises(
        ValueError,
        match="Valeur infinie interdite",
    ):
        model_service.build_input_array(
            features
        )


def test_build_input_array_rejects_negative_infinity():
    features = build_valid_features()

    first_feature = model_service.feature_names[0]

    features[first_feature] = float("-inf")

    with pytest.raises(
        ValueError,
        match="Valeur infinie interdite",
    ):
        model_service.build_input_array(
            features
        )


# -------------------------------------------------------------------
# Validation de la probabilité
# -------------------------------------------------------------------

def test_validate_probability_accepts_zero():
    result = model_service.validate_probability(
        0.0
    )

    assert result == 0.0


def test_validate_probability_accepts_one():
    result = model_service.validate_probability(
        1.0
    )

    assert result == 1.0


def test_validate_probability_accepts_valid_probability():
    result = model_service.validate_probability(
        0.75
    )

    assert result == 0.75


def test_validate_probability_rejects_nan():
    with pytest.raises(
        RuntimeError,
        match="n'est pas finie",
    ):
        model_service.validate_probability(
            float("nan")
        )


def test_validate_probability_rejects_positive_infinity():
    with pytest.raises(
        RuntimeError,
        match="n'est pas finie",
    ):
        model_service.validate_probability(
            float("inf")
        )


def test_validate_probability_rejects_negative_infinity():
    with pytest.raises(
        RuntimeError,
        match="n'est pas finie",
    ):
        model_service.validate_probability(
            float("-inf")
        )


def test_validate_probability_rejects_negative_value():
    with pytest.raises(
        RuntimeError,
        match="n'est pas comprise",
    ):
        model_service.validate_probability(
            -0.1
        )


def test_validate_probability_rejects_value_above_one():
    with pytest.raises(
        RuntimeError,
        match="n'est pas comprise",
    ):
        model_service.validate_probability(
            1.1
        )


# -------------------------------------------------------------------
# Prédiction réelle
# -------------------------------------------------------------------

def test_predict_proba_returns_float():
    features = build_valid_features()

    probability = model_service.predict_proba(
        features
    )

    assert isinstance(
        probability,
        float,
    )


def test_predict_proba_returns_valid_probability():
    features = build_valid_features()

    probability = model_service.predict_proba(
        features
    )

    assert (
        0.0
        <= probability
        <= 1.0
    )


def test_model_is_loaded_after_prediction():
    features = build_valid_features()

    model_service.predict_proba(
        features
    )

    assert model_service.loaded is True
    assert model_service.model is not None
    assert model_service.booster is not None