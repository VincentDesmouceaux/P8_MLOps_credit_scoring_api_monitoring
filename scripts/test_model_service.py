import json

from app.services.model_service import model_service


with open("models/feature_names.json", "r", encoding="utf-8") as file:
    feature_names = json.load(file)

print("Service chargé avant prédiction :", model_service.loaded)
print("Nombre de features connues :", len(feature_names))

test_features = {
    feature: 0.0
    for feature in feature_names
}

validation = model_service.validate_features(test_features)

print("Features manquantes :", len(validation["missing_features"]))
print("Features inconnues :", len(validation["extra_features"]))

probability = model_service.predict_proba(test_features)

print("Probabilité de défaut :", probability)
print("Service chargé après prédiction :", model_service.loaded)