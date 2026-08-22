# 💳 P8 — MLOps Credit Scoring API & Monitoring

## 🎯 Présentation du projet

Ce projet a pour objectif de mettre en production un modèle de **scoring crédit** développé dans le projet P6 pour l'entreprise fictive **« Prêt à Dépenser »**.

Le modèle de Machine Learning existe déjà : il a été entraîné, évalué et versionné avec **MLflow** lors du projet précédent.

Le P8 consiste donc à franchir l'étape suivante du cycle MLOps : **transformer ce modèle en un véritable service de prédiction exploitable en production**.

Le projet couvre l'ensemble de la chaîne :

```text
Modèle MLflow / XGBoost
        ↓
API FastAPI
        ↓
Sécurisation par API Key
        ↓
Docker
        ↓
CI/CD GitHub Actions
        ↓
Render
        ↓
Monitoring PostgreSQL / Supabase
        ↓
Dashboard Streamlit
        ↓
Data Drift
        ↓
Profiling et optimisation
```

L'objectif n'est donc pas seulement d'obtenir une prédiction, mais de disposer d'une solution :

- 🚀 déployable ;
- 🔐 sécurisée ;
- 🧪 testée automatiquement ;
- 📊 monitorée ;
- 🐳 conteneurisée ;
- 🔄 déployée automatiquement ;
- 📈 capable de détecter une évolution des données ;
- ⚡ optimisée pour réduire les temps de réponse ;
- 🛡️ protégée contre les régressions.

---

# 🏗️ 1. Architecture générale

```text
                         ┌──────────────────┐
                         │      Client      │
                         └────────┬─────────┘
                                  │
                                  │ HTTP
                                  ▼
                    ┌──────────────────────────┐
                    │         FastAPI          │
                    │                          │
                    │  🔐 X-API-Key            │
                    │  ✓ Validation features   │
                    │  ⏱️ Mesure latence       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      ModelService        │
                    │                          │
                    │ dict                     │
                    │   ↓                      │
                    │ NumPy float32            │
                    │   ↓                      │
                    │ XGBoost Booster          │
                    │ .inplace_predict()       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                       Score de probabilité
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
             Réponse HTTP              BackgroundTasks
                                              │
                                              ▼
                                  PostgreSQL / Supabase
                                              │
                           ┌──────────────────┼───────────────┐
                           ▼                  ▼               ▼
                       Monitoring        Data Drift       Dashboard
```

### ⚡ Runtime final

Après plusieurs expérimentations, le pipeline final retenu est :

```text
JSON
 ↓
dict Python
 ↓
NumPy float32
 ↓
XGBoost Booster.inplace_predict()
 ↓
Probabilité de défaut
```

Cette architecture évite notamment la création répétée de `DataFrame pandas` pendant l'inférence.

La configuration finale est :

```text
NumPy float32 + XGBoost Booster.inplace_predict
```

**ONNX Runtime** a également été implémenté et benchmarké. Il constitue une expérimentation d'optimisation, mais n'a finalement pas été retenu comme runtime de production.

---

# 🧠 2. Modèle de scoring

Le modèle utilisé dans ce projet provient du **P6**.

Le P8 ne réentraîne donc pas un nouveau modèle à partir de zéro : il réutilise l'artefact précédemment produit et construit autour de lui l'infrastructure nécessaire à sa mise en production.

Modèle :

```text
P6_credit_scoring_default_risk_model
```

Famille :

```text
XGBoost
```

Version :

```text
2
```

Alias MLflow :

```text
champion
```

Seuil métier :

```text
0.45
```

Nombre de variables d'entrée :

```text
656 features
```

Le modèle retourne une **probabilité de défaut** utilisée pour produire la décision finale.

---

# ⚙️ 3. Pré-requis

L'environnement principal repose sur :

- 🐍 Python 3.12 ;
- 📦 `uv` ;
- 🐳 Docker ;
- 🗄️ PostgreSQL / Supabase ;
- 🚀 Render ;
- 🔄 GitHub Actions.

Installer les dépendances :

```bash
uv sync --frozen
```

Cette commande reconstruit l'environnement à partir du lockfile afin d'obtenir des versions reproductibles.

---

# 🔐 4. Variables d'environnement

Créer un fichier :

```text
.env
```

à la racine du projet.

Exemple :

```env
API_KEY=your_api_key
API_URL=https://your-api.onrender.com
DATABASE_URL=postgresql://...
```

Ces variables permettent notamment de configurer :

- l'authentification de `/predict` ;
- l'adresse de l'API de production ;
- la connexion à PostgreSQL / Supabase.

> ⚠️ Le fichier `.env` contient des secrets et ne doit jamais être versionné dans Git.

---

# 🚀 5. Lancer l'API FastAPI

Démarrage local :

```bash
uv run uvicorn app.main:app --reload
```

L'API est alors accessible sur :

```text
http://127.0.0.1:8000
```

Documentation Swagger :

```text
http://127.0.0.1:8000/docs
```

### Endpoints principaux

```text
GET  /health
GET  /model-info
POST /predict
```

### ❤️ Vérifier l'état de l'API

```bash
curl http://127.0.0.1:8000/health
```

### 🧠 Vérifier le modèle déployé

```bash
curl http://127.0.0.1:8000/model-info
```

---

# 🧪 6. Tests automatisés

Les tests sont réalisés avec **Pytest**.

Lancer tous les tests :

```bash
uv run pytest -v
```

Version compacte :

```bash
uv run pytest -q
```

La suite couvre notamment :

- ❤️ disponibilité de `/health` ;
- 🧠 informations retournées par `/model-info` ;
- 🔐 protection de `/predict` par API Key ;
- ✅ prédiction avec un payload valide ;
- ❌ feature manquante ;
- ❌ feature supplémentaire ;
- ❌ donnée non numérique ;
- ❌ valeurs infinies ;
- gestion des valeurs `None` ;
- construction du tableau NumPy ;
- validation des probabilités ;
- fonctionnement du `ModelService` ;
- chargement du modèle ;
- prédiction réelle XGBoost.

À l'état actuel du projet :

```text
30 tests
```

sont exécutés automatiquement.

---

# 🛡️ 7. Coverage et Quality Gate

La couverture du code est mesurée avec `pytest-cov`.

```bash
uv run pytest \
  --cov=app \
  --cov-report=term-missing \
  --cov-fail-under=80 \
  -v
```

Le projet impose un **quality gate de 80 %**.

```text
Push / Pull Request
        ↓
Pytest
        ↓
Coverage
        ↓
   >= 80 % ?
    /     \
  OUI     NON
   │       │
   ▼       └── ❌ Pipeline bloqué
Docker
   ↓
Render
   ↓
Smoke Test
```

Lors de la dernière validation locale :

```text
30 tests passed
Coverage global : 85.08 %
Quality Gate     : PASSED
```

### 📊 Rapport HTML

```bash
uv run pytest \
  --cov=app \
  --cov-report=term-missing \
  --cov-report=html
```

Puis :

```bash
open htmlcov/index.html
```

---

# 🐳 8. Docker

### Construire l'image

```bash
docker build -t p8-credit-scoring-api .
```

### Démarrer le conteneur

```bash
docker run --rm \
  --name p8-credit-scoring-api \
  -p 7860:7860 \
  --env-file .env \
  p8-credit-scoring-api
```

L'API est alors disponible sur :

```text
http://127.0.0.1:7860
```

Vérification :

```bash
curl http://127.0.0.1:7860/health
```

### Tester une prédiction

```bash
uv run python -m scripts.test_docker_predict
```

---

# 📊 9. Dashboard Streamlit

```bash
uv run streamlit run dashboard.py
```

Puis ouvrir :

```text
http://localhost:8501
```

Le dashboard permet de suivre plusieurs indicateurs de production :

- 📥 volume de requêtes ;
- ❌ taux d'erreur ;
- ⏱️ latence ;
- 🌐 distribution des codes HTTP ;
- 📊 distribution des scores ;
- 🟢🔴 répartition des décisions ;
- 📈 Data Drift ;
- 🎯 performances supervisées lorsque les labels réels sont disponibles.

---

# 📡 10. Génération de trafic et monitoring

## Générer du trafic opérationnel

```bash
uv run python -m scripts.generate_monitoring_logs
```

Ce script envoie volontairement des requêtes valides et invalides afin d'alimenter le monitoring opérationnel.

## Envoyer du trafic issu des données P6

```bash
uv run python -m scripts.send_p6_production_traffic
```

## Envoyer du trafic labellisé

```bash
uv run python -m scripts.send_p6_labelled_monitoring_traffic
```

---

# 🔎 11. Analyse du monitoring

Analyser les données collectées :

```bash
uv run python -m scripts.analyze_monitoring
```

Analyser les anomalies :

```bash
uv run python -m scripts.analyze_operational_anomalies
```

Générer les graphiques :

```bash
uv run python -m scripts.plot_monitoring
```

Exemples d'artefacts :

```text
reports/api_latency.png
reports/http_status_codes.png
reports/score_distribution.png
reports/operational_anomalies.json
```

---

# 📈 12. Analyse du Data Drift

```bash
uv run python -m scripts.analyze_data_drift
```

Artefacts :

```text
reports/data_drift_report.html
reports/data_drift_summary.json
reports/data_drift_features.csv
```

### 📊 Rapport Evidently

```bash
open reports/data_drift_report.html
```

### 📓 Notebook

```text
notebooks/data_drift_analysis.ipynb
```

---

# 🔬 13. Profiling avec cProfile

```bash
uv run python -m scripts.profile_model_inference
```

Rapports :

```text
reports/model_inference_profile.txt
reports/model_inference_profile_summary.json
```

---

# 💻 14. Profiling CPU et mémoire

```bash
uv run python -m scripts.profile_resource_usage
```

Rapport :

```text
reports/resource_usage_profile.json
```

---

# ⚡ 15. Benchmark XGBoost : pipeline initial vs NumPy

```bash
uv run python -m scripts.benchmark_xgboost_numpy
```

Rapport :

```text
reports/xgboost_numpy_benchmark.json
```

Comparaison :

```text
Pipeline de référence
        VS
NumPy float32
+
XGBoost Booster.inplace_predict
```

Mesures principales :

- temps moyen ;
- médiane ;
- p90 ;
- p95 ;
- p99 ;
- speedup ;
- équivalence des probabilités.

---

# 🔄 16. Export XGBoost vers ONNX

```bash
uv run python -m scripts.export_xgboost_to_onnx
```

Le modèle obtenu est enregistré dans :

```text
models/onnx/credit_scoring_model.onnx
```

---

# 🥊 17. Benchmark XGBoost vs ONNX Runtime

```bash
uv run python -m scripts.benchmark_onnx_runtime
```

Rapport :

```text
reports/onnx_runtime_benchmark.json
```

Comparaison :

```text
XGBoost Booster.inplace_predict
        VS
ONNX Runtime CPUExecutionProvider
```

---

# 🛡️ 18. Validation de non-régression ONNX

```bash
uv run python -m scripts.validate_onnx_model_metrics
```

Rapport :

```text
reports/onnx_model_validation.json
```

Métriques comparées :

- Accuracy ;
- Precision ;
- Recall ;
- F1-score ;
- TN / FP / FN / TP ;
- coût métier ;
- probabilités ;
- décisions finales.

---

# 🖥️ 19. Comparaison CPU / mémoire : XGBoost vs ONNX

```bash
uv run python -m scripts.benchmark_runtime_resources
```

Rapport :

```text
reports/runtime_resource_comparison.json
```

Mesures :

- ⏱️ latence ;
- 🧮 temps CPU ;
- 💻 utilisation CPU estimée ;
- 🧠 mémoire ;
- delta mémoire.

---

# 🔢 20. Étude de la quantification ONNX

```bash
uv run python -m scripts.analyze_onnx_quantization
```

Rapport :

```text
reports/onnx_quantization_analysis.json
```

Dans ce projet, le modèle ONNX repose principalement sur :

```text
TreeEnsembleClassifier
```

La quantification INT8 a été étudiée et documentée, mais n'a pas été retenue comme optimisation prioritaire.

---

# ✅ 21. Validation finale du modèle optimisé

```bash
uv run python -m scripts.validate_optimized_model_metrics
```

Rapport :

```text
reports/optimized_model_validation.json
```

---

# 🌐 22. Benchmark end-to-end de l'API Render

```bash
uv run python -m scripts.benchmark_api_response_time
```

Rapport :

```text
reports/api_response_time_benchmark.json
```

Mesures :

- taux de succès ;
- moyenne ;
- médiane ;
- p90 ;
- p95 ;
- p99 ;
- minimum ;
- maximum.

Cette mesure inclut le réseau, FastAPI et l'infrastructure Render.

---

# 📏 23. Baseline de performance

```bash
uv run python -m scripts.analyze_performance_baseline
```

Rapport :

```text
reports/performance_baseline.json
```

---

# 🔄 24. CI/CD GitHub Actions

Le pipeline est défini dans :

```text
.github/workflows/ci.yml
```

Déclencheurs :

```text
push → main
pull_request → main
```

Pipeline :

```text
Tests Pytest
+
Coverage >= 80 %
        ↓
Build Docker
        ↓
Deploy Render
        ↓
Smoke Test Production
        ↓
/health + /model-info
```

---

# 🗄️ 25. PostgreSQL / Supabase

Les informations stockées comprennent notamment :

```text
request_id
created_at
input_features
probability_default
prediction
prediction_label
threshold
latency_ms
status_code
error_message
model_name
model_version
actual_default
```

La persistance utilise :

```text
FastAPI BackgroundTasks
```

afin de limiter son impact sur la latence HTTP.

---

# 📂 26. Rapports générés

```text
reports/
├── api_latency.png
├── api_response_time_benchmark.json
├── data_drift_features.csv
├── data_drift_report.html
├── data_drift_summary.json
├── http_status_codes.png
├── model_inference_profile.txt
├── model_inference_profile_summary.json
├── onnx_model_validation.json
├── onnx_quantization_analysis.json
├── onnx_runtime_benchmark.json
├── operational_anomalies.json
├── optimized_model_validation.json
├── performance_baseline.json
├── resource_usage_profile.json
├── runtime_resource_comparison.json
├── score_distribution.png
└── xgboost_numpy_benchmark.json
```

---

# 🛠️ 27. Scripts disponibles

```text
scripts/
├── analyze_data_drift.py
├── analyze_monitoring.py
├── analyze_onnx_quantization.py
├── analyze_operational_anomalies.py
├── analyze_performance_baseline.py
├── benchmark_api_response_time.py
├── benchmark_onnx_runtime.py
├── benchmark_runtime_resources.py
├── benchmark_xgboost_numpy.py
├── build_p6_labelled_monitoring_dataset.py
├── build_p6_monitoring_datasets.py
├── export_xgboost_to_onnx.py
├── generate_monitoring_logs.py
├── plot_monitoring.py
├── profile_model_inference.py
├── profile_resource_usage.py
├── send_p6_labelled_monitoring_traffic.py
├── send_p6_production_traffic.py
├── test_docker_predict.py
├── test_model_loading.py
├── test_predict_endpoint.py
├── test_render_predict.py
├── validate_onnx_model_metrics.py
└── validate_optimized_model_metrics.py
```

---

# 📁 28. Arborescence du projet

```text
P8_MLOps_credit_scoring_api_monitoring/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app/
│   ├── core/
│   ├── monitoring/
│   ├── schemas/
│   ├── security/
│   ├── services/
│   └── main.py
│
├── data/
│   ├── monitoring/
│   └── reference/
│
├── docs/
│   └── performance_optimization_report.md
│
├── models/
│   ├── credit_scoring_model/
│   ├── onnx/
│   └── feature_names.json
│
├── notebooks/
│   └── data_drift_analysis.ipynb
│
├── reports/
├── scripts/
├── tests/
│   ├── test_api.py
│   └── test_model_service.py
│
├── dashboard.py
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# 📚 29. Rapport d'optimisation

Le rapport détaillé est disponible ici :

```text
docs/performance_optimization_report.md
```

Il couvre notamment :

- 🔬 profiling `cProfile` ;
- 🔎 identification des goulots d'étranglement ;
- ⚡ optimisation NumPy ;
- 🌐 benchmark de l'API ;
- 🥊 comparaison XGBoost / ONNX ;
- 💻 mesures CPU / mémoire ;
- 🛡️ tests de non-régression ;
- 🔢 étude de la quantification ;
- 🚀 justification du runtime final.

---

# 🏆 30. Pourquoi XGBoost + NumPy a finalement été retenu

Plusieurs stratégies ont été évaluées au lieu de supposer qu'une technologie serait automatiquement meilleure.

```text
                 Modèle XGBoost
                       │
             ┌─────────┴──────────┐
             ▼                    ▼
       Runtime natif         Export ONNX
             │                    │
             ▼                    ▼
   NumPy float32 +          ONNX Runtime
   inplace_predict          CPU Provider
             │                    │
             └─────────┬──────────┘
                       ▼
                   Benchmarks
                       │
                       ▼
              Tests non-régression
                       │
                       ▼
              CPU / RAM / Latence
                       │
                       ▼
                Choix final
```

Le runtime retenu est :

```text
XGBoost Booster.inplace_predict
+
NumPy float32
+
CPU
```

---

# 🧰 31. Configuration technique finale

| Composant | Technologie retenue |
|---|---|
| 🧠 Modèle | XGBoost |
| 📦 Versioning ML | MLflow |
| ⚡ Runtime | `Booster.inplace_predict` |
| 🔢 Entrée modèle | NumPy `float32` |
| 🌐 API | FastAPI |
| 🔐 Sécurité | API Key |
| 🐳 Conteneurisation | Docker |
| 🧪 Tests | Pytest |
| 🛡️ Quality Gate | Coverage ≥ 80 % |
| 🔄 CI/CD | GitHub Actions |
| 🚀 Production | Render |
| 🗄️ Stockage | PostgreSQL / Supabase |
| 📊 Dashboard | Streamlit |
| 📈 Data Drift | Evidently |
| 🔬 Profiling | cProfile + mesures ressources |
| ⚙️ Hardware | CPU |

---

# ⚡ 32. Commandes utiles — Cheat Sheet

### API

```bash
uv run uvicorn app.main:app --reload
```

### Tests

```bash
uv run pytest -v
```

### Coverage

```bash
uv run pytest \
  --cov=app \
  --cov-report=term-missing \
  --cov-fail-under=80 \
  -v
```

### Streamlit

```bash
uv run streamlit run dashboard.py
```

### Data Drift

```bash
uv run python -m scripts.analyze_data_drift
```

### Profiling

```bash
uv run python -m scripts.profile_model_inference
```

### Benchmark XGBoost / NumPy

```bash
uv run python -m scripts.benchmark_xgboost_numpy
```

### Benchmark XGBoost / ONNX

```bash
uv run python -m scripts.benchmark_onnx_runtime
```

### CPU / RAM

```bash
uv run python -m scripts.benchmark_runtime_resources
```

### Validation ONNX

```bash
uv run python -m scripts.validate_onnx_model_metrics
```

### Quantification

```bash
uv run python -m scripts.analyze_onnx_quantization
```

### Benchmark API de production

```bash
uv run python -m scripts.benchmark_api_response_time
```

### Monitoring

```bash
uv run python -m scripts.analyze_monitoring
```

---

# 🎓 33. Synthèse

Ce projet illustre le passage d'un modèle de Machine Learning expérimental à une **solution MLOps complète et exploitable en production**.

La démarche couvre quatre dimensions complémentaires :

```text
                 MLOps P8
                    │
     ┌──────────────┼───────────────┐
     ▼              ▼               ▼
 Déploiement     Monitoring     Optimisation
     │              │               │
 FastAPI        PostgreSQL        Profiling
 Docker         Streamlit         NumPy
 Render         Data Drift        ONNX
     │              │               │
     └──────────────┼───────────────┘
                    ▼
                 Qualité
                    │
              Pytest + CI/CD
              Coverage >= 80 %
```

Le projet démontre notamment que la mise en production d'un modèle ne s'arrête pas à la création d'une API.

Il faut également être capable de :

- 🚀 **déployer** le modèle de manière reproductible ;
- 🧪 **tester** automatiquement son comportement ;
- 🔐 **sécuriser** son accès ;
- 📡 **observer** son utilisation en production ;
- 📈 **détecter** une évolution des données ;
- ⚡ **mesurer puis optimiser** les performances ;
- 🛡️ **vérifier l'absence de régression** après optimisation ;
- 🔄 **automatiser** les contrôles et le déploiement.

La configuration finale privilégie une architecture simple, mesurable et reproductible :

```text
FastAPI
+
NumPy float32
+
XGBoost Booster.inplace_predict
+
PostgreSQL / Supabase
+
Streamlit
+
Docker
+
GitHub Actions
+
Render
```
