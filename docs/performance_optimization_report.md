# Rapport d'analyse et d'optimisation des performances

## Projet P8 — MLOps Credit Scoring API

## 1. Objectif

Cette phase du projet a pour objectif d'analyser les performances du modèle
de scoring crédit après son déploiement, d'identifier les principaux goulots
d'étranglement et d'évaluer plusieurs stratégies d'optimisation.

Les optimisations ont été évaluées selon plusieurs dimensions :

- temps d'inférence du modèle ;
- temps de réponse HTTP de l'API ;
- utilisation CPU ;
- consommation mémoire ;
- compatibilité avec l'environnement de production ;
- conservation des prédictions et des performances métier.

Les mesures ont été réalisées à la fois localement et sur l'API FastAPI
déployée sur Render.

Le modèle métier utilisé reste le modèle XGBoost développé lors du projet P6.

Le seuil de décision métier est fixé à :

`0.45`

---

## 2. Architecture avant optimisation

L'API expose un endpoint :

`POST /predict`

recevant les 656 features utilisées par le modèle de scoring.

Le pipeline initial réalisait plusieurs opérations de transformation avant
l'inférence, notamment la reconstruction des données dans des structures
Pandas avant leur transmission au modèle XGBoost.

Le système de monitoring en production permet également de mesurer :

- la latence d'inférence ;
- le temps total du handler ;
- le temps de persistance du monitoring ;
- le statut HTTP ;
- les informations du modèle ;
- les prédictions produites.

Les données de monitoring sont persistées dans PostgreSQL/Supabase.

---

## 3. Benchmark initial de l'API en production

Un premier benchmark end-to-end a été réalisé depuis un client local vers
l'API déployée sur Render.

### Configuration

- 100 requêtes de prédiction ;
- 5 requêtes de warm-up ;
- 656 features par observation ;
- connexion HTTP persistante via `httpx.Client`.

### Résultats initiaux

| Métrique | Valeur |
|---|---:|
| Requêtes réussies | 100 / 100 |
| Taux de succès | 100 % |
| Temps moyen | 852.913 ms |
| Médiane | 857.984 ms |
| p90 | 896.439 ms |
| p95 | 903.273 ms |
| p99 | 929.204 ms |
| Minimum | 810.933 ms |
| Maximum | 930.681 ms |

Ces résultats ont montré que le temps de réponse observé par le client était
très supérieur au temps d'inférence pur du modèle.

Il était donc nécessaire de distinguer plusieurs composantes :

1. préparation des données ;
2. inférence du modèle ;
3. traitement FastAPI ;
4. persistance du monitoring ;
5. latence réseau et infrastructure.

---

## 4. Profiling du pipeline d'inférence

Le pipeline a été analysé avec des outils de profiling Python, notamment
`cProfile`.

L'objectif était d'identifier les opérations les plus coûteuses exécutées
pour chaque prédiction.

L'analyse a montré qu'une partie significative du coût ne provenait pas
directement de l'algorithme XGBoost, mais des opérations Python/Pandas
nécessaires à la préparation de l'observation avant l'inférence.

Cela a conduit à tester une première optimisation consistant à supprimer
Pandas du chemin critique d'inférence.

---

## 5. Optimisation du pipeline XGBoost avec NumPy

### 5.1 Pipeline initial

Le pipeline de référence effectuait des transformations de données avant
l'appel au modèle.

### 5.2 Pipeline optimisé

Le pipeline a été remplacé par :

`dict -> NumPy float32 -> XGBoost Booster.inplace_predict`

Les 656 features sont reconstruites directement dans leur ordre attendu dans
un tableau NumPy de type `float32`.

Les valeurs manquantes sont conservées sous forme de `np.nan`.

L'utilisation de :

`Booster.inplace_predict`

permet d'éviter plusieurs transformations intermédiaires.

### 5.3 Benchmark

Le benchmark a été effectué sur 500 observations.

| Métrique | Pipeline de référence | Pipeline NumPy |
|---|---:|---:|
| Temps moyen | 1.2408 ms | 0.2740 ms |
| Médiane | 1.2270 ms | 0.2200 ms |
| p90 | 1.3121 ms | 0.2829 ms |
| p95 | 1.3982 ms | 0.3118 ms |
| p99 | 1.4610 ms | 0.3432 ms |

Le pipeline NumPy atteint :

- **speedup : x4.53**
- **amélioration moyenne : 77.92 %**

### 5.4 Validation fonctionnelle

Les prédictions des deux pipelines ont été comparées.

Résultats :

- prédictions équivalentes : `True`
- différence maximale : `0.0`
- différence moyenne : `0.0`

L'optimisation NumPy n'introduit donc aucune modification numérique des
prédictions sur le benchmark réalisé.

---

## 6. Expérimentation ONNX Runtime

Une deuxième stratégie d'optimisation a consisté à convertir le modèle
XGBoost au format ONNX.

Le modèle a été exporté puis exécuté avec :

`ONNX Runtime`

et :

`CPUExecutionProvider`

### Configuration testée

- mode : `ORT_SEQUENTIAL`
- threads intra-op : `1`
- threads inter-op : `1`
- CPU uniquement ;
- 656 features `float32`.

Cette configuration permet de tester ONNX dans des conditions adaptées à un
service disposant de ressources CPU limitées.

---

## 7. Benchmark XGBoost vs ONNX Runtime

Le benchmark comparatif a été effectué sur 500 observations.

### XGBoost

| Métrique | Valeur |
|---|---:|
| Moyenne | 0.2253 ms |
| Médiane | 0.2179 ms |
| p90 | 0.2806 ms |
| p95 | 0.3052 ms |
| p99 | 0.3425 ms |

### ONNX Runtime

| Métrique | Valeur |
|---|---:|
| Moyenne | 0.0167 ms |
| Médiane | 0.0163 ms |
| p90 | 0.0177 ms |
| p95 | 0.0185 ms |
| p99 | 0.0208 ms |

### Résultat

ONNX Runtime atteint :

- **speedup : x13.49**
- **amélioration moyenne : 92.59 %**

ONNX Runtime est donc significativement plus rapide que XGBoost pour
l'inférence pure dans ce benchmark local.

---

## 8. Validation de non-régression ONNX

L'optimisation du runtime ne doit pas modifier les décisions du modèle.

Une validation spécifique a donc été réalisée sur 1 000 observations
labellisées.

### XGBoost

| Métrique | Valeur |
|---|---:|
| Accuracy | 0.710 |
| Precision | 0.191740 |
| Recall | 0.802469 |
| F1-score | 0.309524 |
| TN | 645 |
| FP | 274 |
| FN | 16 |
| TP | 65 |
| Coût métier | 434 |

### ONNX Runtime

Les mêmes résultats ont été obtenus :

| Métrique | Valeur |
|---|---:|
| Accuracy | 0.710 |
| Precision | 0.191740 |
| Recall | 0.802469 |
| F1-score | 0.309524 |
| TN | 645 |
| FP | 274 |
| FN | 16 |
| TP | 65 |
| Coût métier | 434 |

La comparaison a également montré :

- probabilités équivalentes : `True`
- prédictions identiques : `True`
- métriques identiques : `True`
- différence maximale : `3.8743e-07`
- différence moyenne : `5.7636e-08`
- régression détectée : `False`

La conversion ONNX n'a donc introduit aucune régression métier mesurable sur
cet échantillon.

---

## 9. Analyse CPU et mémoire

Une comparaison des ressources utilisées par les deux runtimes a également
été réalisée sur 1 000 observations.

### XGBoost

- durée totale : `0.2424 s`
- latence moyenne : `0.2424 ms`
- temps CPU : `1.011088 s`
- estimation d'utilisation CPU : `417.12 %`
- mémoire avant : `283.95 MB`
- mémoire après : `284.03 MB`
- variation mémoire : `0.08 MB`

### ONNX Runtime

- durée totale : `0.0088 s`
- latence moyenne : `0.0088 ms`
- temps CPU : `0.008839 s`
- estimation d'utilisation CPU : `100.44 %`
- mémoire avant : `284.03 MB`
- mémoire après : `284.05 MB`
- variation mémoire : `0.02 MB`

Aucun GPU n'a été utilisé.

L'utilisation CPU supérieure à 100 % observée avec XGBoost correspond à une
utilisation de plusieurs threads CPU pendant l'inférence.

ONNX Runtime mono-thread présente une consommation CPU plus prévisible et une
latence d'inférence très faible.

---

## 10. Optimisation du monitoring

L'analyse de production a montré qu'un autre goulot d'étranglement important
était la persistance synchrone des données de monitoring dans
PostgreSQL/Supabase.

Cette écriture pouvait prendre plusieurs centaines de millisecondes alors que
l'inférence elle-même ne nécessitait que quelques millisecondes ou moins.

Le stockage du monitoring a donc été déplacé hors du chemin critique de la
réponse HTTP avec :

`FastAPI BackgroundTasks`

Le fonctionnement devient :

`requête -> inférence -> réponse HTTP`

puis :

`persistance du monitoring en arrière-plan`

Le temps de stockage continue d'être mesuré et journalisé afin de conserver
l'observabilité du système.

Cette optimisation permet de réduire la dépendance du temps de réponse client
à la latence du stockage PostgreSQL/Supabase.

---

## 11. Analyse de l'expérimentation ONNX en production

ONNX Runtime a démontré un gain très important sur l'inférence pure.

Cependant, les benchmarks end-to-end réalisés sur l'API déployée ont montré
que le temps d'inférence du modèle ne constituait plus le facteur dominant du
temps de réponse global.

Un benchmark réalisé pendant cette phase expérimentale a notamment produit
un temps de réponse moyen d'environ :

`308 ms`

malgré une inférence ONNX extrêmement rapide.

Ce résultat montre qu'une optimisation du moteur d'inférence isolé ne garantit
pas automatiquement une amélioration équivalente de la latence HTTP globale.

Le réseau, l'infrastructure Render, FastAPI et les opérations périphériques
restent prépondérants une fois que l'inférence est inférieure à la
milliseconde.

---

## 12. Configuration finale retenue

La configuration finale retenue pour la production est :

### Modèle

- famille : XGBoost ;
- modèle métier P6 conservé ;
- seuil métier : `0.45`.

### Pipeline d'inférence

`JSON -> NumPy float32 -> XGBoost Booster.inplace_predict`

### Monitoring

- logs JSON structurés ;
- PostgreSQL/Supabase ;
- persistance via `FastAPI BackgroundTasks`.

### Infrastructure

- API FastAPI ;
- déploiement Render ;
- CPU ;
- aucun GPU nécessaire.

### Justification

ONNX Runtime a démontré qu'il était techniquement possible d'accélérer
fortement l'inférence pure.

Cependant, le pipeline NumPy/XGBoost est conservé comme configuration finale
car :

1. il améliore déjà fortement l'inférence par rapport au pipeline initial ;
2. il conserve exactement les prédictions du modèle de référence ;
3. il réduit la complexité du runtime de production ;
4. il reste directement compatible avec l'artefact XGBoost/MLflow existant ;
5. le benchmark end-to-end montre que l'inférence n'est plus le principal
   facteur limitant ;
6. ONNX n'a pas démontré de gain end-to-end suffisamment déterminant dans
   l'environnement Render testé.

Cette décision distingue donc l'optimisation théorique de l'inférence de
l'optimisation réelle du service en production.

---

## 13. Benchmark final en production

Après intégration du pipeline final et déploiement via CI/CD, un nouveau
benchmark de 100 requêtes a été effectué sur l'API Render.

### Résultats

| Métrique | Valeur |
|---|---:|
| Requêtes | 100 |
| Succès | 100 |
| Échecs | 0 |
| Taux de succès | 100 % |
| Temps moyen | 120.320 ms |
| Médiane | 116.056 ms |
| p90 | 122.581 ms |
| p95 | 142.757 ms |
| p99 | 212.834 ms |
| Minimum | 108.096 ms |
| Maximum | 239.460 ms |

Le service présente donc une latence stable sur la majorité des requêtes,
avec une médiane proche de 116 ms.

---

## 14. Comparaison avant / après optimisation

Le premier benchmark de production avait mesuré :

`852.913 ms`

de temps de réponse moyen.

La configuration finale atteint :

`120.320 ms`

Le gain end-to-end est donc d'environ :

- **732.593 ms économisées par requête en moyenne**
- **85.89 % de réduction du temps de réponse moyen**
- **speedup d'environ x7.09**

En parallèle, l'optimisation spécifique du pipeline XGBoost apporte :

- **77.92 % de réduction du temps d'inférence**
- **speedup x4.53**

sans modification des prédictions.

---

## 15. Synthèse des optimisations

| Optimisation | Résultat | Décision |
|---|---|---|
| Profiling cProfile | Identification du coût de préparation des données | Conservé |
| Suppression de Pandas du chemin critique | Réduction importante de l'inférence | Conservé |
| NumPy float32 | Pipeline plus léger | Conservé |
| XGBoost `inplace_predict` | Speedup x4.53 du pipeline | Conservé |
| ONNX Runtime | Speedup x13.49 en inférence pure | Testé, non retenu en production finale |
| Validation ONNX | Aucune régression détectée | Validé |
| ONNX mono-thread | CPU plus prévisible | Testé |
| GPU | Non nécessaire | Non retenu |
| Monitoring synchrone | Goulot d'étranglement identifié | Abandonné |
| FastAPI BackgroundTasks | Stockage retiré du chemin critique | Conservé |
| Benchmark Render | 120.320 ms moyen final | Validé |

---

## 16. Conclusion

L'analyse montre que l'optimisation d'un système de Machine Learning en
production ne doit pas se limiter au temps d'exécution du modèle.

Le profiling a d'abord permis d'identifier le coût des transformations de
données et de remplacer le pipeline initial par une implémentation NumPy
`float32` utilisant directement `XGBoost Booster.inplace_predict`.

Cette optimisation réduit le temps d'inférence moyen de 77.92 % sans modifier
les prédictions.

ONNX Runtime a ensuite été évalué et a démontré un speedup de x13.49 sur
l'inférence pure, sans régression des métriques du modèle. Cette expérimentation
a cependant également démontré que l'inférence n'était plus le principal
goulot d'étranglement du service.

L'analyse du monitoring a notamment mis en évidence le coût de la persistance
PostgreSQL/Supabase. Son déplacement vers des tâches d'arrière-plan permet de
ne plus bloquer directement le chemin critique de la réponse.

Enfin, le benchmark réalisé après déploiement de la configuration finale sur
Render mesure un temps de réponse moyen de 120.320 ms contre 852.913 ms lors
du benchmark initial, soit une réduction d'environ 85.89 %.

La configuration finale privilégie ainsi un compromis entre performance,
simplicité, compatibilité, observabilité et stabilité du système de
production.

---

---

## 17. Reproductibilité des expérimentations

Les résultats présentés dans ce rapport sont issus de scripts versionnés
dans le dépôt Git du projet.

Les différents benchmarks ont été exécutés sur des données représentatives
du projet P6 et les résultats générés sont conservés sous forme de rapports
JSON dans le dossier `reports/`.

Cette organisation permet :

- de reproduire les expérimentations ;
- de comparer les différentes stratégies d'optimisation ;
- de vérifier l'absence de régression ;
- de conserver une trace des performances mesurées ;
- de justifier techniquement la configuration retenue en production.

---

### 17.1 Profiling du modèle

Script :

`scripts/profile_model_inference.py`

#### Objectif

Le profiling permet d'analyser le pipeline d'inférence afin d'identifier
les opérations responsables de la majorité du temps d'exécution.

Cette analyse a notamment permis d'orienter les optimisations vers :

- la suppression des transformations Pandas inutiles pendant l'inférence ;
- l'utilisation directe de tableaux NumPy ;
- l'utilisation du type `float32` ;
- l'utilisation de `XGBoost Booster.inplace_predict` ;
- puis l'évaluation d'ONNX Runtime comme moteur d'inférence alternatif.

#### Exécution

```bash
uv run python -m scripts.profile_model_inference
```

Le profiling constitue ainsi le point de départ des optimisations :
les modifications n'ont pas été réalisées arbitrairement, mais à partir
des goulots d'étranglement identifiés expérimentalement.

---

### 17.2 Benchmark Pandas / NumPy / XGBoost

Script :

`scripts/benchmark_xgboost_numpy.py`

Rapport :

`reports/xgboost_numpy_benchmark.json`

#### Objectif

Cette expérimentation mesure l'impact de l'optimisation du pipeline
de préparation des données avant l'inférence XGBoost.

Le pipeline optimisé utilise :

```text
features
    ↓
NumPy float32
    ↓
XGBoost Booster.inplace_predict
    ↓
probabilité de défaut
```

Cette approche évite la création et les transformations répétées
de structures Pandas pour chaque prédiction.

#### Exécution

```bash
uv run python -m scripts.benchmark_xgboost_numpy
```

#### Résultats

Sur 500 observations et 656 features :

| Métrique | Pipeline initial | Pipeline NumPy |
|---|---:|---:|
| Temps moyen | 1.2408 ms | 0.2740 ms |
| Médiane | 1.2270 ms | 0.2200 ms |
| p90 | 1.3121 ms | 0.2829 ms |
| p95 | 1.3982 ms | 0.3118 ms |
| p99 | 1.4610 ms | 0.3432 ms |

Résultat global :

- speedup : **x4.53** ;
- amélioration moyenne : **77.92 %** ;
- prédictions équivalentes : **True** ;
- différence maximale : **0.0** ;
- différence moyenne : **0.0**.

Cette première optimisation démontre qu'une partie importante du coût
initial provenait du pipeline de préparation des données et non
exclusivement du modèle lui-même.

---

### 17.3 Export XGBoost vers ONNX

Script :

`scripts/export_xgboost_to_onnx.py`

Artefact généré :

`models/onnx/credit_scoring_model.onnx`

#### Objectif

La seconde stratégie d'optimisation consiste à convertir le modèle
XGBoost dans le format ONNX afin de pouvoir utiliser ONNX Runtime
comme moteur d'inférence.

L'objectif est de conserver le comportement prédictif du modèle
tout en réduisant le coût d'exécution en production.

#### Exécution

```bash
uv run python -m scripts.export_xgboost_to_onnx
```

L'export a nécessité une normalisation temporaire des noms de features
XGBoost afin de respecter le format attendu par le convertisseur ONNX.

Les noms originaux des 656 features sont conservés côté application
afin de préserver le contrat d'entrée de l'API.

Après conversion :

- le modèle ONNX est valide ;
- il contient les 656 features attendues ;
- sa taille est d'environ **0.37 MB** ;
- il peut être chargé avec ONNX Runtime.

---

### 17.4 Benchmark XGBoost / ONNX Runtime

Script :

`scripts/benchmark_onnx_runtime.py`

Rapport :

`reports/onnx_runtime_benchmark.json`

#### Objectif

Après l'optimisation du pipeline XGBoost avec NumPy `float32` et
`Booster.inplace_predict`, une seconde stratégie d'optimisation a été
expérimentée : l'utilisation d'ONNX Runtime.

L'objectif de cette expérimentation était de déterminer s'il était
possible de réduire davantage le temps d'inférence pure, sans modifier
les probabilités ni les décisions produites par le modèle de référence.

ONNX Runtime a donc été évalué comme **runtime alternatif expérimental**.
Cette expérimentation permet de mesurer son potentiel d'optimisation,
mais ne signifie pas qu'ONNX a été retenu comme runtime final de
production.

La configuration finale de production reste :

```text
JSON features
    ↓
NumPy float32
    ↓
XGBoost Booster.inplace_predict
    ↓
Probabilité de défaut
```

#### Configuration ONNX expérimentale

Afin d'obtenir une comparaison reproductible et cohérente avec un
environnement CPU, ONNX Runtime a été configuré de la manière suivante :

```text
Provider         : CPUExecutionProvider
Execution mode   : ORT_SEQUENTIAL
Threads intra-op : 1
Threads inter-op : 1
```

#### Exécution

```bash
uv run python -m scripts.benchmark_onnx_runtime
```

#### Résultats

Benchmark réalisé sur :

- 500 observations ;
- 656 features ;
- 20 observations de warm-up.

##### XGBoost natif

| Métrique | Temps |
|---|---:|
| Moyenne | 0.2253 ms |
| Médiane | 0.2179 ms |
| p90 | 0.2806 ms |
| p95 | 0.3052 ms |
| p99 | 0.3425 ms |
| Minimum | 0.1531 ms |
| Maximum | 0.5862 ms |

##### ONNX Runtime mono-thread

| Métrique | Temps |
|---|---:|
| Moyenne | 0.0167 ms |
| Médiane | 0.0163 ms |
| p90 | 0.0177 ms |
| p95 | 0.0185 ms |
| p99 | 0.0208 ms |
| Minimum | 0.0149 ms |
| Maximum | 0.0714 ms |

Résultat global :

- speedup ONNX vs XGBoost : **x13.49** ;
- amélioration moyenne : **92.59 %** ;
- probabilités équivalentes : **True** ;
- prédictions identiques : **True** ;
- prédictions différentes : **0** ;
- différence maximale : **3.2782554626464844e-07** ;
- différence moyenne : **4.2557716369628904e-08**.

Cette expérimentation démontre qu'ONNX Runtime constitue une piste
d'optimisation très performante pour l'inférence pure.

Cependant, l'optimisation d'un système de production doit être évaluée
sur l'ensemble du chemin de requête et pas uniquement sur quelques
centièmes de milliseconde d'inférence.

La configuration finale retenue en production reste donc le pipeline
NumPy `float32` + XGBoost `Booster.inplace_predict`, tandis qu'ONNX
Runtime est conservé comme expérimentation documentée et validée.

### 17.5 Validation métier ONNX

Script :

`scripts/validate_onnx_model_metrics.py`

Rapport :

`reports/onnx_model_validation.json`

#### Objectif

Une optimisation de performance ne peut être retenue si elle modifie
les performances prédictives ou le comportement métier du modèle.

Le modèle ONNX a donc été comparé au modèle XGBoost sur un ensemble
de **1 000 observations labellisées**.

#### Exécution

```bash
uv run python -m scripts.validate_onnx_model_metrics
```

#### Résultats

Les deux runtimes obtiennent exactement les mêmes métriques :

| Métrique | XGBoost | ONNX Runtime |
|---|---:|---:|
| Accuracy | 0.7100 | 0.7100 |
| Precision | 0.191740 | 0.191740 |
| Recall | 0.802469 | 0.802469 |
| F1-score | 0.309524 | 0.309524 |
| TN | 645 | 645 |
| FP | 274 | 274 |
| FN | 16 | 16 |
| TP | 65 | 65 |
| Coût métier | 434 | 434 |

Comparaison numérique :

- probabilités équivalentes : **True** ;
- prédictions identiques : **True** ;
- métriques identiques : **True** ;
- différence maximale : **3.8743019104003906e-07** ;
- différence moyenne : **5.7635828852653506e-08** ;
- régression détectée : **False**.

L'optimisation ONNX n'introduit donc aucune régression mesurable
sur cet échantillon de validation.

---

### 17.6 Analyse CPU et mémoire

Script :

`scripts/benchmark_runtime_resources.py`

Rapport :

`reports/runtime_resource_comparison.json`

#### Objectif

La performance d'un moteur d'inférence ne doit pas être évaluée
uniquement à partir de sa latence.

Une expérimentation complémentaire a donc comparé la consommation
de ressources de :

- XGBoost avec `Booster.inplace_predict` ;
- ONNX Runtime avec `CPUExecutionProvider`.

Cette comparaison permet d'évaluer ONNX comme stratégie
d'optimisation potentielle et d'étudier l'intérêt éventuel
d'une modification de la configuration hardware.

#### Exécution

```bash
uv run python -m scripts.benchmark_runtime_resources
```

#### Résultats

Sur 1 000 observations :

| Métrique | XGBoost | ONNX Runtime |
|---|---:|---:|
| Temps total | 0.2424 s | 0.0088 s |
| Latence moyenne | 0.2424 ms | 0.0088 ms |
| Temps CPU | 1.011088 s | 0.008839 s |
| CPU estimé | 417.12 % | 100.44 % |
| Mémoire avant | 283.95 MB | 284.03 MB |
| Mémoire après | 284.03 MB | 284.05 MB |
| Variation mémoire | +0.08 MB | +0.02 MB |

GPU utilisé :

**False**

Les résultats montrent qu'ONNX Runtime présente un excellent potentiel
en termes de temps d'exécution et de consommation de ressources pour
l'inférence pure.

Ils montrent également qu'un GPU n'est pas nécessaire pour ce modèle
de scoring tabulaire de petite taille.

Le déploiement CPU est donc conservé. L'ajout d'une infrastructure GPU
augmenterait le coût et la complexité du déploiement sans bénéfice
démontré pour ce cas d'usage.

La configuration finale de production utilise ainsi :

- une infrastructure CPU ;
- des entrées NumPy `float32` ;
- XGBoost `Booster.inplace_predict`.

ONNX Runtime reste une stratégie d'optimisation expérimentale validée
et documentée, mais n'est pas utilisé comme runtime final de l'API.

### 17.7 Benchmark end-to-end de l'API déployée

Script :

`scripts/benchmark_api_response_time.py`

Rapport :

`reports/api_response_time_benchmark.json`

#### Objectif

Les benchmarks locaux du moteur d'inférence ne représentent qu'une
partie des performances réellement perçues par un client.

Un benchmark end-to-end a donc été exécuté depuis un client local
vers l'API FastAPI réellement déployée sur Render.

Contrairement au benchmark ONNX expérimental, cette mesure évalue
la **configuration finale réellement utilisée en production**.

Le pipeline de production retenu est :

```text
Requête HTTP
    ↓
FastAPI
    ↓
Validation des 656 features
    ↓
NumPy float32
    ↓
XGBoost Booster.inplace_predict
    ↓
Application du seuil métier
    ↓
Sérialisation de la réponse
    ↓
Réponse HTTP
```

La persistance du monitoring PostgreSQL/Supabase est exécutée
en arrière-plan afin de ne pas maintenir cette opération dans
le chemin critique de la réponse HTTP.

#### Le benchmark end-to-end inclut

- le transport réseau entre le client et Render ;
- FastAPI ;
- l'authentification par clé API ;
- la validation du payload ;
- la reconstruction des 656 features ;
- XGBoost `Booster.inplace_predict` avec entrée NumPy `float32` ;
- l'application du seuil métier ;
- la sérialisation de la réponse ;
- l'infrastructure Render.

#### Exécution

```bash
uv run python -m scripts.benchmark_api_response_time
```

Configuration du benchmark :

- 5 appels de warm-up ;
- 100 appels mesurés ;
- 656 features par observation ;
- données issues du dataset de production simulée P6.

#### Résultats finaux

| Métrique | Résultat |
|---|---:|
| Appels | 100 |
| Succès | 100 |
| Échecs | 0 |
| Taux de succès | 100.00 % |
| Temps moyen | 120.320 ms |
| Médiane | 116.056 ms |
| p90 | 122.581 ms |
| p95 | 142.757 ms |
| p99 | 212.834 ms |
| Minimum | 108.096 ms |
| Maximum | 239.460 ms |

Le benchmark démontre que l'API optimisée reste stable sur les
100 requêtes exécutées, avec un taux de succès de **100 %**.

Le benchmark final utilise le pipeline NumPy `float32` +
XGBoost `Booster.inplace_predict` retenu pour la production.

Les expérimentations ONNX ont montré qu'il était possible de réduire
davantage le temps d'inférence pure. Cependant, les mesures end-to-end
montrent que le moteur d'inférence n'est plus le principal facteur
limitant une fois le pipeline XGBoost optimisé.

Avec un temps de réponse moyen de **120.320 ms** sur Render, la latence
restante provient principalement des couches applicatives, du réseau
et de l'infrastructure d'hébergement plutôt que du calcul du modèle.

Cette observation est importante : une optimisation supplémentaire
de quelques dixièmes ou centièmes de milliseconde sur le moteur
d'inférence aurait désormais un impact limité sur la latence totale
perçue par le client.

La stratégie finale privilégie donc une configuration simple,
compatible avec l'environnement de production et déjà fortement
optimisée :

```text
CPU
+
NumPy float32
+
XGBoost Booster.inplace_predict
+
FastAPI
+
monitoring asynchrone / hors chemin critique
```

ONNX Runtime reste documenté comme une optimisation expérimentale
performante et comme une piste possible d'évolution future si les
contraintes de charge ou d'infrastructure venaient à évoluer.

---

### 17.8 Validation automatisée

Les optimisations ont été intégrées sans supprimer les contrôles
fonctionnels existants.

Avant déploiement, la suite de tests peut être exécutée avec :

```bash
uv run pytest -q
```

Dernier résultat observé :

```text
........
8 passed
```

La validité syntaxique des modules de l'application peut également
être contrôlée avec :

```bash
uv run python -m compileall app
```

Les modules suivants sont notamment vérifiés :

```text
app/
├── api/
├── core/
├── models/
├── monitoring/
├── schemas/
├── security/
└── services/
```

Ces validations sont complémentaires aux benchmarks de performance.

Elles permettent de vérifier que le passage :

```text
Pipeline initial
      ↓
Optimisation NumPy / XGBoost
      ↓
Export ONNX
      ↓
ONNX Runtime CPU
      ↓
Intégration FastAPI
      ↓
Déploiement CI/CD
```

n'introduit pas de rupture fonctionnelle dans l'application.

La validation de l'optimisation repose ainsi sur trois niveaux
complémentaires :

1. **validation fonctionnelle** : tests automatisés de l'API ;
2. **validation prédictive** : comparaison des probabilités,
   prédictions, métriques et du coût métier ;
3. **validation opérationnelle** : benchmarks de latence,
   CPU, mémoire et temps de réponse end-to-end.

La version optimisée peut donc être retenue pour la production :
elle améliore fortement les performances d'inférence, conserve
les résultats du modèle de référence et reste compatible avec
l'environnement CPU utilisé pour le déploiement.

### 17.9 Analyse de la pertinence de la quantification

Script :

`scripts/analyze_onnx_quantization.py`

Rapport :

`reports/onnx_quantization_analysis.json`

#### Objectif

Une stratégie supplémentaire d'optimisation étudiée concerne
la quantification du modèle.

La quantification INT8 est fréquemment utilisée pour réduire :

- la taille des modèles ;
- leur consommation mémoire ;
- leur temps d'inférence ;
- la charge CPU.

Cependant, son intérêt dépend fortement de la structure du graphe
du modèle.

Avant d'appliquer une transformation de quantification, le graphe
ONNX du modèle XGBoost a donc été inspecté afin de déterminer
si celui-ci contient des opérateurs réellement compatibles avec
les mécanismes classiques de quantification ONNX Runtime.

#### Exécution

```bash
uv run python -m scripts.analyze_onnx_quantization
```

#### Structure du modèle analysé

Le modèle ONNX possède une taille de :

`0.371 MB`

Le graphe ne contient qu'un seul noeud :

```text
TreeEnsembleClassifier : 1
```

Aucun opérateur matriciel classiquement ciblé par les outils
de quantification ONNX Runtime n'a été détecté.

En particulier, le graphe ne contient pas :

```text
MatMul
Gemm
Conv
Attention
LSTM
GRU
```

Le modèle ne contient également aucun initializer :

```text
Initializers : 0
```

Cela signifie que le modèle ne stocke pas ses paramètres sous la
forme de matrices de poids classiques qui pourraient être converties
de `float32` vers `int8` avec les mécanismes standards de
quantification.

#### Diagnostic

Résultats de l'analyse :

| Élément | Résultat |
|---|---|
| Modèle basé sur des arbres | Oui |
| `TreeEnsembleClassifier` détecté | Oui |
| Opérateurs matriciels quantifiables | Aucun |
| Initializers | 0 |
| Candidat à une quantification INT8 standard | Non |

Le modèle XGBoost exporté vers ONNX est représenté par un opérateur
`TreeEnsembleClassifier`.

La quantification INT8 classique proposée par ONNX Runtime cible
principalement des opérations matricielles comme `MatMul`, `Gemm`
ou `Conv`, fréquemment utilisées dans les réseaux de neurones.

Ces opérateurs étant absents du graphe, une quantification INT8
standard n'est pas une stratégie pertinente pour ce modèle.

#### Décision

La quantification n'est donc pas appliquée à la version de production.

Cette décision ne repose pas sur une hypothèse mais sur l'analyse
de la structure réelle du graphe ONNX.

Les stratégies d'optimisation retenues ou évaluées sont finalement :

| Stratégie | Résultat |
|---|---|
| Suppression de Pandas du chemin critique | Retenue |
| NumPy `float32` | Retenue |
| XGBoost `Booster.inplace_predict` | Retenue |
| Monitoring avec `BackgroundTasks` | Retenue |
| ONNX Runtime | Évalué, non retenu en production |
| ONNX mono-thread | Évalué |
| Quantification INT8 | Analysée, non applicable au graphe |
| GPU | Non nécessaire |
| CPU | Retenu |

Cette analyse permet de justifier la configuration finale sans
ajouter une transformation qui augmenterait la complexité du système
sans bénéfice technique démontré.