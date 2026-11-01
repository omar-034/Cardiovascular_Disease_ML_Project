# 🫀 CardioScan AI — Système Expert de Dépistage et Stratification du Risque Cardiovasculaire

[![Projet d'Apprentissage](https://img.shields.io/badge/Projet-Apprentissage%20%2F%20Portfolio%20MLOps-0052CC.svg?style=for-the-badge&logo=gitbook&logoColor=white)]()
<br>

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Clinical Recall](https://img.shields.io/badge/Recall%20(Sensibilit%C3%A9)-100%25-00C853.svg?style=flat-square)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9999-00E676.svg?style=flat-square)]()
[![Accuracy](https://img.shields.io/badge/Exactitude-99.00%25-00B0FF.svg?style=flat-square)]()
[![Status](https://img.shields.io/badge/Statut-Production--Ready-6200EA.svg?style=flat-square)]()
[![License](https://img.shields.io/badge/Licence-MIT-blue.svg?style=flat-square)](LICENSE)

> 🎓 **NOTE DE CADRE — PROJET D'APPRENTISSAGE & PORTFOLIO MLOps**  
> Ce dépôt constitue un **projet d'apprentissage et de démonstration technique** développé dans un cadre académique et d'auto-formation avancée. Il illustre l'application concrète des meilleures pratiques d'ingénierie logicielle et de **MLOps appliqué à la santé** : pipeline modulaire, prévention stricte des fuites de données (*data leakage*), calibration de seuil décisionnel asymétrique, benchmarking multi-modèles et déploiement d'une application web interactive. Il a vocation pédagogique et ne constitue pas un dispositif médical homologué.

---

## 📑 Sommaire
1. [Contexte Médical & Problématique Clinique](#-1-contexte-médical--problématique-clinique)
2. [Indicateurs Clés de Performance (KPIs)](#-2-indicateurs-clés-de-performance-kpis)
3. [Benchmark Multi-Modèles & Validation Croisée](#-3-benchmark-multi-modèles--validation-croisée)
4. [Ingénierie des Caractéristiques Cliniques](#-4-ingénierie-des-caractéristiques-cliniques)
5. [Architecture MLOps & Prévention des Fuites (Data Leakage)](#-5-architecture-mlops--prévention-des-fuites-data-leakage)
6. [Application Web Clinique Streamlit](#-6-application-web-clinique-streamlit)
7. [Structure du Répertoire](#-7-structure-du-répertoire)
8. [Installation & Démarrage Rapide](#-8-installation--démarrage-rapide)
9. [Pipelines de Production (CLI)](#-9-pipelines-de-production-cli)
10. [Éthique, Cadre d'Apprentissage & Avertissement Médical](#-10-éthique-cadre-dapprentissage--avertissement-médical)
11. [Auteur & Licence](#-11-auteur--licence)

---

## 🩺 1. Contexte Médical & Problématique Clinique

Les cardiopathies ischémiques demeurent la première cause de mortalité à l'échelle mondiale. En cardiologie diagnostique, la matrice des coûts d'erreur est **hautement asymétrique** :

- **Faux Positif (FP)** : Déclenche des explorations de seconde intention (scintigraphie myocardique, coroscanner) sans engager le pronostic vital du patient.
- **Faux Négatif (FN)** : Renvoie au domicile un patient coronarien sévère sans traitement, avec un risque immédiat d'infarctus du myocarde (IDM) ou d'arrêt cardio-circulatoire.

### Le Dilemme de l'Exactitude Classique
Dans les applications cliniques, une exactitude globale (*Accuracy*) de 95% est trompeuse si elle masque 5% de faux négatifs mortels. C'est pourquoi **CardioScan AI** est optimisé selon la règle médicale cardinale :
- Maximisation absolue de la **Sensibilité / Rappel (Recall = 100%)**
- Optimisation du **F2-Score** (qui quadruple l'importance relative du Rappel par rapport à la Précision)

La formule mathématique du $F_2$-Score appliquée au modèle est définie par :

$$
F_2 = (1 + 2^2) \times \frac{\text{Precision} \times \text{Recall}}{(2^2 \times \text{Precision}) + \text{Recall}} = 5 \times \frac{\text{Precision} \times \text{Recall}}{4 \times \text{Precision} + \text{Recall}}
$$

---

## 🎯 2. Indicateurs Clés de Performance (KPIs)

Sur le jeu de test indépendant (**Holdout de 20% inviolé**, 200 patients), après calibration du seuil de décision clinique ($\theta^* = 0.10$) :

| Métrique Clinique | Résultat Modèle Champion | Interprétation Médicale |
| :--- | :---: | :--- |
| **Rappel (Recall / Sensibilité)** | **100.0%** | **Zéro faux négatif** : 100% des patients coronariens détectés. |
| **F2-Score Clinique** | **0.9966** | Équilibre optimal avec priorité absolue accordée à la détection des malades. |
| **ROC-AUC (Aire sous la courbe)** | **0.9999** | Capacité de discrimination quasi-absolue entre sains et pathologiques. |
| **Exactitude Globale (Accuracy)** | **99.00%** | Taux global de prédictions correctes sur la cohorte d'évaluation. |
| **Précision Diagnostique** | **98.31%** | Taux extrêmement faible de fausses alertes (spécificité préservée). |
| **Seuil Décisionnel Dédié (θ*)** | **0.10** | Seuil calibré par optimisation F2 (vs seuil standard par défaut à 0.50). |

---

## 🏆 3. Benchmark Multi-Modèles & Validation Croisée

Une validation croisée stratifiée à 5 plis (**Stratified 5-Fold Cross-Validation**) a été menée pour comparer 5 architectures d'apprentissage supervisé :

| Rang | Modèle Algorithmique | Recall Moyen | F2-Score Moyen | ROC-AUC Moyen | Accuracy Moyenne | Précision Moyenne |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **HistGradientBoosting** *(Champion)* | **98.49% ± 0.53%** | **0.9807 ± 0.40%** | **0.9950 ± 0.61%** | **97.00%** | **96.47%** |
| 🥈 | **Random Forest Classifier** | 98.06% ± 1.06% | 0.9755 ± 0.99% | 0.9927 ± 0.67% | 96.25% | 95.62% |
| 🥉 | **XGBoost Classifier** | 97.62% ± 1.87% | 0.9724 ± 1.58% | 0.9916 ± 1.18% | 96.12% | 95.78% |
| 4 | **Régression Logistique (L2)** | 97.41% ± 1.63% | 0.9728 ± 1.41% | 0.9918 ± 0.70% | 96.62% | 96.79% |
| 5 | **Support Vector Machine (RBF)** | 97.41% ± 1.48% | 0.9720 ± 1.22% | 0.9928 ± 0.47% | 96.38% | 96.39% |

### Pourquoi HistGradientBoosting est Champion ?
1. **Modélisation non linéaire supérieure** : Capture les interactions complexes entre le sous-décalage du segment ST (`oldpeak`) et la réserve cardiaque.
2. **Robustesse aux distributions asymétriques** : Traite naturellement les variables d'épreuve d'effort sans instabilité numérique.
3. **Sensibilité diagnostique maximale** : Présente la plus faible variance d'erreur de rappel ($\pm 0.53\%$).

---

## 🔬 4. Ingénierie des Caractéristiques Cliniques

Le transformateur personnalisé [`ClinicalFeatureEngineer`](src/components/feature_engineering.py) synthétise de nouveaux biomarqueurs validés par la littérature cardiologique :

### 1. Ratio de Réserve Chronotrope (`hr_reserve_ratio`)
Mesure la capacité du cœur à mobiliser son potentiel chronotrope à l'effort maximal (selon la formule d'Astrand : $\text{FC}_{\max} = 220 - \text{âge}$) :

$$
\text{HR Reserve Ratio} = \frac{\text{MaxHeartRate}}{220 - \text{Age}}
$$

*Implémentation Python :*
```python
hr_reserve_ratio = maxheartrate / (220 - age)
```

### 2. Indice de Charge Ischémique (`ischemia_burden_index`)
Quantifie la sévérité clinique en combinant l'altération électrique objective (dépression du segment ST en mm) et la symptomatologie subjective (angine induite par l'effort) :

$$
\text{Ischemia Burden Index} = \text{Oldpeak} \times \text{ExerciseAngina}
$$

*Implémentation Python :*
```python
ischemia_burden_index = oldpeak * exerciseangia
```

### 3. Proxy de Rigidité Artérielle & Surcharge Vasculaire (`vascular_stiffness_proxy`)
Estime l'accumulation du stress pariétal artériel combinant la charge hypertensive et le vieillissement vasculaire :

$$
\text{Vascular Stiffness Proxy} = \frac{\text{RestingBP} \times \text{Age}}{100}
$$

*Implémentation Python :*
```python
vascular_stiffness_proxy = (restingBP * age) / 100.0
```

### 4. Traitement Anti-Fuite du Cholestérol Nul
- **Observation biologique** : Un taux de `serumcholestrol == 0 mg/dL` est biologiquement aberrant pour un être humain vivant (valeur manquante masquée).
- **Correction rigoureuse** : Remplacement des zéros par `np.nan`, puis imputation par la médiane apprise **strictement sur le sous-ensemble d'apprentissage (Train)**, garantissant l'absence totale de fuite d'information (*anti-data leakage*).

---

## 🛡️ 5. Architecture MLOps & Prévention des Fuites (Data Leakage)

Le pipeline de prétraitement est scellé dans un `ColumnTransformer` hermétique :

```text
                                  [ Données Brutes ]
                                          │
                                          ▼
                            [ Split Stratifié 80 / 20 ]
                           ┌──────────────┴──────────────┐
                           ▼                             ▼
                    [ TRAIN SET ]                  [ TEST SET ]
                           │                        (Inviolé)
                           ▼                             │
            [ ClinicalFeatureEngineer ]                  │
                           │                             │
                           ▼                             │
               [ ColumnTransformer ]                     │
          ┌────────────────┼────────────────┐            │
          ▼                ▼                ▼            │
    [ Continues ]     [ Nominales ]    [ Ordinales ]     │
     Imputer(Med)     Imputer(Mode)    Imputer(Mode)     │
          +                +                +            │
     RobustScaler       OneHot        StandardScaler     │
          │                │                │            │
          └────────────────┼────────────────┘            │
                           │                             │
                           ▼ (Fit & Transform)           ▼ (Transform ONLY)
                    [ X_train_scaled ]             [ X_test_scaled ]
                           │                             │
                           ▼                             │
                  [ 5-Fold Strat CV ]                    │
                           │                             │
                           ▼                             │
             [ Calibration Seuil θ* = 0.10 ]             │
                           │                             │
                           ▼                             ▼
                     [ Modèle Champion ] ───────> [ Évaluation Finale ]
                     (HistGradient)              (Recall = 100%)
```

---

## 🖥️ 6. Application Web Clinique Streamlit

L'interface graphique interactive [`app.py`](app.py) offre une expérience utilisateur de niveau professionnel :

### Fonctionnalités Clés :
1. **Évaluation Individuelle en Temps Réel** :
   - Formulaire clinique structuré en 2 volets : *Données Démographiques & Biologiques* vs *Symptomatologie & Épreuve d'Effort*.
   - Calcul interactif immédiat des 3 biomarqueurs dérivés (`hr_reserve_ratio`, `ischemia_burden_index`, `vascular_stiffness_proxy`).
   - Jauge de risque visuelle et recommandation médicale graduée (*Très Faible*, *Vigilance*, *Élevé*, *Urgence Critique*).
2. **Chargement Rapide de Profils Types (1-Clic)** :
   - 🟢 **Profil Témoin Sain** : 25 ans, femme, biomarqueurs physiologiques optimaux.
   - 🟡 **Profil à Risque Modéré** : 55 ans, homme, anomalies légères de l'onde ST.
   - 🔴 **Profil Ischémique Critique** : 76 ans, homme, angine d'effort, dépression ST 3.2 mm, 2 vaisseaux atteints.
3. **Dépistage par Cohorte (Batch CSV)** :
   - Téléversement d'un fichier CSV multi-patients.
   - Modèle CSV d'exemple téléchargeable directement depuis l'application.
   - Inférence vectorisée, statistiques d'incidence de cohorte et export du rapport CSV enrichi.
4. **Gouvernance & Auditabilité du Modèle** :
   - Tableau comparatif complet de la validation croisée 5-Fold.
   - Justification de la calibration du seuil orientée zéro faux négatif.

---

## 📁 7. Structure du Répertoire

```text
Cardiovascular Disease/
├── artifacts/                           # Modèles, préprocesseurs et métriques sérialisés (non trackés)
│   ├── model.pkl                        # Modèle champion HistGradientBoosting sérialisé
│   ├── preprocessor.pkl                 # Pipeline scikit-learn complète scellée
│   ├── threshold.json                   # Seuil clinique calibré (θ* = 0.10)
│   ├── metrics.json                     # Métriques officielles du jeu de test
│   ├── model_comparison.csv             # Tableau benchmark de la validation croisée
│   ├── train.csv                        # Jeu d'entraînement (800 observations)
│   ├── test.csv                         # Jeu de test inviolé (200 observations)
│   └── raw.csv                          # Copie intègre du jeu de données brut
├── notebooks/
│   ├── data/
│   │   └── cardiovascular.csv           # Dataset clinique source (1000 patients)
│   ├── 01_Data_overview.ipynb           # Phase 1.1 : Structure, types et intégrité
│   ├── 02_Univariate_analysis.ipynb     # Phase 1.2 : Distributions et détection d'outliers
│   ├── 03_Bivariate_analysis.ipynb      # Phase 1.3 : Corrélations et relations avec la cible
│   ├── 04_Multivariate_analysis.ipynb   # Phase 1.4 : Clusters et analyse factorielle
│   └── 05_Model_Training.ipynb          # Phase 3 : Benchmarking et optimisation de seuil
├── src/
│   ├── __init__.py                      # Package source
│   ├── exception.py                     # Gestion unifiée des exceptions enrichies
│   ├── logger.py                        # Journalisation horodatée fichier + console
│   ├── utils.py                         # Utilitaires de sérialisation pickle
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py            # Phase 2 : Découpage stratifié 80/20 anti-fuite
│   │   ├── feature_engineering.py       # Phase 2 : Transformateur ClinicalFeatureEngineer
│   │   ├── data_transformation.py       # Phase 2 : Assemblage du ColumnTransformer
│   │   └── model_trainer.py             # Phase 3 : Benchmark 5-Fold et tuning de seuil
│   └── pipeline/
│       ├── __init__.py
│       ├── train_pipeline.py            # Phase 4 : Pipeline d'entraînement complet 1-clic
│       └── predict_pipeline.py          # Phase 4 : Moteur d'inférence (CustomData)
├── app.py                               # Phase 5 : Interface utilisateur Streamlit
├── requirements.txt                     # Dépendances minimales sécurisées (0 CVE)
├── setup.py                             # Métadonnées et packaging du projet Python
└── README.md                            # Documentation de référence du projet
```

---

## 🚀 8. Installation & Démarrage Rapide

### Prérequis
- Python 3.10, 3.11, 3.12 ou 3.13
- Git

### 1. Cloner le Dépôt
```bash
git clone https://github.com/omar-034/Cardiovascular_Disease_ML_Project.git
cd "Cardiovascular Disease"
```

### 2. Créer et Activer l'Environnement Virtuel
```powershell
# Sous Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Sous Linux / macOS (Bash)
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installer les Dépendances
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Lancer l'Application Web
```bash
streamlit run app.py
```
L'interface s'ouvre automatiquement dans votre navigateur à l'adresse : **`http://localhost:8501`**

---

## ⚙️ 9. Pipelines de Production (CLI)

### Réentraînement Complet en 1 Ligne
Pour relancer l'ingestion, le prétraitement, le benchmark des 5 algorithmes et la calibration du seuil :
```bash
python -m src.pipeline.train_pipeline
```

### Inférence Unitaire en Ligne de Commande
Pour tester l'évaluation clinique d'un patient virtuel via le terminal :
```bash
python -m src.pipeline.predict_pipeline
```

---

## ⚖️ 10. Éthique, Cadre d'Apprentissage & Avertissement Médical

- 🎓 **Nature Pédagogique du Projet** : Ce projet est développé dans un cadre strict d'**apprentissage, de recherche et de portfolio technique**. Il vise à démontrer la maîtrise de l'ensemble du cycle de vie MLOps (Data Engineering, Modeling, Evaluation, Deployment).
- 🏥 **Non-Substitution à l'Avis Médical** : Ce système d'aide à la décision clinique (SADC) est un prototype algorithmique d'apprentissage. **Il ne constitue pas un dispositif médical certifié (CE/FDA) et ne doit en aucun cas être utilisé pour établir un diagnostic médical en situation clinique réelle.** Tout diagnostic de maladie coronarienne exige une évaluation clinique par un médecin cardiologue et des examens de référence (coronarographie, échographie, scintigraphie).
- 🔒 **Protection des Données & RGPD** : Le traitement s'effectue intégralement en local sans transmission de données de santé vers des serveurs externes. L'identifiant patient (`patientid`) est rigoureusement exclu de la modélisation pour respecter le principe de minimisation des données.

---

## 👨‍💻 11. Auteur & Licence

- **Auteur :** Omar Diop
- **Email :** [od14034@gmail.com](mailto:od14034@gmail.com)
- **Dépôt GitHub :** [Cardiovascular_Disease_ML_Project](https://github.com/omar-034/Cardiovascular_Disease_ML_Project)
- **Licence :** Distribué sous licence **MIT**. Consultez le fichier `LICENSE` pour plus de détails.
