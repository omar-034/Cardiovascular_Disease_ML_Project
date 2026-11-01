# 🫀 CardioScan AI — Système Expert de Dépistage et Prédiction du Risque Cardiovasculaire

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![ML Framework](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![UI](https://img.shields.io/badge/Streamlit-1.66+-FF4B4B.svg)](https://streamlit.io/)
[![Recall](https://img.shields.io/badge/Clinical%20Recall-100%25-brightgreen.svg)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9999-success.svg)]()
[![Status](https://img.shields.io/badge/Status-Production--Ready-blueviolet.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **CardioScan AI** est une solution intégrale de Machine Learning et un système d'aide à la décision clinique conçu pour l'évaluation et la stratification du risque de maladies cardiovasculaires et coronariennes à partir de biomarqueurs physiologiques et électrocardiographiques d'effort.

---

## 📑 Table des Matières
- [1. Contexte Médical & Objectifs](#-1-contexte-médical--objectifs)
- [2. Architecture & Cycle de Vie MLOps](#-2-architecture--cycle-de-vie-mlops)
- [3. Résultats & Métriques de Performance](#-3-résultats--métriques-de-performance)
- [4. Ingénierie des Caractéristiques Cliniques](#-4-ingénierie-des-caractéristiques-cliniques)
- [5. Arborescence du Projet](#-5-arborescence-du-projet)
- [6. Installation & Démarrage Rapide](#-6-installation--démarrage-rapide)
- [7. Utilisation de l'Application Web Streamlit](#-7-utilisation-de-lapplication-web-streamlit)
- [8. Pipeline de Réentraînement & Inférence Automatisée](#-8-pipeline-de-réentraînement--inférence-automatisée)
- [9. Auteur & Licence](#-9-auteur--licence)

---

## 🩺 1. Contexte Médical & Objectifs

Les cardiopathies ischémiques représentent la première cause de mortalité dans le monde. Dans le domaine du diagnostic médical assisté par ordinateur, le **coût d'une erreur est hautement asymétrique** :
- **Faux Positif (FP)** : Déclenche des examens complémentaires de réassurance (échographie, bilan lipidique) sans danger vital immédiat.
- **Faux Négatif (FN)** : Renvoie un patient coronarien chez lui sans prise en charge, augmentant drastiquement le risque d'infarctus du myocarde (IDM) ou d'arrêt cardiaque.

**Objectif Prioritaire :** Maximiser la sensibilité (**Recall = 100%**) et le **$F_2$-Score** (qui double le poids accordé au Rappel par rapport à la Précision), tout en maintenant une excellente spécificité et une exactitude globale supérieure à 98%.

---

## 🏗️ 2. Architecture & Cycle de Vie MLOps

Le projet a été mené selon un cycle de développement modulaire et test-driven, formalisé par 6 phases de travail indépendantes :

```
[ Données Cliniques Brutes ]
            │
            ▼ (Phase 1: EDA)
[ 4 Notebooks Exploratoires ] ──> Détection anomalies (cholestérol nul, pente ST hors plage)
            │
            ▼ (Phase 2: Ingestion & Feature Engineering)
[ ClinicalFeatureEngineer ] ──> Imputation conditionnelle + 3 Biomarqueurs dérivés
            │
            ▼ (Phase 2: Prétraitement Anti-Fuite)
[ ColumnTransformer ] ──> RobustScaler (numériques) + OneHot (nominales) + StandardScaler (ordinales)
            │
            ▼ (Phase 3: Entraînement & Benchmark)
[ 5-Fold Stratified CV ] ──> Benchmark de 5 modèles (HistGradient, RF, XGB, LogReg, SVC)
            │
            ▼ (Phase 3: Calibration du Seuil)
[ Optimal Threshold θ* ] ──> Calibration post-modèle orientée F2-Score (Recall = 100%)
            │
            ▼ (Phase 4: Pipelines de Production)
[ TrainPipeline & PredictPipeline ] ──> Sérialisation / Inférence unitaire & batch
            │
            ▼ (Phase 5: Interface Utilisateur)
[ Application Streamlit (app.py) ] ──> UI clinique responsive avec profil type & export CSV
```

---

## 📊 3. Résultats & Métriques de Performance

### Benchmark en Validation Croisée (Stratified 5-Fold CV)

| Modèle | Recall Moyen | $F_2$-Score Moyen | ROC-AUC Moyen | Accuracy Moyenne |
| :--- | :---: | :---: | :---: | :---: |
| 🏆 **HistGradientBoosting** | **98.49%** | **0.9807** | **0.9950** | **97.00%** |
| Random Forest | 98.06% | 0.9755 | 0.9927 | 96.25% |
| XGBoost | 97.62% | 0.9724 | 0.9916 | 96.12% |
| Logistic Regression | 97.41% | 0.9728 | 0.9918 | 96.62% |
| SVC (Kernel RBF) | 97.41% | 0.9720 | 0.9928 | 96.38% |

### Performance Finale sur le Jeu de Test Indépendant (Holdout 20%)

Après calibration du seuil clinique optimal $\theta^* = 0.10$ :
- **Rappel Clinique (Recall / Sensibilité) : 100.0%** (0 Faux Négatif sur l'échantillon test)
- **$F_2$-Score Clinique : 0.9966**
- **Exactitude Globale (Accuracy) : 99.00%**
- **Précision : 98.31%**
- **Aire sous la courbe ROC-AUC : 0.9999**

---

## 🔬 4. Ingénierie des Caractéristiques Cliniques

Le module `ClinicalFeatureEngineer` enrichit les 12 biomarqueurs de base avec des indices physiologiques de référence :

1. **Ratio de Réserve Cardiaque (`hr_reserve_ratio`)** :
   $$\text{hr\_reserve\_ratio} = \frac{\text{maxheartrate}}{220 - \text{age}}$$
   Évalue la capacité chronotrope du myocarde sous stress.
2. **Indice de Charge Ischémique (`ischemia_burden_index`)** :
   $$\text{ischemia\_burden\_index} = \text{oldpeak} \times \mathbb{I}(\text{exerciseangia} = 1)$$
   Combine l'expression symptomatique de l'angine et la sévérité du sous-décalage électrique ST.
3. **Proxy de Rigidité Vasculaire (`vascular_stiffness_proxy`)** :
   $$\text{vascular\_stiffness\_proxy} = \frac{\text{restingBP}}{\text{age} + 1}$$
   Reflète la charge hémodynamique normalisée par l'âge physiologique.
4. **Correction Biologique du Cholestérol Nul** :
   Remplacement des zéros physiologiquement impossibles (`serumcholestrol == 0`) par la médiane conditionnelle calculée exclusivement sur le sous-ensemble d'apprentissage (garantie stricte d'absence de fuite d'information).

---

## 📁 5. Arborescence du Projet

```
Cardiovascular Disease/
├── artifacts/                           # Modèles et métriques sérialisés (non suivis par Git)
│   ├── model.pkl                        # Modèle champion sérialisé
│   ├── preprocessor.pkl                 # Pipeline de transformation scellée
│   ├── threshold.json                   # Seuil clinique calibré (θ*)
│   ├── metrics.json                     # Métriques de validation et de test
│   └── model_comparison.csv             # Tableau comparatif du benchmark
├── notebooks/
│   ├── data/
│   │   └── cardiovascular.csv           # Données cliniques de référence
│   ├── 01_Data_overview.ipynb           # Phase 1.1 : Structure et intégrité
│   ├── 02_Univariate_analysis.ipynb     # Phase 1.2 : Distributions univariées
│   ├── 03_Bivariate_analysis.ipynb      # Phase 1.3 : Corrélations et relations cibles
│   ├── 04_Multivariate_analysis.ipynb   # Phase 1.4 : Clusters et ACP
│   └── 05_Model_Training.ipynb          # Phase 3 : Benchmark & optimisation du seuil
├── src/
│   ├── __init__.py
│   ├── exception.py                     # Gestion unifiée des exceptions
│   ├── logger.py                        # Journalisation horodatée des événements
│   ├── utils.py                         # Utilitaires de sérialisation d'objets
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py            # Phase 2 : Découpage stratifié 80/20
│   │   ├── feature_engineering.py       # Phase 2 : Ingénierie clinique
│   │   ├── data_transformation.py       # Phase 2 : Préprocesseur scikit-learn
│   │   └── model_trainer.py             # Phase 3 : Entraîneur et sélection du champion
│   └── pipeline/
│       ├── __init__.py
│       ├── train_pipeline.py            # Phase 4 : Pipeline d'entraînement complet
│       └── predict_pipeline.py          # Phase 4 : Moteur d'inférence (CustomData)
├── app.py                               # Phase 5 : Application Web Streamlit
├── requirements.txt                     # Dépendances minimales sécurisées
├── setup.py                             # Configuration du packaging Python
└── README.md                            # Documentation complète du projet
```

---

## 🚀 6. Installation & Démarrage Rapide

### Prérequis
- Python 3.10 ou version ultérieure
- Git

### 1. Cloner le Dépôt
```bash
git clone https://github.com/votre-nom/Cardiovascular_Disease_ML_Project.git
cd "Cardiovascular Disease"
```

### 2. Créer et Activer un Environnement Virtuel
```bash
# Sous Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Sous Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installer les Dépendances
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🖥️ 7. Utilisation de l'Application Web Streamlit

Lancez l'interface graphique interactive avec la commande :

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement à l'adresse : **`http://localhost:8501`**

### Fonctionnalités Clés de l'Application :
1. **Évaluation Individuelle (Saisie Patient)** :
   - Formulaire clinique complet organisé par domaines (Démographie / Épreuve d'effort).
   - Calcul et affichage en temps réel des 3 biomarqueurs dérivés.
   - Chargement en 1 clic de profils types (*Patient Témoin Sain*, *Risque Modéré*, *Cas Ischémique Critique*).
   - Diagnostic immédiat, niveau de risque gradué, jauge visuelle et recommandations médicales prioritaires.
2. **Dépistage par Cohorte (Batch CSV)** :
   - Téléversement d'un fichier CSV multi-patients.
   - Modèle CSV d'exemple téléchargeable directement depuis l'application.
   - Évaluation automatisée, statistiques d'incidence de cohorte et export du rapport CSV enrichi.
3. **Gouvernance & Transparence MLOps** :
   - Consultation des métriques de validation croisée et de test.
   - Explication de la politique de décision clinique et de la minimisation des faux négatifs.

---

## ⚙️ 8. Pipeline de Réentraînement & Inférence Automatisée

### Exécuter le Réentraînement Complet en 1 Ligne
Pour relancer l'ingestion, le préprocessing, le benchmark et la calibration du seuil :
```bash
python -m src.pipeline.train_pipeline
```

### Tester l'Inférence en Ligne de Commande
Pour évaluer un patient virtuel avec le moteur `predict_pipeline` :
```bash
python -m src.pipeline.predict_pipeline
```

---

## 👨‍💻 9. Auteur & Licence

- **Auteur :** Omar Diop
- **Email :** od14034@gmail.com
- **Projet :** Système Expert de Prédiction du Risque Cardiovasculaire
- **Licence :** Ce projet est distribué sous licence MIT. Consultez le fichier `LICENSE` pour plus de détails.
