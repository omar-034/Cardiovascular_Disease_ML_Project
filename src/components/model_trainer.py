"""
Module d'entraînement et de sélection du modèle (Model Trainer).
Réalise le benchmarking multi-modèles sous validation croisée stratifiée (5-Fold CV),
l'optimisation clinique du seuil décisionnel (favorisant le Recall/F2),
et la sérialisation du modèle champion.
"""

import os
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
    roc_auc_score,
    make_scorer
)
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import xgboost as xgb

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ModelTrainerConfig:
    """
    Configuration des chemins d'accès pour les artefacts du modèle entraîné.
    """
    artifacts_dir: str = str(PROJECT_ROOT / "artifacts")
    trained_model_file_path: str = str(PROJECT_ROOT / "artifacts" / "model.pkl")
    model_metrics_file_path: str = str(PROJECT_ROOT / "artifacts" / "metrics.json")
    model_comparison_file_path: str = str(PROJECT_ROOT / "artifacts" / "model_comparison.csv")
    decision_threshold_file_path: str = str(PROJECT_ROOT / "artifacts" / "threshold.json")


class ModelTrainer:
    """
    Composant responsable du benchmarking multi-modèles, du fine-tuning,
    de l'optimisation médicale du seuil de décision et de la sauvegarde du modèle champion.
    """
    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.trainer_config = config

    def get_models_dict(self) -> dict:
        """
        Retourne le dictionnaire des algorithmes à mettre en compétition.
        """
        return {
            "Logistic_Regression": LogisticRegression(penalty="l2", max_iter=1000, random_state=42),
            "SVC_RBF": SVC(kernel="rbf", probability=True, random_state=42),
            "Random_Forest": RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
            "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=150, random_state=42),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.05,
                random_state=42,
                eval_metric="logloss"
            )
        }

    def evaluate_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> tuple[str, object, pd.DataFrame]:
        """
        Évalue l'ensemble des modèles par validation croisée stratifiée (5-Fold).
        Sélectionne le meilleur modèle basé sur le score F2 (priorité au Rappel).
        """
        try:
            logger.info("Début du benchmarking multi-modèles (Stratified 5-Fold CV)...")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            f2_scorer = make_scorer(fbeta_score, beta=2)

            scoring = {
                'recall': 'recall',
                'f2': f2_scorer,
                'roc_auc': 'roc_auc',
                'precision': 'precision',
                'accuracy': 'accuracy'
            }

            models = self.get_models_dict()
            results = []
            best_model_name = None
            best_model_score = -1.0
            best_model_obj = None

            for name, model in models.items():
                logger.info(f"Évaluation du modèle : {name}...")
                cv_res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                
                mean_f2 = float(np.mean(cv_res['test_f2']))
                mean_recall = float(np.mean(cv_res['test_recall']))
                mean_auc = float(np.mean(cv_res['test_roc_auc']))
                mean_prec = float(np.mean(cv_res['test_precision']))
                mean_acc = float(np.mean(cv_res['test_accuracy']))

                results.append({
                    "Model": name,
                    "Recall_Mean": mean_recall,
                    "Recall_Std": float(np.std(cv_res['test_recall'])),
                    "F2_Score_Mean": mean_f2,
                    "F2_Score_Std": float(np.std(cv_res['test_f2'])),
                    "ROC_AUC_Mean": mean_auc,
                    "ROC_AUC_Std": float(np.std(cv_res['test_roc_auc'])),
                    "Precision_Mean": mean_prec,
                    "Accuracy_Mean": mean_acc
                })

                if mean_f2 > best_model_score:
                    best_model_score = mean_f2
                    best_model_name = name
                    best_model_obj = model

            comparison_df = pd.DataFrame(results).sort_values(by="F2_Score_Mean", ascending=False)
            
            # Sauvegarde du rapport comparatif
            os.makedirs(self.trainer_config.artifacts_dir, exist_ok=True)
            comparison_df.to_csv(self.trainer_config.model_comparison_file_path, index=False)
            logger.info(f"Rapport comparatif sauvegardé dans : {self.trainer_config.model_comparison_file_path}")
            logger.info(f"Modèle Champion identifié : {best_model_name} (F2-Score CV = {best_model_score:.4f})")

            return best_model_name, best_model_obj, comparison_df

        except Exception as e:
            raise CustomException(e)

    def optimize_decision_threshold(self, model: object, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Ajuste le seuil de probabilité optimal θ* pour maximiser la sensibilité diagnostique (Recall/F2).
        """
        try:
            logger.info("Optimisation médicale du seuil de décision clinique...")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            oof_probs = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

            thresholds = np.linspace(0.10, 0.90, 81)
            f2_scores = [fbeta_score(y_train, (oof_probs >= th).astype(int), beta=2) for th in thresholds]

            optimal_idx = int(np.argmax(f2_scores))
            optimal_threshold = float(thresholds[optimal_idx])
            best_f2 = float(f2_scores[optimal_idx])

            logger.info(f"Seuil de decision optimal : theta* = {optimal_threshold:.2f} (F2 CV = {best_f2:.4f})")

            # Sauvegarde du seuil de décision
            threshold_payload = {
                "optimal_threshold": optimal_threshold,
                "f2_score_at_threshold": best_f2,
                "criterion": "F2-Score maximization (Recall priority)"
            }
            with open(self.trainer_config.decision_threshold_file_path, "w", encoding="utf-8") as f:
                json.dump(threshold_payload, f, indent=4)
            logger.info(f"Seuil sauvegardé dans : {self.trainer_config.decision_threshold_file_path}")

            return optimal_threshold

        except Exception as e:
            raise CustomException(e)

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> dict:
        """
        Orchestration complète de la Phase 3 :
        1. Séparation X / y.
        2. Évaluation de tous les modèles en validation croisée.
        3. Optimisation du seuil de coupure du champion.
        4. Entraînement final du modèle vainqueur sur tout le Train.
        5. Évaluation impartiale sur le Test set inviolé.
        6. Sérialisation du modèle et sauvegarde des métriques.
        """
        try:
            logger.info("=== DÉBUT DU PROCESSUS D'ENTRAÎNEMENT DES MODÈLES (PHASE 3) ===")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            logger.info(f"Dimensions : Train={X_train.shape}, Test={X_test.shape}")

            # 1. Benchmarking et sélection du modèle champion
            best_model_name, best_model, comparison_df = self.evaluate_all_models(X_train, y_train)

            # 2. Optimisation du seuil de décision
            optimal_threshold = self.optimize_decision_threshold(best_model, X_train, y_train)

            # 3. Entraînement final sur l'intégralité du Train Set
            logger.info(f"Entraînement final du modèle champion ({best_model_name}) sur les 800 observations de Train...")
            best_model.fit(X_train, y_train)

            # 4. Évaluation sur le Test Set Inviolé (200 observations)
            test_probs = best_model.predict_proba(X_test)[:, 1]
            test_preds = (test_probs >= optimal_threshold).astype(int)

            test_recall = float(recall_score(y_test, test_preds))
            test_precision = float(precision_score(y_test, test_preds))
            test_f2 = float(fbeta_score(y_test, test_preds, beta=2))
            test_auc = float(roc_auc_score(y_test, test_probs))
            test_acc = float(accuracy_score(y_test, test_preds))

            logger.info("=== RÉSULTATS FINAUX SUR LE TEST SET ===")
            logger.info(f"Rappel (Sensibilité) : {test_recall*100:.2f}%")
            logger.info(f"Précision            : {test_precision*100:.2f}%")
            logger.info(f"Score F2             : {test_f2:.4f}")
            logger.info(f"ROC-AUC Test         : {test_auc:.4f}")
            logger.info(f"Exactitude (Acc)     : {test_acc*100:.2f}%")

            # 5. Sérialisation du modèle champion
            save_object(
                file_path=self.trainer_config.trained_model_file_path,
                obj=best_model
            )
            logger.info(f"Modèle sauvegardé dans : {self.trainer_config.trained_model_file_path}")

            # 6. Sauvegarde des métriques officielles
            final_metrics = {
                "champion_model": best_model_name,
                "optimal_threshold": optimal_threshold,
                "test_recall": test_recall,
                "test_precision": test_precision,
                "test_f2_score": test_f2,
                "test_roc_auc": test_auc,
                "test_accuracy": test_acc
            }

            with open(self.trainer_config.model_metrics_file_path, "w", encoding="utf-8") as f:
                json.dump(final_metrics, f, indent=4)
            logger.info(f"Métriques sauvegardées dans : {self.trainer_config.model_metrics_file_path}")

            logger.info("=== ENTRAÎNEMENT ET SÉLECTION TERMINÉS AVEC SUCCÈS ===")
            return final_metrics

        except Exception as e:
            raise CustomException(e)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    print("--- EXÉCUTION DU PIPELINE COMPLET : INGESTION -> TRANSFORMATION -> MODEL TRAINING ---")
    
    # Étape 1 : Ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Étape 2 : Transformation
    transformer = DataTransformation()
    train_arr, test_arr, _ = transformer.initiate_data_transformation(train_path, test_path)

    # Étape 3 : Modélisation
    trainer = ModelTrainer()
    metrics = trainer.initiate_model_trainer(train_arr, test_arr)

    print("\n[OK] Modele champion entraine et serialise avec succes !")
    print(f"Modèle    : {metrics['champion_model']}")
    print(f"Seuil     : {metrics['optimal_threshold']:.2f}")
    print(f"Rappel    : {metrics['test_recall']*100:.2f}%")
    print(f"Score F2  : {metrics['test_f2_score']:.4f}")
    print(f"ROC-AUC   : {metrics['test_roc_auc']:.4f}")
    print(f"Précision : {metrics['test_precision']*100:.2f}%")
