"""
Pipeline d'entraînement complet (End-to-End Training Pipeline).
Orchestre l'ingestion des données, la transformation/feature engineering,
et l'entraînement/sélection du modèle champion avec son seuil clinique optimal.
"""

import sys
from src.exception import CustomException
from src.logger import logger
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    """
    Orchestrateur du flux complet d'entraînement et d'évaluation du modèle.
    """
    def __init__(self):
        pass

    def run_pipeline(self) -> dict:
        """
        Exécute le cycle complet d'entraînement :
        1. Ingestion & split stratifié (DataIngestion)
        2. Prétraitement, imputation, feature engineering & normalisation (DataTransformation)
        3. Benchmarking multi-modèles, tuning et seuil clinique optimal (ModelTrainer)
        
        Returns:
            dict: Métriques finales d'évaluation sur le test set.
        """
        try:
            logger.info("==================================================================")
            logger.info("[START] DEMARRAGE DU PIPELINE COMPLET D'ENTRAINEMENT (TRAIN PIPELINE)")
            logger.info("==================================================================")

            # Étape 1 : Ingestion des données
            logger.info("[1/3] Étape 1 : Ingestion des données et partitionnement stratifié...")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            logger.info(f"-> Données d'entraînement : {train_data_path}")
            logger.info(f"-> Données de test         : {test_data_path}")

            # Étape 2 : Transformation & Feature Engineering
            logger.info("[2/3] Étape 2 : Prétraitement et Feature Engineering...")
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_path=train_data_path,
                test_path=test_data_path
            )
            logger.info(f"-> Préprocesseur sérialisé : {preprocessor_path}")
            logger.info(f"-> Matrices transformées    : Train={train_arr.shape}, Test={test_arr.shape}")

            # Étape 3 : Entraînement et Sélection du Modèle Champion
            logger.info("[3/3] Étape 3 : Benchmarking, optimisation du seuil et sélection du modèle champion...")
            model_trainer = ModelTrainer()
            metrics = model_trainer.initiate_model_trainer(train_arr, test_arr)
            
            logger.info("==================================================================")
            logger.info("[SUCCESS] PIPELINE D'ENTRAINEMENT EXECUTE AVEC SUCCES !")
            logger.info(f"-> Modèle Champion         : {metrics['champion_model']}")
            logger.info(f"-> Seuil Clinique Optimal   : {metrics['optimal_threshold']:.2f}")
            logger.info(f"-> Rappel (Sensibilité)     : {metrics['test_recall']*100:.2f}%")
            logger.info(f"-> Score F2                 : {metrics['test_f2_score']:.4f}")
            logger.info(f"-> ROC-AUC Test             : {metrics['test_roc_auc']:.4f}")
            logger.info(f"-> Précision Test           : {metrics['test_precision']*100:.2f}%")
            logger.info("==================================================================")

            return metrics

        except Exception as e:
            logger.error("❌ Échec lors de l'exécution du TrainPipeline.")
            raise CustomException(e)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    summary = pipeline.run_pipeline()
    print("\n--- RÉSUMÉ DE L'ENTRAÎNEMENT ---")
    print(f"Modèle Retenu : {summary['champion_model']}")
    print(f"Seuil de Risque : {summary['optimal_threshold']:.2f}")
    print(f"Rappel Test     : {summary['test_recall']*100:.2f}%")
    print(f"ROC-AUC Test    : {summary['test_roc_auc']:.4f}")

