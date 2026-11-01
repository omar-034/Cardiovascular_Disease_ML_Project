"""
Module d'ingestion des données cliniques (Data Ingestion).
Charge le dataset brut, valide l'intégrité, supprime les doublons,
et partitionne de manière stratifiée en jeux d'entraînement (80%) et de test (20%).
"""

import os
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logger

# Détermination dynamique de la racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class DataIngestionConfig:
    """
    Configuration des chemins d'accès pour les artefacts d'ingestion de données.
    """
    artifacts_dir: str = str(PROJECT_ROOT / "artifacts")
    raw_data_path: str = str(PROJECT_ROOT / "artifacts" / "raw.csv")
    train_data_path: str = str(PROJECT_ROOT / "artifacts" / "train.csv")
    test_data_path: str = str(PROJECT_ROOT / "artifacts" / "test.csv")
    test_size: float = 0.20
    random_state: int = 42


class DataIngestion:
    """
    Composant responsable du chargement, de la validation d'intégrité
    et du partitionnement stratifié (Train/Test Split) sans fuite de données.
    """
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.ingestion_config = config

    def initiate_data_ingestion(self) -> tuple[str, str]:
        """
        Exécute le pipeline d'ingestion des données :
        1. Recherche et chargement robuste du dataset brut.
        2. Sauvegarde de la copie brute dans artifacts/raw.csv.
        3. Vérification de l'intégrité et suppression d'éventuels doublons.
        4. Split stratifié Train (80%) / Test (20%) sur la variable cible 'target'.
        5. Exportation de artifacts/train.csv et artifacts/test.csv.

        Returns:
            tuple[str, str]: Chemins d'accès respectifs vers train.csv et test.csv.
        """
        logger.info("=== DÉBUT DU PROCESSUS D'INGESTION DES DONNÉES ===")
        try:
            # 1. Recherche robuste du fichier source
            possible_source_paths = [
                str(PROJECT_ROOT / "data" / "cardiovascular.csv"),
                str(PROJECT_ROOT / "notebooks" / "data" / "cardiovascular.csv"),
                os.path.join(os.getcwd(), "data", "cardiovascular.csv"),
                os.path.join(os.getcwd(), "notebooks", "data", "cardiovascular.csv")
            ]

            source_path = next((p for p in possible_source_paths if os.path.exists(p)), None)
            if source_path is None:
                raise FileNotFoundError("Le fichier source 'cardiovascular.csv' est introuvable dans data/ ou notebooks/data/.")

            logger.info(f"Fichier source détecté : {source_path}")
            df = pd.read_csv(source_path)
            logger.info(f"Dataset brut chargé avec succès. Dimensions : {df.shape[0]} lignes, {df.shape[1]} colonnes.")

            # 2. Création du répertoire artifacts
            os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)

            # 3. Sauvegarde de l'intégralité du dataset brut
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.info(f"Données brutes sauvegardées dans : {self.ingestion_config.raw_data_path}")

            # 4. Contrôle d'intégrité
            if 'target' not in df.columns:
                raise ValueError("La colonne cible 'target' est absente du jeu de données.")

            duplicates_count = df.duplicated().sum()
            if duplicates_count > 0:
                logger.warning(f"{duplicates_count} doublons détectés. Suppression en cours...")
                df = df.drop_duplicates().reset_index(drop=True)
                logger.info(f"Dimensions après suppression des doublons : {df.shape}")
            else:
                logger.info("Aucun doublon détecté dans le jeu de données.")

            # 5. Découpage stratifié Train / Test (Split strict anti-data leakage)
            logger.info(f"Partitionnement Train/Test (Ratio: {1 - self.ingestion_config.test_size:.0%} / {self.ingestion_config.test_size:.0%}, Stratifié sur 'target')...")
            train_set, test_set = train_test_split(
                df,
                test_size=self.ingestion_config.test_size,
                random_state=self.ingestion_config.random_state,
                stratify=df['target']
            )

            train_target_ratio = train_set['target'].mean() * 100
            test_target_ratio = test_set['target'].mean() * 100
            logger.info(f"Train Set : {len(train_set)} observations (Taux de malades: {train_target_ratio:.2f}%)")
            logger.info(f"Test Set  : {len(test_set)} observations (Taux de malades: {test_target_ratio:.2f}%)")

            # 6. Sauvegarde des jeux de données d'entraînement et de test
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logger.info(f"Jeu d'entraînement sauvegardé dans : {self.ingestion_config.train_data_path}")
            logger.info(f"Jeu de test sauvegardé dans         : {self.ingestion_config.test_data_path}")
            logger.info("=== INGESTION DES DONNÉES TERMINÉE AVEC SUCCÈS ===")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e)


if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    print(f"Train Data Path: {train_path}")
    print(f"Test Data Path:  {test_path}")
