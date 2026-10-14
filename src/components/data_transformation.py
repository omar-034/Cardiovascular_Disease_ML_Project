import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object
from src.components.feature_engineering import ClinicalFeatureEngineer


@dataclass
class DataTransformationConfig:
    """
    Configuration pour le composant de transformation des données.
    """
    artifacts_dir: str = os.path.join(os.getcwd(), "artifacts")
    preprocessor_obj_file_path: str = os.path.join(os.getcwd(), "artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    Composant responsable de l'ingénierie des caractéristiques, du traitement
    des valeurs manquantes, de l'encodage et de la normalisation sans fuite d'information.
    """
    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        self.transformation_config = config

    def get_data_transformer_object(self) -> Pipeline:
        """
        Construit et retourne la Pipeline Scikit-Learn globale d'ingénierie et de prétraitement.
        
        Returns:
            Pipeline: Pipeline complète incluant l'ingénierie clinique et le ColumnTransformer.
        """
        try:
            logger.info("Construction de la Pipeline de prétraitement et de Feature Engineering...")

            # Définition des groupes de variables après application de ClinicalFeatureEngineer
            continuous_features = [
                'age',
                'restingBP',
                'serumcholestrol',
                'maxheartrate',
                'oldpeak',
                'hr_reserve_ratio',
                'ischemia_burden_index',
                'vascular_stiffness_proxy'
            ]

            nominal_features = [
                'chestpain',
                'restingrelectro'
            ]

            ordinal_and_binary_features = [
                'gender',
                'fastingbloodsugar',
                'exerciseangia',
                'slope',
                'noofmajorvessels'
            ]

            # 1. Sous-pipeline pour les variables continues (Imputation médiane + RobustScaler contre les outliers)
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])

            # 2. Sous-pipeline pour les variables nominales (Imputation mode + One-Hot Encoding)
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            # 3. Sous-pipeline pour les variables ordinales et binaires (Imputation mode + StandardScaler)
            ord_bin_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('scaler', StandardScaler())
            ])

            # 4. Assemblage via ColumnTransformer
            column_transformer = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, continuous_features),
                    ('cat_pipeline', cat_pipeline, nominal_features),
                    ('ord_bin_pipeline', ord_bin_pipeline, ordinal_and_binary_features)
                ],
                remainder='drop'
            )

            # 5. Pipeline globale englobant l'ingénierie clinique initiale et les transformations
            full_pipeline = Pipeline(steps=[
                ('feature_engineer', ClinicalFeatureEngineer(add_features=True)),
                ('col_transformer', column_transformer)
            ])

            logger.info("Pipeline de transformation assemblée avec succès.")
            return full_pipeline

        except Exception as e:
            raise CustomException(e)

    def initiate_data_transformation(self, train_path: str, test_path: str) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Exécute la transformation complète :
        1. Chargement des jeux train et test.
        2. Séparation de la cible ('target') et exclusion de l'identifiant ('patientid').
        3. Entraînement (fit) du préprocesseur EXCLUSIVEMENT sur le train set.
        4. Transformation du train set et du test set.
        5. Sérialisation et sauvegarde de preprocessor.pkl.

        Args:
            train_path (str): Chemin d'accès au fichier train.csv.
            test_path (str): Chemin d'accès au fichier test.csv.

        Returns:
            tuple[np.ndarray, np.ndarray, str]: 
                - Matrice transformée d'entraînement (features + target).
                - Matrice transformée de test (features + target).
                - Chemin d'accès à l'objet préprocesseur sérialisé.
        """
        try:
            logger.info("=== DÉBUT DU PROCESSUS DE TRANSFORMATION DES DONNÉES ===")
            logger.info(f"Chargement des données : Train='{train_path}', Test='{test_path}'")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = 'target'
            id_column = 'patientid'

            # Séparation Features (X) / Cible (y)
            X_train = train_df.drop(columns=[target_column], errors='ignore')
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column], errors='ignore')
            y_test = test_df[target_column]

            if id_column in X_train.columns:
                X_train = X_train.drop(columns=[id_column])
            if id_column in X_test.columns:
                X_test = X_test.drop(columns=[id_column])

            logger.info(f"Variables d'entrée : {list(X_train.columns)}")
            logger.info(f"Dimensions initiales : X_train={X_train.shape}, X_test={X_test.shape}")

            # Instanciation de la pipeline de transformation
            preprocessing_pipeline = self.get_data_transformer_object()

            # FIT_TRANSFORM sur Train SEULEMENT, puis TRANSFORM sur Test (Anti-Leakage strict)
            logger.info("Ajustement (fit_transform) du préprocesseur sur le jeu d'entraînement...")
            X_train_transformed = preprocessing_pipeline.fit_transform(X_train)

            logger.info("Transformation (transform) du jeu de test à l'aide des paramètres appris...")
            X_test_transformed = preprocessing_pipeline.transform(X_test)

            logger.info(f"Dimensions après transformation : X_train={X_train_transformed.shape}, X_test={X_test_transformed.shape}")

            # Concaténation des matrices de features transformées avec la colonne cible
            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            # Sauvegarde de l'objet préprocesseur sérialisé
            logger.info(f"Sauvegarde du préprocesseur dans : {self.transformation_config.preprocessor_obj_file_path}")
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_pipeline
            )

            logger.info("=== TRANSFORMATION DES DONNÉES TERMINÉE AVEC SUCCÈS ===")
            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    print("--- TEST DE BOUT EN BOUT : INGESTION & TRANSFORMATION ---")
    ingestion = DataIngestion()
    train_data_path, test_data_path = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    train_array, test_array, preprocessor_path = transformer.initiate_data_transformation(
        train_path=train_data_path,
        test_path=test_data_path
    )

    print(f"\nMatrice Train transformée (lignes, colonnes) : {train_array.shape}")
    print(f"Matrice Test transformée  (lignes, colonnes) : {test_array.shape}")
    print(f"Préprocesseur sauvegardé avec succès dans    : {preprocessor_path}")

