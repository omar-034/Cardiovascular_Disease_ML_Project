"""
Module d'ingénierie des caractéristiques cliniques (Feature Engineering).
Définit le transformateur Scikit-Learn ClinicalFeatureEngineer pour l'enrichissement
des biomarqueurs cardiovasculaires et le traitement des zéros physiologiques.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException
from src.logger import logger


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformateur Scikit-Learn sur-mesure pour :
    1. Corriger les aberrations physiologiques (remplacer serumcholestrol == 0 par NaN).
    2. Créer des biomarqueurs et ratios cliniques dérivés issus de la littérature médicale :
       - hr_reserve_ratio : Fréquence cardiaque max / FC max théorique (220 - âge)
       - ischemia_burden_index : ST depression * exercice angina
       - vascular_stiffness_proxy : (Pression artérielle * âge) / 100
    """
    def __init__(self, add_features: bool = True):
        self.add_features = add_features

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Méthode d'ajustement sans état (stateless transformer).
        """
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Applique l'imputation conditionnelle et le calcul des biomarqueurs composites.
        
        Args:
            X (pd.DataFrame): DataFrame des variables cliniques brutes.
            y (Optional[pd.Series]): Vecteur cible optionnel.
            
        Returns:
            pd.DataFrame: DataFrame enrichi des nouveaux biomarqueurs cliniques.
        """
        try:
            X_df = X.copy()
            if not isinstance(X_df, pd.DataFrame):
                raise TypeError("ClinicalFeatureEngineer attend un pandas DataFrame.")

            # 1. Correction de l'aberration physiologique : cholestérol == 0 -> NaN
            if 'serumcholestrol' in X_df.columns:
                zero_chol_count = (X_df['serumcholestrol'] == 0).sum()
                if zero_chol_count > 0:
                    X_df.loc[X_df['serumcholestrol'] == 0, 'serumcholestrol'] = np.nan

            if self.add_features:
                # 2. Ratio de Réserve Chronotrope (Fréquence cardiaque max / FC max théorique d'Astrand : 220 - âge)
                if 'maxheartrate' in X_df.columns and 'age' in X_df.columns:
                    theoretical_max_hr = 220 - X_df['age']
                    theoretical_max_hr = np.where(theoretical_max_hr <= 0, 150, theoretical_max_hr)
                    X_df['hr_reserve_ratio'] = X_df['maxheartrate'] / theoretical_max_hr

                # 3. Charge Ischémique d'Effort (Interaction synergique dépression ST * angine d'effort)
                if 'oldpeak' in X_df.columns and 'exerciseangia' in X_df.columns:
                    X_df['ischemia_burden_index'] = X_df['oldpeak'] * X_df['exerciseangia']

                # 4. Proxy de Rigidité Artérielle et Surcharge Vasculaire ((Pression artérielle * Âge) / 100)
                if 'restingBP' in X_df.columns and 'age' in X_df.columns:
                    X_df['vascular_stiffness_proxy'] = (X_df['restingBP'] * X_df['age']) / 100.0

            return X_df

        except Exception as e:
            logger.error(f"Erreur dans ClinicalFeatureEngineer.transform : {e}")
            raise CustomException(e)
