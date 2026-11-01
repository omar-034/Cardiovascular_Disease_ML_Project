"""
Pipeline d'inférence et de prédiction clinique (Inference & Prediction Pipeline).
Permet de charger les artefacts sérialisés (préprocesseur, modèle champion, seuil optimal)
et de prédire le risque de maladie cardiovasculaire pour un ou plusieurs patients.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logger
from src.utils import load_object


class PredictPipeline:
    """
    Moteur d'inférence pour la prédiction du risque cardiovasculaire.
    """
    def __init__(self, artifacts_dir: str = None):
        if artifacts_dir is None:
            # Détection dynamique du dossier artifacts
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            self.artifacts_dir = os.path.join(project_root, "artifacts")
        else:
            self.artifacts_dir = artifacts_dir

        self.model_path = os.path.join(self.artifacts_dir, "model.pkl")
        self.preprocessor_path = os.path.join(self.artifacts_dir, "preprocessor.pkl")
        self.threshold_path = os.path.join(self.artifacts_dir, "threshold.json")

    def _load_decision_threshold(self) -> float:
        """
        Charge le seuil clinique optimisé lors de la Phase 3.
        Repli sur 0.50 si le fichier n'est pas trouvé.
        """
        try:
            if os.path.exists(self.threshold_path):
                with open(self.threshold_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return float(data.get("optimal_threshold", 0.50))
            logger.warning(f"Fichier de seuil non trouvé ({self.threshold_path}). Utilisation du seuil par défaut 0.50.")
            return 0.50
        except Exception as e:
            logger.warning(f"Erreur lors du chargement du seuil : {e}. Utilisation du seuil 0.50.")
            return 0.50

    def predict(self, features: pd.DataFrame) -> dict:
        """
        Évalue le risque cardiovasculaire pour les observations fournies dans le DataFrame.
        
        Args:
            features (pd.DataFrame): Données brutes des patients comportant les 12 biomarqueurs requis.
            
        Returns:
            dict: Dictionnaire contenant les diagnostics (0/1), les probabilités de risque,
                  le seuil appliqué, le niveau de risque qualitatif et les recommandations médicales.
        """
        try:
            logger.info("Chargement des artefacts sérialisés pour l'inférence...")
            
            if not os.path.exists(self.preprocessor_path):
                raise FileNotFoundError(f"Préprocesseur introuvable dans : {self.preprocessor_path}")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modèle introuvable dans : {self.model_path}")

            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)
            threshold = self._load_decision_threshold()

            logger.info(f"Transformation des caractéristiques patient (taille initiale : {features.shape})...")
            # Transformation via ClinicalFeatureEngineer + ColumnTransformer
            data_scaled = preprocessor.transform(features)

            logger.info("Calcul des probabilités de risque par le modèle champion...")
            probabilities = model.predict_proba(data_scaled)[:, 1]

            # Application du seuil clinique optimisé
            predictions = (probabilities >= threshold).astype(int)

            results = []
            for idx, prob in enumerate(probabilities):
                pred = int(predictions[idx])
                
                # Échelle de graduation du risque
                if prob < 0.25:
                    risk_level = "Très Faible Risque"
                    recommendation = "Profil rassurant. Poursuivre une bonne hygiène de vie et un contrôle médical régulier."
                elif prob < threshold:
                    risk_level = "Risque Modéré (Zone de Vigilance)"
                    recommendation = "Présence de signaux légers. Surveillance recommandée de la tension et du bilan lipidique."
                elif prob < 0.75:
                    risk_level = "Risque Élevé (Suspicion Forte)"
                    recommendation = "Alerte diagnostique. Consultation cardiologique et explorations complémentaires (épreuve d'effort) conseillées."
                else:
                    risk_level = "Risque Critique (Urgence Cardiologique)"
                    recommendation = "Suspicion très sévère de coronaropathie. Prise en charge cardiologique spécialisée urgente requise."

                results.append({
                    "patient_index": idx,
                    "prediction": pred,
                    "prediction_label": "Pathologie Cardiaque Détectée" if pred == 1 else "Absence de Pathologie Détectée",
                    "risk_probability": round(float(prob), 4),
                    "risk_percentage": f"{prob * 100:.1f}%",
                    "threshold_applied": round(float(threshold), 2),
                    "risk_level": risk_level,
                    "clinical_recommendation": recommendation
                })

            return {
                "total_patients": len(features),
                "threshold_applied": round(float(threshold), 2),
                "patients_evaluated": results
            }

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction : {e}")
            raise CustomException(e)


class CustomData:
    """
    Classe utilitaire représentant les biomarqueurs d'un patient unique,
    permettant de convertir des entrées saisies (formulaire, API) en DataFrame exploitable par le pipeline.
    """
    def __init__(
        self,
        age: float,
        gender: int,
        chestpain: int,
        restingBP: float,
        serumcholestrol: float,
        fastingbloodsugar: int,
        restingrelectro: int,
        maxheartrate: float,
        exerciseangia: int,
        oldpeak: float,
        slope: int,
        noofmajorvessels: int
    ):
        self.age = age
        self.gender = gender
        self.chestpain = chestpain
        self.restingBP = restingBP
        self.serumcholestrol = serumcholestrol
        self.fastingbloodsugar = fastingbloodsugar
        self.restingrelectro = restingrelectro
        self.maxheartrate = maxheartrate
        self.exerciseangia = exerciseangia
        self.oldpeak = oldpeak
        self.slope = slope
        self.noofmajorvessels = noofmajorvessels

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """
        Convertit les paramètres cliniques saisis en DataFrame structuré
        avec les noms de colonnes exacts attendus par la pipeline.
        """
        try:
            data_dict = {
                "age": [self.age],
                "gender": [self.gender],
                "chestpain": [self.chestpain],
                "restingBP": [self.restingBP],
                "serumcholestrol": [self.serumcholestrol],
                "fastingbloodsugar": [self.fastingbloodsugar],
                "restingrelectro": [self.restingrelectro],
                "maxheartrate": [self.maxheartrate],
                "exerciseangia": [self.exerciseangia],
                "oldpeak": [self.oldpeak],
                "slope": [self.slope],
                "noofmajorvessels": [self.noofmajorvessels]
            }
            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(e)


if __name__ == "__main__":
    print("--- TEST UNITAIRE DU PREDICT PIPELINE SUR UN PATIENT VIRTUEL ---")
    
    # Simulation d'un patient à risque ischémique élevé :
    # Homme de 58 ans, douleur angineuse typique, ST depression 3.5 mm, angine d'effort, 2 vaisseaux atteints
    high_risk_patient = CustomData(
        age=58,
        gender=1,
        chestpain=3,
        restingBP=160,
        serumcholestrol=280,
        fastingbloodsugar=1,
        restingrelectro=2,
        maxheartrate=115,
        exerciseangia=1,
        oldpeak=3.5,
        slope=2,
        noofmajorvessels=2
    )

    df_patient = high_risk_patient.get_data_as_data_frame()
    print("Données du patient injectées :")
    print(df_patient)

    pipeline = PredictPipeline()
    result = pipeline.predict(df_patient)

    print("\n--- RÉSULTAT DE L'ÉVALUATION CLINIQUE ---")
    patient_res = result["patients_evaluated"][0]
    print(f"Diagnostic      : {patient_res['prediction_label']}")
    print(f"Probabilité     : {patient_res['risk_percentage']} (Seuil: {patient_res['threshold_applied']})")
    print(f"Niveau de Risque: {patient_res['risk_level']}")
    print(f"Recommandation  : {patient_res['clinical_recommendation']}")

