"""
Fonctions utilitaires communes pour la sérialisation et le chargement d'objets (modèles, préprocesseurs).
"""

import os
import pickle
from src.exception import CustomException
from src.logger import logger


def save_object(file_path: str, obj: object) -> None:
    """
    Sauvegarde un objet Python (modèle, preprocessor, pipeline) sous format sérialisé (.pkl).
    
    Args:
        file_path (str): Chemin de destination du fichier.
        obj (object): Objet Python à sérialiser.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
        logger.info(f"Objet sauvegardé avec succès dans : {file_path}")
        
    except Exception as e:
        raise CustomException(e)


def load_object(file_path: str) -> object:
    """
    Charge un objet Python sérialisé depuis un fichier (.pkl).
    
    Args:
        file_path (str): Chemin d'accès au fichier sérialisé.
        
    Returns:
        object: L'objet désérialisé.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Le fichier spécifié n'existe pas : {file_path}")
            
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
            
        logger.info(f"Objet chargé avec succès depuis : {file_path}")
        return obj
        
    except Exception as e:
        raise CustomException(e)
