"""
Logging centralisé pour l'ensemble du projet ML.
Utilise des chemins absolus pour garantir la reproductibilité.
"""

import logging
from pathlib import Path

def get_logger(name, log_level=logging.DEBUG):
    """
    Retourne un logger configuré avec fichier et console.
    
    Args:
        name (str): Nom du logger (__name__)
        log_level: Niveau de logging (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        logging.Logger: Logger configuré avec handlers fichier et console
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Message d'information")
    """
    
    # Chemin absolu au répertoire logs/ (reproductible)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    log_dir = project_root / 'logs'
    
    # Créer le répertoire s'il n'existe pas
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Créer le logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Éviter les doublons de handlers si le logger est appelé plusieurs fois
    if logger.handlers:
        return logger
    
    # Format détaillé avec informations utiles pour le debugging
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler pour fichier (verbose)
    log_file = log_dir / f"{name.replace('.', '_')}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Handler pour console (moins verbose)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


# Logger racine pour la compatibilité avec le code existant
logger = get_logger(__name__)