"""
Module de gestion centralisée des exceptions personnalisées (CustomException).
Enrichit les messages d'erreurs avec le nom du fichier source et le numéro de ligne.
"""

from types import TracebackType
from src.logger import logger


def get_error_message(error: Exception, tb: TracebackType) -> str:
    """
    Formate un message d'erreur détaillé incluant le fichier, la ligne et le message d'origine.
    """
    if tb is None:
        return f"Error: {str(error)}"
    
    filename = tb.tb_frame.f_code.co_filename
    line_number = tb.tb_lineno
    return f"Error in file '{filename}' at line {line_number}: {str(error)}"


class CustomException(Exception):
    """
    Classe d'exception personnalisée étendant Exception et consignant l'erreur dans les logs.
    """
    def __init__(self, error_message: Exception):
        super().__init__(str(error_message))
        self.error_message = get_error_message(error_message, error_message.__traceback__)
        logger.error(self.error_message)
        
    def __str__(self) -> str:
        return self.error_message
