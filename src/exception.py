from types import TracebackType
from src.logger import logger
def get_error_message(error: Exception, tb: TracebackType) -> str:
    """
    This function takes an exception as input and returns a formatted message containing the filename and the line number where the exception occurred, along with the error message.
    """
    if tb is None:
        return f"Error: {str(error)}"
    
    filename = tb.tb_frame.f_code.co_filename
    line_number = tb.tb_lineno
    return f"Error in file '{filename}' at line {line_number}: {str(error)}"


class CustomException(Exception):
    """
    Custom exception class that extends the built-in Exception class.
    It takes an error message and an optional original exception as input.
    """
    def __init__(self,  error_message:Exception ):
        super().__init__(str(error_message))
        self.error_message = get_error_message(error_message, error_message.__traceback__)
        
        logger.error(self.error_message)
        
    def __str__(self) -> str:
        return self.error_message
    
