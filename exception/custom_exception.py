import sys
import traceback
from logger.custom_logger import CustomLogger

logger = CustomLogger().get_logger(__file__)
class DocumentPortalException(Exception): #custom exception class that is going to inherit the Exception Class
    """Custom exception for Document Portal"""
    def __init__(self, error_message,error_details:sys): #the init method here takes the error message and the system module
        _,_,exc_tb = error_details.exc_info() #here there is error details. the exc_nfo gives the complete details of the traceback
        self.file_name = exc_tb.tb_frame.f_code.co_filename
        self.lineno = exc_tb.tb_lineno
        self.error_message=str(error_message)
        self.traceback_str = ''.join(traceback.format_exception(*error_details.exc_info())) #this captures more info about the error
    def __str__(self):
        return f"""
            Error in [{self.file_name}] at line [{self.lineno}]
            Message: {self.error_message}
            Traceback:
            {self.traceback_str}
        """

if __name__ == "__main__":
    try:
         a = 1/0
         print(a)
    except Exception as e:
        app_exc = DocumentPortalException(e,sys)
        logger.error(app_exc)
        raise app_exc
