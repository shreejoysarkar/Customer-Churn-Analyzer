import sys
from src.logger import logging


def error_message_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))
    else:
        error_message = f"Error occured: {str(error)}"    
    return error_message
    

class CustomException(Exception):
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_details)

    def __str__(self):
        return self.error_message