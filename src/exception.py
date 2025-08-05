import sys

# Helper function to extract detailed error info
def error_message_details(error, error_details: sys):
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

# Custom Exception class
class CustomException(Exception):
    def __init__(self, error, error_details: sys):
        self.error_message = error_message_details(error, error_details)
        super().__init__(self.error_message)

    def __str__(self):
        return self.error_message
    

