from datetime import datetime
from typing import Literal
import config

configuration = config.get_config_object()
log_file = configuration.resume.resume_log_file

def log_writer(message: str, log_type: Literal["INFO", "ERROR"]) -> None:
    """
    Writes a log message to a specified log file with a timestamp and log type.

    Args:
        message (str): The message to log.
        log_type (Literal["INFO", "ERROR"]): The type of log entry, either "INFO" or "ERROR".
    
    Log File Output:
        A log file specified by `configuration.taxonomy.taxonomy_log_file` is saved in the `log_dir`.
        Each entry is prefixed with the current datetime and log type.
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as myfile:
        myfile.write(f'{current_time}" {log_type}: {message} "\n"')