import sys
import streamlit as st
from configparser import ConfigParser
import time
import logging
import os
from datetime import datetime
import logging.config
import sys
from types import TracebackType
from typing import Type

class CustomLogger:
    """A custom logger class to set up and handle logging."""
    def __init__(self):
        """Initializes the CustomLogger by setting up the logging configuration."""
        self.setup_logger()

    def setup_logger(self):
        """Sets up the logger configuration with a default logging dictionary."""
        DEFAULT_LOGGING = {
            'version': 1,
            'disable_existing_loggers': False,
            'loggers': {
                '': {
                    'level': 'INFO',
                },
                'another.module': {
                    'level': 'INFO',
                },
            }
        }

        logging.config.dictConfig(DEFAULT_LOGGING)
        dt = datetime.now()
        log_path = f'./logs/{dt.strftime("%d-%m-%y")}'
        os.makedirs(log_path, exist_ok = True)
        logging.basicConfig(
            filename = f'{log_path}/{dt.strftime("%H")}.log',
            level = logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        self.logger = logging
        self.logger.info("-------------------------------------")

    def write_error_log(self, exc_type: Type[BaseException], exc_tb: TracebackType, msg: str) -> None:
        """
        Logs an error message with exception details.

        Args:
            exc_type (Type[BaseException]): The type of exception.
            exc_tb (TracebackType): The traceback object for the exception.
            msg (str): The error message to log.
        """
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        error = str(msg)+" | "+str(exc_type)+" | "+ str(fname) +" | "+ str(exc_tb.tb_lineno)+" | "
        self.logger.error(error)

class Utils(CustomLogger):
    """
    A utility class that extends CustomLogger to include configuration file reading.
    """
    def __init__(self):
        """
        Initializes the Utils class, reads the configuration file, and logs errors if any.
        """
        super().__init__()
        self.read_config()

    def read_config(self):
        """
        Reads the configuration file and stores its contents in a dictionary.
        Logs an error if the file is not found or another exception occurs.
        """
        self.config = ConfigParser()
        config_file = "config.ini"
        try:
            if not os.path.exists(config_file):
                raise Exception("Config file not found...")

            self.config.read(config_file)
            time.sleep(2)
            self.config_data = {}
            for section in self.config.sections():
                for field in self.config.options(str(section)):
                    self.config_data[field] = str(self.config.get(str(section),field))

        except Exception as e:   
            exc_type, exc_obj, exc_tb = sys.exc_info()
            msg = exc_obj.args[0]
            self.write_error_log(exc_type,exc_tb,msg)
            st.error(msg)
            sys.exit(msg)

            
    