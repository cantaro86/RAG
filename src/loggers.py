import logging
import sys
from logging.handlers import TimedRotatingFileHandler

from ._load_env import cfg


class Logger:
    # %(levelname)s: [%(filename)s:%(lineno)s - %(funcName)10s() ] %(message)s
    FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] -  %(message)s")
    LOG_FILE = cfg.log_file if cfg.log_to_file else "agent_rag.log"

    @classmethod
    def _get_console_handler(cls):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(cls.FORMATTER)
        return console_handler

    @classmethod
    def _get_file_handler(cls):
        file_handler = TimedRotatingFileHandler(cls.LOG_FILE, when="midnight", backupCount=7, encoding="utf-8")
        file_handler.setFormatter(cls.FORMATTER)
        return file_handler

    @classmethod
    def get_logger(cls, logger_name):
        logger = logging.getLogger(logger_name)

        if logger.handlers:
            return logger

        if cfg.log_level == "DEBUG":
            logger.setLevel(logging.DEBUG)
        elif cfg.log_level == "INFO":
            logger.setLevel(logging.INFO)
        elif cfg.log_level == "WARNING":
            logger.setLevel(logging.WARNING)
        elif cfg.log_level == "ERROR":
            logger.setLevel(logging.ERROR)
        else:
            raise ValueError(f"Invalid log level: {cfg.log_level}")

        if cfg.log_to_file:
            logger.addHandler(cls._get_file_handler())
        else:
            logger.addHandler(cls._get_console_handler())

        # with this pattern, it's rarely necessary to propagate the error up to parent
        logger.propagate = False
        return logger
