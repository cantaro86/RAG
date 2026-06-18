import logging
import sys
from logging.handlers import TimedRotatingFileHandler

from agentic_rag._load_env import cfg


class Logger:
    FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] -  %(message)s")
    LOG_FILE = cfg.log_file if cfg.log_to_file else "agent_rag.log"

    _file_handler: logging.Handler | None = None  # <-- shared singleton
    _console_handler: logging.Handler | None = None  # <-- shared singleton

    @classmethod
    def _get_console_handler(cls):
        if cls._console_handler is None:
            cls._console_handler = logging.StreamHandler(sys.stdout)
            cls._console_handler.setFormatter(cls.FORMATTER)
        return cls._console_handler

    @classmethod
    def _get_file_handler(cls):
        if cls._file_handler is None:  # only created once
            cls._file_handler = TimedRotatingFileHandler(cls.LOG_FILE, when="midnight", backupCount=7, encoding="utf-8")
            cls._file_handler.setFormatter(cls.FORMATTER)
        return cls._file_handler

    @classmethod
    def get_logger(cls, logger_name):
        logger = logging.getLogger(logger_name)

        if logger.handlers:
            return logger

        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        if cfg.log_level not in level_map:
            raise ValueError(f"Invalid log level: {cfg.log_level}")
        logger.setLevel(level_map[cfg.log_level])

        logger.addHandler(cls._get_file_handler() if cfg.log_to_file else cls._get_console_handler())
        logger.propagate = False
        return logger
