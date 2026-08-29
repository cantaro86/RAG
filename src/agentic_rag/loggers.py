import logging
import sys
import threading
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from agentic_rag._load_env import cfg


class Logger:
    FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] -  %(message)s")
    LOG_FILE = cfg.log_file if cfg.log_to_file else "agent_rag.log"
    _FACTORY_HANDLER_ATTRIBUTE = "_agentic_rag_factory_handler"

    _file_handler: logging.Handler | None = None
    _console_handler: logging.Handler | None = None
    _lock = threading.RLock()

    @classmethod
    def _get_console_handler(cls):
        with cls._lock:
            if cls._console_handler is None:
                cls._console_handler = logging.StreamHandler(sys.stdout)
                cls._console_handler.setFormatter(cls.FORMATTER)
                setattr(cls._console_handler, cls._FACTORY_HANDLER_ATTRIBUTE, "console")
            return cls._console_handler

    @classmethod
    def _get_file_handler(cls):
        with cls._lock:
            if cls._file_handler is None:
                log_path = Path(cls.LOG_FILE).expanduser()
                log_path.parent.mkdir(parents=True, exist_ok=True)
                cls._file_handler = TimedRotatingFileHandler(
                    log_path,
                    when="midnight",
                    backupCount=7,
                    encoding="utf-8",
                    delay=True,
                )
                cls._file_handler.setFormatter(cls.FORMATTER)
                setattr(cls._file_handler, cls._FACTORY_HANDLER_ATTRIBUTE, "file")
            return cls._file_handler

    @classmethod
    def get_logger(cls, logger_name):
        logger = logging.getLogger(logger_name)
        with cls._lock:
            level_map = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }
            if cfg.log_level not in level_map:
                raise ValueError(f"Invalid log level: {cfg.log_level}")
            logger.setLevel(level_map[cfg.log_level])

            factory_handlers = [
                handler for handler in logger.handlers if hasattr(handler, cls._FACTORY_HANDLER_ATTRIBUTE)
            ]
            external_handlers = [handler for handler in logger.handlers if handler not in factory_handlers]
            real_external_handlers = [
                handler for handler in external_handlers if not isinstance(handler, logging.NullHandler)
            ]

            if real_external_handlers:
                for handler in factory_handlers:
                    logger.removeHandler(handler)
            else:
                for handler in external_handlers:
                    logger.removeHandler(handler)

                handler_kind = "file" if cfg.log_to_file else "console"
                matching_handlers = [
                    handler
                    for handler in factory_handlers
                    if getattr(handler, cls._FACTORY_HANDLER_ATTRIBUTE) == handler_kind
                ]
                if not matching_handlers:
                    logger.addHandler(cls._get_file_handler() if cfg.log_to_file else cls._get_console_handler())
                for duplicate in matching_handlers[1:]:
                    logger.removeHandler(duplicate)
                for stale_handler in factory_handlers:
                    if getattr(stale_handler, cls._FACTORY_HANDLER_ATTRIBUTE) != handler_kind:
                        logger.removeHandler(stale_handler)

            logger.propagate = False
            return logger
