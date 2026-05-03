"""
VEDA Logging Factory.

Provides a centralized logger factory that creates consistently configured
loggers for every module in the backend. Each logger outputs to both the
console (INFO level) and a rotating file (DEBUG level).

This module exists so that every VEDA component shares the same log format,
rotation policy, and encoding settings — avoiding duplicated setup code.

Leverages: Python standard logging, RotatingFileHandler.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler


class LoggerFactory:
    """
    Factory for creating and configuring VEDA loggers.

    Ensures each named logger is set up exactly once with a console handler
    (INFO) and a rotating file handler (DEBUG). Subsequent calls with the
    same name return the previously configured logger without adding
    duplicate handlers.

    Leverages: logging.StreamHandler, RotatingFileHandler.
    """

    LOG_FORMAT = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - "
        "[%(filename)s:%(lineno)d:%(funcName)s] - %(message)s"
    )

    @staticmethod
    def get_logger(name: str = "VEDA_API") -> logging.Logger:
        """
        Create or retrieve a logger with the given name.

        Returns a logger configured with both console and rotating-file
        handlers. Handlers are added only on first call to prevent
        duplicates when the same logger name is requested multiple times.

        Leverages: logging, RotatingFileHandler with UTF-8 encoding.
        """
        logger = logging.getLogger(name)

        if not logger.handlers:
            logger.setLevel(logging.DEBUG)

            c_handler = logging.StreamHandler(sys.stdout)
            c_handler.setLevel(logging.INFO)
            c_handler.setFormatter(LoggerFactory.LOG_FORMAT)

            f_handler = RotatingFileHandler(
                "app.log",
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            f_handler.setLevel(logging.DEBUG)
            f_handler.setFormatter(LoggerFactory.LOG_FORMAT)

            logger.addHandler(c_handler)
            logger.addHandler(f_handler)

        return logger


get_logger = LoggerFactory.get_logger
