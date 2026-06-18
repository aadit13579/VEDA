import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Log format: TIME - LOGGER NAME - LEVEL - [FILE:LINE:FUNC] - MESSAGE
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s')

def get_logger(name="VEDA_API"):
    """
    Creates and configures a logger with the given name.
    Logs are output to both the console and a rotating file.
    """
    logger = logging.getLogger(name)
    
    # Only add handlers if they are not already added to avoid duplicates
    if not logger.handlers:
        logger.setLevel(logging.DEBUG) # capture down to DEBUG on the logger itself
        
        # Console handler
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(logging.INFO) # Default console to INFO
        c_handler.setFormatter(log_format)
        
        # Rotating file handler — UTF-8 encoding so emoji/Unicode log messages
        # don't silently fail on Windows (cp1252 default would drop them).
        f_handler = RotatingFileHandler(
            'app.log', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
        )
        f_handler.setLevel(logging.DEBUG) # Save all logs to file
        f_handler.setFormatter(log_format)
        
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
    return logger
