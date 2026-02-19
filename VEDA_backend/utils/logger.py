import logging
import sys
from logging.handlers import RotatingFileHandler

# Create a custom logger
logger = logging.getLogger("VEDA_API")
logger.setLevel(logging.INFO)

# Create handlers
c_handler = logging.StreamHandler(sys.stdout)
f_handler = RotatingFileHandler('app.log', maxBytes=5*1024*1024, backupCount=2) # 5MB per file

c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
# Format: TIME - LEVEL - MESSAGE
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(log_format)
f_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

def get_logger():
    return logger
