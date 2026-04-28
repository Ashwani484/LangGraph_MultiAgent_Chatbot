import logging
import os
from datetime import datetime
from paths import *

def setup_logger(LOG_DIR, log_file_prefix="AI_Driven_Portfolio_Logs"):
    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Create a timestamped log file
    timestamp = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(LOG_DIR, f"{log_file_prefix}_{timestamp}.log")
    
    # Create handlers
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    console_handler = logging.StreamHandler()
    
    # Define formatter
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger
