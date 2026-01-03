import logging
import os
import sys
from logging.handlers import RotatingFileHandler

def setup_logger(name="zertam", log_dir="logs", level=logging.INFO):
    """
    Sets up a logger with console and file output.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File Handler (Rotating)
    log_file = os.path.join(log_dir, "zertam.log")
    file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def add_gui_handler(logger, handler):
    """Adds a custom handler (e.g. for GUI display) to the logger."""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
