# src/logger.py

import logging
import os
from datetime import datetime

def get_logger(name="ContextEval"):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"{name}_{ts}.log")

    logger = logging.getLogger(name)
    logger.handlers.clear()  # Always reset handlers for fresh runs
    formatter = logging.Formatter('%(levelname)s | %(message)s')
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger, log_file

log, log_file = get_logger()
