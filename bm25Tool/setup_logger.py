"""setup_logger.py"""
import os
import logging
from logging import Logger

from config_reader import get_base_directory


BASE_DIR: str = get_base_directory()
CONFIG_PATH: str = os.path.join(BASE_DIR, "config.ini")


def setup_logger(name: str) -> Logger:
    """Sets up a logger for the module."""
    log_file_name: str = os.path.basename(name)
    log_file_path: str = os.path.join(BASE_DIR, "logs", f"{log_file_name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
