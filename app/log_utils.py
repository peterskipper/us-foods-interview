"""Logging utilities"""

import logging
import os


def _instantiate_log(log_name: str):
    """Boiler plate to configure a new logger"""
    log = logging.getLogger(log_name)
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(os.environ.get("LOG_LEVEL", logging.INFO))
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log
