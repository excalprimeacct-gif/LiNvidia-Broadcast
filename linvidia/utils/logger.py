"""Logging configuration"""

import sys
from loguru import logger


def setup_logger(
    level: str = "INFO",
    log_file: str = None,
    colorize: bool = True
):
    """
    Setup logger configuration

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
        colorize: Enable colored output
    """
    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
        colorize=colorize
    )

    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
            level=level,
            rotation="10 MB",
            compression="zip"
        )

    return logger
