"""
self-healing-rag/core/logger.py
Structured logging via loguru with consistent format.
"""
import sys
from loguru import logger as _logger
from core.config import get_settings


def setup_logger() -> None:
    settings = get_settings()
    _logger.remove()
    _logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    _logger.add(
        "data/logs/app.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        serialize=False,
    )


setup_logger()
logger = _logger