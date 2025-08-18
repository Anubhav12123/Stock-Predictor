from __future__ import annotations
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level="INFO", serialize=True)

__all__ = ["logger"]
