"""
Shared logging and error-report helpers for CLI scripts.
"""

import json
import logging
import os
import sys
from typing import Any


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def _to_jsonable(value: Any) -> Any:
    """Convert nested values into JSON-safe structures."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    return str(value)


def setup_logging(name: str | None = None, level: str | int | None = None) -> logging.Logger:
    """
    Configure the root logger once and return a named logger.
    """
    resolved_level = level or os.getenv("RENDI_LOG_LEVEL", "INFO")
    if isinstance(resolved_level, str):
        resolved_level = getattr(logging, resolved_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)

    formatter = logging.Formatter(LOG_FORMAT)
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)

    return logging.getLogger(name)


def build_error_result(
    message: str,
    *,
    error: Exception | None = None,
    stage: str | None = None,
    **context: Any,
) -> dict:
    """
    Build a consistent, JSON-serializable error payload.
    """
    result = {"error": message}
    if stage:
        result["error_stage"] = stage
    if error is not None:
        result["error_type"] = type(error).__name__
    if context:
        result["error_context"] = _to_jsonable(context)
    return result


def log_error_result(logger: logging.Logger, prefix: str, error_result: dict, level: int = logging.ERROR) -> None:
    """
    Log a structured error payload in a readable way.
    """
    stage = error_result.get("error_stage", "unknown")
    error_type = error_result.get("error_type", "UnknownError")
    message = error_result.get("error", "Unknown error")
    logger.log(level, "%s | stage=%s | error_type=%s | error=%s", prefix, stage, error_type, message)

    context = error_result.get("error_context")
    if context:
        logger.log(level, "%s context=%s", prefix, json.dumps(context, ensure_ascii=False, sort_keys=True))
