"""Structured logging setup shared across the application.

Uses structlog to emit JSON-friendly structured logs. Never logs secret
values - callers must pass already-redacted data (see `config.Settings.redacted`).
"""

from __future__ import annotations

import logging
import sys

import structlog

_REDACT_KEYS = {"api_key", "token", "password", "secret", "authorization"}


def _redact_processor(_logger: object, _method_name: str, event_dict: dict) -> dict:
    for key in list(event_dict.keys()):
        if any(token in key.lower() for token in _REDACT_KEYS):
            event_dict[key] = "***REDACTED***"
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog + stdlib logging once for the whole process."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _redact_processor,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structured logger bound to `name`."""
    return structlog.get_logger(name)
