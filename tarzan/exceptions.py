"""Domain-specific exceptions for financial data processing."""

from __future__ import annotations


class TarzanError(Exception):
    """Base exception for all Tarzan errors."""


class DataIngestionError(TarzanError):
    """Raised when input data cannot be loaded or parsed."""
