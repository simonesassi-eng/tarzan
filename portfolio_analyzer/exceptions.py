"""Domain-specific exceptions for financial data processing."""

from __future__ import annotations


class PortfolioAnalyzerError(Exception):
    """Base exception for all portfolio analyzer errors."""


class DataIngestionError(PortfolioAnalyzerError):
    """Raised when input data cannot be loaded or parsed."""


class DataEnrichmentError(PortfolioAnalyzerError):
    """Raised when market data enrichment fails for a holding."""

    def __init__(self, ticker: str, message: str):
        self.ticker = ticker
        super().__init__(f"Enrichment failed for {ticker}: {message}")


class InsufficientDataError(PortfolioAnalyzerError):
    """Raised when there is not enough data to compute a metric."""

    def __init__(self, metric: str, required: int, available: int):
        self.metric, self.required, self.available = metric, required, available
        super().__init__(f"Insufficient data for {metric}: need {required}, have {available}")


class MetricCalculationError(PortfolioAnalyzerError):
    """Raised when a metric calculation encounters a numerical error."""

    def __init__(self, metric: str, reason: str):
        self.metric, self.reason = metric, reason
        super().__init__(f"Cannot compute {metric}: {reason}")


class ClassificationError(PortfolioAnalyzerError):
    """Raised when an instrument cannot be classified."""

    def __init__(self, isin: str, reason: str):
        self.isin = isin
        super().__init__(f"Classification failed for {isin}: {reason}")


class ConfigurationError(PortfolioAnalyzerError):
    """Raised when configuration is invalid or missing."""
