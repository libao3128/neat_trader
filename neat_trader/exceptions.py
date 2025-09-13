"""
Custom exceptions for NEAT Trader package.

This module defines custom exception classes for better error handling
and debugging throughout the package.
"""


class NEATTraderError(Exception):
    """Base exception class for NEAT Trader package."""
    pass


class ConfigurationError(NEATTraderError):
    """Raised when there's an issue with configuration."""
    pass


class DataError(NEATTraderError):
    """Raised when there's an issue with data handling."""
    pass


class DatabaseError(DataError):
    """Raised when there's an issue with database operations."""
    pass


class InsufficientDataError(DataError):
    """Raised when there's insufficient data for the requested operation."""
    pass


class EvaluationError(NEATTraderError):
    """Raised when there's an issue during genome evaluation."""
    pass


class BacktestError(EvaluationError):
    """Raised when there's an issue during backtesting."""
    pass


class FitnessError(NEATTraderError):
    """Raised when there's an issue with fitness calculation."""
    pass


class VisualizationError(NEATTraderError):
    """Raised when there's an issue with visualization."""
    pass


class CheckpointError(NEATTraderError):
    """Raised when there's an issue with checkpoint operations."""
    pass
