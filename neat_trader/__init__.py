"""
NEAT Trader: A NEAT Algorithm-based Stock Trading Strategy Package.

This package implements a NeuroEvolution of Augmenting Topologies (NEAT) algorithm
for automated stock trading strategies using multiple technical indicators.

Main Components:
- algorithm: Core NEAT implementation, evaluation, and fitness functions
- utils: Data handling and visualization utilities
- config: Configuration and constants
- exceptions: Custom exceptions for error handling

Example:
    >>> from neat_trader import NeatTrader, Evaluator, DataHandler
    >>> from neat_trader.algorithm.fitness_fn import multi_objective_fitness_function_2
    >>> 
    >>> # Initialize components
    >>> data_handler = DataHandler()
    >>> evaluator = Evaluator(data_handler=data_handler, fitness_fn=multi_objective_fitness_function_2)
    >>> trader = NeatTrader(config, evaluator)
    >>> 
    >>> # Run evolution
    >>> winner = trader.evolve(total_generation=100)
    >>> trader.generate_report()
"""

from .algorithm import NeatTrader, Evaluator, NEATStrategyBase
from .utils import DataHandler, CryptoDataHandler
from .algorithm.fitness_fn import multi_objective_fitness_function_2
from .config import get_config, NODE_NAMES
from .exceptions import NEATTraderError, ConfigurationError, DataError, EvaluationError

__version__ = "1.0.0"
__author__ = "NEAT Trader Team"

__all__ = [
    'NeatTrader',
    'Evaluator', 
    'NEATStrategyBase',
    'DataHandler',
    'CryptoDataHandler',
    'multi_objective_fitness_function_2',
    'get_config',
    'NODE_NAMES',
    'NEATTraderError',
    'ConfigurationError',
    'DataError',
    'EvaluationError'
]
