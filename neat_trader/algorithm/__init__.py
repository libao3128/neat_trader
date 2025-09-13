"""
Algorithm module for NEAT Trader package.

This module contains the core NEAT algorithm implementation, evaluation logic,
fitness functions, and trading strategies.
"""

from .neat_trader import NeatTrader
from .evaluate import Evaluator
from .fitness_fn import (
    sample_fitness,
    SQN,
    multi_objective_fitness_function_1,
    multi_objective_fitness_function_2,
    outperform_benchmark,
    gpt_fitness_fn,
    crypto_fitness_fn
)
from .strategy import NEATStrategyBase

__all__ = [
    'NeatTrader',
    'Evaluator',
    'NEATStrategyBase',
    'sample_fitness',
    'SQN',
    'multi_objective_fitness_function_1',
    'multi_objective_fitness_function_2',
    'outperform_benchmark',
    'gpt_fitness_fn',
    'crypto_fitness_fn'
]