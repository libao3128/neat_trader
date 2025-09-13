"""
Test fitness functions.

This module tests all fitness functions with various
performance data scenarios.
"""

import pytest
import sys
import os
import pandas as pd
from datetime import timedelta

# Add the package to path for testing
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestFitnessFunctions:
    """Test fitness functions."""
    
    def test_sample_fitness(self, sample_performance_data):
        """Test sample fitness function."""
        from neat_trader.algorithm.fitness_fn import sample_fitness
        
        score = sample_fitness(sample_performance_data)
        assert isinstance(score, float)
        assert score == 0.0
    
    def test_sqn_fitness(self, sample_performance_data):
        """Test SQN fitness function."""
        from neat_trader.algorithm.fitness_fn import SQN
        
        score = SQN(sample_performance_data)
        assert isinstance(score, (int, float))
        assert score == 2.1
    
    def test_multi_objective_fitness_function_1(self, sample_performance_data):
        """Test multi-objective fitness function 1."""
        from neat_trader.algorithm.fitness_fn import multi_objective_fitness_function_1
        
        score = multi_objective_fitness_function_1(sample_performance_data)
        assert isinstance(score, (int, float))
        assert score >= 0  # Should be non-negative for this test data
    
    def test_multi_objective_fitness_function_2(self, sample_performance_data):
        """Test multi-objective fitness function 2."""
        from neat_trader.algorithm.fitness_fn import multi_objective_fitness_function_2
        
        score = multi_objective_fitness_function_2(sample_performance_data)
        assert isinstance(score, (int, float))
    
    def test_outperform_benchmark(self, sample_performance_data):
        """Test outperform benchmark fitness function."""
        from neat_trader.algorithm.fitness_fn import outperform_benchmark
        
        score = outperform_benchmark(sample_performance_data)
        assert isinstance(score, (int, float))
    
    def test_gpt_fitness_fn(self, sample_performance_data):
        """Test GPT fitness function."""
        from neat_trader.algorithm.fitness_fn import gpt_fitness_fn
        
        score = gpt_fitness_fn(sample_performance_data)
        assert isinstance(score, (int, float))
        assert score >= 0
    
    def test_crypto_fitness_fn(self, sample_performance_data):
        """Test crypto fitness function."""
        from neat_trader.algorithm.fitness_fn import crypto_fitness_fn
        
        score = crypto_fitness_fn(sample_performance_data)
        assert isinstance(score, (int, float))
    
    def test_fitness_functions_with_edge_cases(self):
        """Test fitness functions with edge cases."""
        from neat_trader.algorithm.fitness_fn import (
            multi_objective_fitness_function_2,
            gpt_fitness_fn
        )
        
        # Test with zero trades
        zero_trades_data = pd.Series({
            'Return [%]': 0.0,
            'Buy & Hold Return [%]': 0.0,
            'Max. Drawdown [%]': 0.0,
            '# Trades': 0,
            'Avg. Trade Duration': timedelta(days=0),
            'Sharpe Ratio': 0.0,
            'Return (Ann.) [%]': 0.0,
            'Duration': timedelta(days=365),
            'SQN': 0.0
        })
        
        score1 = multi_objective_fitness_function_2(zero_trades_data)
        assert isinstance(score1, (int, float))
        
        score2 = gpt_fitness_fn(zero_trades_data)
        assert isinstance(score2, (int, float))
        assert score2 == 0.0  # Should return 0 for zero trades
    
    def test_fitness_functions_with_none_values(self):
        """Test fitness functions with None values."""
        from neat_trader.algorithm.fitness_fn import multi_objective_fitness_function_2
        
        # Test with None drawdown
        none_drawdown_data = pd.Series({
            'Return [%]': 10.0,
            'Buy & Hold Return [%]': 8.0,
            'Max. Drawdown [%]': None,
            '# Trades': 5,
            'Avg. Trade Duration': timedelta(days=3),
            'Sharpe Ratio': 1.0,
            'Return (Ann.) [%]': 12.0,
            'Duration': timedelta(days=365),
            'SQN': 1.5
        })
        
        score = multi_objective_fitness_function_2(none_drawdown_data)
        assert isinstance(score, (int, float))
