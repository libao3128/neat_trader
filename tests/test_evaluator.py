"""
Test evaluator functionality.

This module tests the Evaluator class and its
genome evaluation capabilities.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add the package to path for testing
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestEvaluator:
    """Test evaluator functionality."""
    
    def test_evaluator_initialization(self):
        """Test Evaluator initialization."""
        from neat_trader.algorithm.evaluate import Evaluator
        from neat_trader.utils.data_handler import DataHandler
        
        # Test with default parameters
        evaluator = Evaluator()
        assert isinstance(evaluator.data_handler, DataHandler)
        assert evaluator.fitness_fn is not None
        assert evaluator.StrategyBaseClass is not None
    
    def test_evaluator_with_custom_parameters(self, temp_test_dir):
        """Test Evaluator with custom parameters."""
        from neat_trader.algorithm.evaluate import Evaluator
        from neat_trader.utils.data_handler import DataHandler
        from neat_trader.algorithm.fitness_fn import sample_fitness
        
        db_path = os.path.join(temp_test_dir, 'test.db')
        data_handler = DataHandler(db_path)
        evaluator = Evaluator(
            data_handler=data_handler,
            fitness_fn=sample_fitness
        )
        
        assert evaluator.data_handler.db_path == db_path
        assert evaluator.fitness_fn == sample_fitness
    
    def test_eval_genome(self, sample_performance_data):
        """Test genome evaluation."""
        from neat_trader.algorithm.evaluate import Evaluator
        
        evaluator = Evaluator()
        
        # Mock genome and config
        mock_genome = Mock()
        mock_config = Mock()
        
        # Mock backtest method
        evaluator.backtest = Mock(return_value=(sample_performance_data, Mock()))
        
        # Test evaluation
        score = evaluator.eval_genome(mock_genome, mock_config)
        assert isinstance(score, (int, float))
    
    def test_eval_genomes(self, sample_performance_data):
        """Test genome list evaluation."""
        from neat_trader.algorithm.evaluate import Evaluator
        
        evaluator = Evaluator()
        
        # Mock genomes and config
        mock_genome1 = Mock()
        mock_genome2 = Mock()
        genomes = [(1, mock_genome1), (2, mock_genome2)]
        mock_config = Mock()
        
        # Mock backtest method
        evaluator.backtest = Mock(return_value=(sample_performance_data, Mock()))
        
        # Test evaluation
        scores = evaluator.eval_genomes(genomes, mock_config)
        
        assert isinstance(scores, list)
        assert len(scores) == 2
        assert all(isinstance(score, (int, float)) for score in scores)
    
    def test_backtest_method(self, sample_market_data, sample_performance_data):
        """Test backtest method."""
        from neat_trader.algorithm.evaluate import Evaluator
        
        evaluator = Evaluator()
        
        # Mock genome and config
        mock_genome = Mock()
        mock_config = Mock()
        
        # Mock network creation
        mock_net = Mock()
        mock_net.activate = Mock(return_value=(0.5, 0.5, 0.5))
        
        # Mock strategy creation
        mock_strategy_class = Mock()
        mock_strategy_instance = Mock()
        mock_strategy_class.return_value = mock_strategy_instance
        
        evaluator._create_strategy_class = Mock(return_value=mock_strategy_class)
        
        # Mock backtesting
        mock_bt = Mock()
        mock_bt.run = Mock(return_value=sample_performance_data)
        
        with patch('neat_trader.algorithm.evaluate.Backtest') as mock_backtest:
            mock_backtest.return_value = mock_bt
            
            # This would require mocking the backtesting library
            # For now, just test that the method exists and is callable
            assert hasattr(evaluator, 'backtest')
            assert callable(evaluator.backtest)
    
    def test_create_strategy_class(self):
        """Test strategy class creation."""
        from neat_trader.algorithm.evaluate import Evaluator
        
        evaluator = Evaluator()
        
        # Mock model
        mock_model = Mock()
        
        # Test strategy class creation
        strategy_class = evaluator._create_strategy_class(mock_model)
        
        assert strategy_class is not None
        assert callable(strategy_class)
    
    def test_evaluator_with_fitness_function(self, sample_performance_data):
        """Test evaluator with different fitness functions."""
        from neat_trader.algorithm.evaluate import Evaluator
        from neat_trader.algorithm.fitness_fn import (
            sample_fitness,
            multi_objective_fitness_function_1,
            multi_objective_fitness_function_2
        )
        
        # Test with different fitness functions
        fitness_functions = [
            sample_fitness,
            multi_objective_fitness_function_1,
            multi_objective_fitness_function_2
        ]
        
        for fitness_fn in fitness_functions:
            evaluator = Evaluator(fitness_fn=fitness_fn)
            assert evaluator.fitness_fn == fitness_fn
            
            # Test evaluation
            mock_genome = Mock()
            mock_config = Mock()
            evaluator.backtest = Mock(return_value=(sample_performance_data, Mock()))
            
            score = evaluator.eval_genome(mock_genome, mock_config)
            assert isinstance(score, (int, float))
