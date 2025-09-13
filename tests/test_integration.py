"""
Test integration between components.

This module tests that all components work together
correctly and integration scenarios.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add the package to path for testing
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestIntegration:
    """Test integration between components."""
    
    def test_package_integration(self):
        """Test that all components work together."""
        from neat_trader import (
            NeatTrader, 
            Evaluator, 
            DataHandler, 
            multi_objective_fitness_function_2,
            get_config
        )
        
        # Test that we can create all components
        config_dict = get_config()
        data_handler = DataHandler()
        evaluator = Evaluator(
            data_handler=data_handler,
            fitness_fn=multi_objective_fitness_function_2
        )
        
        # Mock NEAT config
        mock_config = Mock()
        
        # Test NeatTrader initialization
        trader = NeatTrader(mock_config, evaluator)
        
        assert trader.config is not None
        assert trader.evaluator is not None
        assert trader.population is not None
        assert trader.stats is not None
        assert trader.winner is None  # Should be None initially
    
    def test_data_handler_evaluator_integration(self, temp_database):
        """Test DataHandler and Evaluator integration."""
        from neat_trader import Evaluator, DataHandler, multi_objective_fitness_function_2
        
        # Create components
        data_handler = DataHandler(temp_database)
        evaluator = Evaluator(
            data_handler=data_handler,
            fitness_fn=multi_objective_fitness_function_2
        )
        
        # Test that evaluator uses the data handler
        assert evaluator.data_handler == data_handler
        
        # Test that evaluator can get data
        data = evaluator.data_handler.get_random_data(data_length=2)
        assert data is not None
        assert len(data) > 0
    
    def test_fitness_function_integration(self, sample_performance_data):
        """Test fitness function integration with evaluator."""
        from neat_trader import Evaluator, DataHandler
        from neat_trader.algorithm.fitness_fn import (
            sample_fitness,
            multi_objective_fitness_function_1,
            multi_objective_fitness_function_2
        )
        
        data_handler = DataHandler()
        
        # Test with different fitness functions
        fitness_functions = [
            sample_fitness,
            multi_objective_fitness_function_1,
            multi_objective_fitness_function_2
        ]
        
        for fitness_fn in fitness_functions:
            evaluator = Evaluator(
                data_handler=data_handler,
                fitness_fn=fitness_fn
            )
            
            # Mock genome evaluation
            mock_genome = Mock()
            mock_config = Mock()
            evaluator.backtest = Mock(return_value=(sample_performance_data, Mock()))
            
            score = evaluator.eval_genome(mock_genome, mock_config)
            assert isinstance(score, (int, float))
    
    def test_strategy_evaluator_integration(self):
        """Test strategy and evaluator integration."""
        from neat_trader.algorithm.evaluate import Evaluator
        from neat_trader.algorithm.strategy import NEATStrategyBase
        from neat_trader.utils.data_handler import DataHandler
        
        # Create evaluator
        data_handler = DataHandler()
        evaluator = Evaluator(data_handler=data_handler)
        
        # Test that evaluator has strategy base class
        assert evaluator.StrategyBaseClass == NEATStrategyBase
        
        # Test strategy class creation
        mock_model = Mock()
        strategy_class = evaluator._create_strategy_class(mock_model)
        
        # Test that the strategy class is created correctly
        assert strategy_class is not None
        assert issubclass(strategy_class, NEATStrategyBase)
        
        # Test that the strategy class has the required methods
        assert hasattr(strategy_class, '__init__')
        assert hasattr(strategy_class, 'init')
        assert hasattr(strategy_class, 'next')
        assert hasattr(strategy_class, 'data_preprocessed')
        
        # Test that the strategy class can be instantiated (without calling init)
        # We'll just test that the class structure is correct
        assert strategy_class.__name__ == 'CustomNEATStrategy'
        
        # Test that the evaluator can create multiple strategy classes
        mock_model2 = Mock()
        strategy_class2 = evaluator._create_strategy_class(mock_model2)
        assert strategy_class2 is not None
        assert strategy_class2 != strategy_class  # Should be different classes
    
    def test_configuration_integration(self):
        """Test configuration integration across components."""
        from neat_trader import get_config, NODE_NAMES
        from neat_trader.config import FITNESS_WEIGHTS, RISK_THRESHOLDS
        from neat_trader.algorithm.fitness_fn import multi_objective_fitness_function_2
        from neat_trader.utils.data_handler import DataHandler
        
        # Get configuration
        config = get_config()
        
        # Test that configuration is used consistently
        assert config['node_names'] == NODE_NAMES
        assert config['fitness'] == FITNESS_WEIGHTS
        assert config['risk'] == RISK_THRESHOLDS
        
        # Test that components can access configuration
        data_handler = DataHandler()
        assert data_handler.db_path == config['database']['default_path']
    
    def test_error_handling_integration(self, temp_test_dir):
        """Test error handling integration."""
        from neat_trader import DataHandler
        from neat_trader.exceptions import DataError, DatabaseError
        
        # Test data handler error handling
        nonexistent_db_path = os.path.join(temp_test_dir, 'nonexistent.db')
        handler = DataHandler(nonexistent_db_path)
        
        with pytest.raises(DatabaseError):
            handler.get_random_data()
        
        # Test that errors propagate correctly
        try:
            handler.get_random_data()
        except DatabaseError as e:
            assert isinstance(e, DataError)  # Should also be DataError
            assert isinstance(e, Exception)  # Should also be Exception
    
    def test_type_hints_integration(self):
        """Test type hints integration across components."""
        import inspect
        from neat_trader import (
            NeatTrader, 
            Evaluator, 
            DataHandler, 
            multi_objective_fitness_function_2
        )
        
        # Test function signatures
        sig1 = inspect.signature(multi_objective_fitness_function_2)
        assert sig1.return_annotation != inspect.Signature.empty
        
        sig2 = inspect.signature(DataHandler.__init__)
        assert 'db_path' in sig2.parameters
        
        sig3 = inspect.signature(Evaluator.__init__)
        assert 'data_handler' in sig3.parameters
        assert 'fitness_fn' in sig3.parameters
    
    def test_import_integration(self):
        """Test import integration."""
        # Test that all imports work together
        from neat_trader import (
            NeatTrader, 
            Evaluator, 
            DataHandler, 
            CryptoDataHandler,
            multi_objective_fitness_function_2,
            get_config,
            NODE_NAMES,
            NEATTraderError,
            DataError,
            EvaluationError
        )
        
        # Test that imports don't conflict
        assert NeatTrader is not None
        assert Evaluator is not None
        assert DataHandler is not None
        assert CryptoDataHandler is not None
        assert multi_objective_fitness_function_2 is not None
        assert get_config is not None
        assert NODE_NAMES is not None
        assert NEATTraderError is not None
        assert DataError is not None
        assert EvaluationError is not None
