"""
Test package structure and imports.

This module tests that the refactored package structure is correct
and all imports work as expected.
"""

import pytest
import sys
import os

# Add the package to path for testing
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestPackageImports:
    """Test package import functionality."""
    
    def test_main_package_imports(self):
        """Test main package imports."""
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
        assert True  # If we get here, imports worked
    
    def test_algorithm_module_imports(self):
        """Test algorithm module imports."""
        from neat_trader.algorithm import (
            NeatTrader,
            Evaluator,
            NEATStrategyBase,
            sample_fitness,
            SQN,
            multi_objective_fitness_function_1,
            multi_objective_fitness_function_2,
            outperform_benchmark,
            gpt_fitness_fn,
            crypto_fitness_fn
        )
        assert True
    
    def test_utils_module_imports(self):
        """Test utils module imports."""
        from neat_trader.utils import (
            DataHandler,
            CryptoDataHandler,
            plot_stats,
            plot_species,
            draw_net,
            plot_spikes
        )
        assert True
    
    def test_config_module_imports(self):
        """Test config module imports."""
        from neat_trader.config import (
            DEFAULT_DB_PATH,
            DEFAULT_CRYPTO_DB_PATH,
            DEFAULT_TRADING_PERIOD_LENGTH,
            DEFAULT_TOTAL_GENERATION,
            FITNESS_WEIGHTS,
            RISK_THRESHOLDS,
            NODE_NAMES
        )
        assert True
    
    def test_exceptions_module_imports(self):
        """Test exceptions module imports."""
        from neat_trader.exceptions import (
            NEATTraderError,
            ConfigurationError,
            DataError,
            DatabaseError,
            InsufficientDataError,
            EvaluationError,
            BacktestError,
            FitnessError,
            VisualizationError,
            CheckpointError
        )
        assert True


class TestPackageStructure:
    """Test package structure and metadata."""
    
    def test_package_metadata(self):
        """Test package metadata."""
        import neat_trader
        
        assert hasattr(neat_trader, '__version__')
        assert hasattr(neat_trader, '__author__')
        assert hasattr(neat_trader, '__all__')
        assert neat_trader.__version__ == "1.0.0"
    
    def test_package_exports(self):
        """Test package exports."""
        import neat_trader
        
        expected_exports = [
            'NeatTrader', 'Evaluator', 'NEATStrategyBase',
            'DataHandler', 'CryptoDataHandler',
            'multi_objective_fitness_function_2',
            'get_config', 'NODE_NAMES',
            'NEATTraderError', 'ConfigurationError', 'DataError', 'EvaluationError'
        ]
        
        for export in expected_exports:
            assert export in neat_trader.__all__
    
    def test_type_hints(self):
        """Test that type hints are properly defined."""
        import inspect
        from neat_trader.algorithm.fitness_fn import sample_fitness, multi_objective_fitness_function_2
        from neat_trader.utils.data_handler import DataHandler
        
        # Test function signatures
        sig1 = inspect.signature(sample_fitness)
        assert 'performance' in sig1.parameters
        assert sig1.return_annotation != inspect.Signature.empty
        
        sig2 = inspect.signature(multi_objective_fitness_function_2)
        assert 'performance' in sig2.parameters
        assert sig2.return_annotation != inspect.Signature.empty
        
        # Test class signatures
        sig3 = inspect.signature(DataHandler.__init__)
        assert 'db_path' in sig3.parameters
