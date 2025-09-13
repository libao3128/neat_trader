#!/usr/bin/env python3
"""
Quick test script for NEAT Trader package.

This script performs basic functionality tests to ensure the refactored
package is working correctly. It's designed to run quickly and provide
immediate feedback.

Usage:
    python quick_test.py
"""

import sys
import os

# Add the package to path for testing
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        # Test main package imports
        from neat_trader import (
            NeatTrader, 
            Evaluator, 
            DataHandler, 
            CryptoDataHandler,
            multi_objective_fitness_function_2,
            get_config,
            NODE_NAMES
        )
        print("✅ Main package imports successful")
        
        # Test submodule imports
        from neat_trader.algorithm import (
            NEATStrategyBase,
            sample_fitness,
            SQN,
            multi_objective_fitness_function_1
        )
        print("✅ Algorithm module imports successful")
        
        from neat_trader.utils import (
            plot_stats,
            plot_species,
            draw_net
        )
        print("✅ Utils module imports successful")
        
        from neat_trader.config import (
            DEFAULT_DB_PATH,
            FITNESS_WEIGHTS,
            RISK_THRESHOLDS
        )
        print("✅ Config module imports successful")
        
        from neat_trader.exceptions import (
            NEATTraderError,
            DataError,
            EvaluationError
        )
        print("✅ Exceptions module imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration...")
    
    try:
        from neat_trader import get_config, NODE_NAMES
        
        # Test get_config
        config = get_config()
        assert isinstance(config, dict), "Config should be a dictionary"
        assert 'database' in config, "Config should contain database settings"
        assert 'trading' in config, "Config should contain trading settings"
        assert 'fitness' in config, "Config should contain fitness settings"
        print("✅ Configuration system working")
        
        # Test NODE_NAMES
        assert isinstance(NODE_NAMES, dict), "NODE_NAMES should be a dictionary"
        assert len(NODE_NAMES) > 0, "NODE_NAMES should not be empty"
        print("✅ Node names mapping working")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_data_handlers():
    """Test data handler initialization."""
    print("\nTesting data handlers...")
    
    try:
        from neat_trader import DataHandler, CryptoDataHandler
        
        # Test DataHandler
        handler = DataHandler()
        assert hasattr(handler, 'db_path'), "DataHandler should have db_path attribute"
        print("✅ DataHandler initialization successful")
        
        # Test CryptoDataHandler
        crypto_handler = CryptoDataHandler()
        assert hasattr(crypto_handler, 'db_path'), "CryptoDataHandler should have db_path attribute"
        assert isinstance(crypto_handler, DataHandler), "CryptoDataHandler should inherit from DataHandler"
        print("✅ CryptoDataHandler initialization successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Data handler error: {e}")
        return False

def test_fitness_functions():
    """Test fitness functions with sample data."""
    print("\nTesting fitness functions...")
    
    try:
        import pandas as pd
        from datetime import timedelta
        from neat_trader.algorithm.fitness_fn import (
            sample_fitness,
            SQN,
            multi_objective_fitness_function_1,
            multi_objective_fitness_function_2
        )
        
        # Create sample performance data
        performance = pd.Series({
            'Return [%]': 15.5,
            'Buy & Hold Return [%]': 12.3,
            'Max. Drawdown [%]': 8.2,
            '# Trades': 25,
            'Avg. Trade Duration': timedelta(days=5),
            'Sharpe Ratio': 1.8,
            'Return (Ann.) [%]': 18.2,
            'Duration': timedelta(days=365),
            'SQN': 2.1
        })
        
        # Test fitness functions
        score1 = sample_fitness(performance)
        assert isinstance(score1, (int, float)), "sample_fitness should return numeric score"
        
        score2 = SQN(performance)
        assert isinstance(score2, (int, float)), "SQN should return numeric score"
        
        score3 = multi_objective_fitness_function_1(performance)
        assert isinstance(score3, (int, float)), "multi_objective_fitness_function_1 should return numeric score"
        
        score4 = multi_objective_fitness_function_2(performance)
        assert isinstance(score4, (int, float)), "multi_objective_fitness_function_2 should return numeric score"
        
        print("✅ All fitness functions working")
        return True
        
    except Exception as e:
        print(f"❌ Fitness function error: {e}")
        return False

def test_strategy():
    """Test strategy initialization."""
    print("\nTesting strategy...")
    
    try:
        from neat_trader.algorithm.strategy import NEATStrategyBase
        
        # Test strategy initialization
        strategy = NEATStrategyBase()
        assert hasattr(strategy, 'n1'), "Strategy should have n1 attribute"
        assert hasattr(strategy, 'n2'), "Strategy should have n2 attribute"
        assert hasattr(strategy, 'n3'), "Strategy should have n3 attribute"
        assert hasattr(strategy, 'threshold'), "Strategy should have threshold attribute"
        assert hasattr(strategy, 'model'), "Strategy should have model attribute"
        
        print("✅ Strategy initialization successful")
        return True
        
    except Exception as e:
        print(f"❌ Strategy error: {e}")
        return False

def test_evaluator():
    """Test evaluator initialization."""
    print("\nTesting evaluator...")
    
    try:
        from neat_trader import Evaluator, DataHandler, multi_objective_fitness_function_2
        
        # Test evaluator initialization
        data_handler = DataHandler()
        evaluator = Evaluator(
            data_handler=data_handler,
            fitness_fn=multi_objective_fitness_function_2
        )
        
        assert hasattr(evaluator, 'data_handler'), "Evaluator should have data_handler attribute"
        assert hasattr(evaluator, 'fitness_fn'), "Evaluator should have fitness_fn attribute"
        assert hasattr(evaluator, 'StrategyBaseClass'), "Evaluator should have StrategyBaseClass attribute"
        
        print("✅ Evaluator initialization successful")
        return True
        
    except Exception as e:
        print(f"❌ Evaluator error: {e}")
        return False

def test_exceptions():
    """Test custom exceptions."""
    print("\nTesting exceptions...")
    
    try:
        from neat_trader.exceptions import (
            NEATTraderError,
            DataError,
            DatabaseError,
            EvaluationError
        )
        
        # Test exception hierarchy
        assert issubclass(DataError, NEATTraderError), "DataError should inherit from NEATTraderError"
        assert issubclass(DatabaseError, DataError), "DatabaseError should inherit from DataError"
        assert issubclass(EvaluationError, NEATTraderError), "EvaluationError should inherit from NEATTraderError"
        
        # Test exception raising
        try:
            raise DataError("Test error")
        except DataError as e:
            assert str(e) == "Test error", "Exception message should be preserved"
        
        print("✅ Custom exceptions working")
        return True
        
    except Exception as e:
        print(f"❌ Exception error: {e}")
        return False

def test_integration():
    """Test integration between components."""
    print("\nTesting integration...")
    
    try:
        from neat_trader import (
            NeatTrader, 
            Evaluator, 
            DataHandler, 
            multi_objective_fitness_function_2
        )
        
        # Create components
        data_handler = DataHandler()
        evaluator = Evaluator(
            data_handler=data_handler,
            fitness_fn=multi_objective_fitness_function_2
        )
        
        # Mock NEAT config
        class MockConfig:
            pass
        
        mock_config = MockConfig()
        
        # Test NeatTrader initialization
        trader = NeatTrader(mock_config, evaluator)
        
        assert hasattr(trader, 'config'), "NeatTrader should have config attribute"
        assert hasattr(trader, 'evaluator'), "NeatTrader should have evaluator attribute"
        assert hasattr(trader, 'population'), "NeatTrader should have population attribute"
        assert hasattr(trader, 'stats'), "NeatTrader should have stats attribute"
        assert trader.winner is None, "Winner should be None initially"
        
        print("✅ Integration test successful")
        return True
        
    except Exception as e:
        print(f"❌ Integration error: {e}")
        return False

def main():
    """Run all quick tests."""
    print("NEAT Trader Package - Quick Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_configuration,
        test_data_handlers,
        test_fitness_functions,
        test_strategy,
        test_evaluator,
        test_exceptions,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Quick Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Package is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
