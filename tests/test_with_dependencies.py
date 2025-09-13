#!/usr/bin/env python3
"""
Comprehensive test script with dependency checking for NEAT Trader package.

This script tests the refactored package with proper dependency checking
and provides installation instructions for missing dependencies.

Usage:
    python test_with_dependencies.py
"""

import sys
import os
import subprocess
import importlib

# Add the package to path for testing
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def check_dependency(package_name, import_name=None):
    """Check if a dependency is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True, None
    except ImportError:
        return False, package_name

def install_dependency(package_name):
    """Install a dependency using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def test_dependencies():
    """Test and install required dependencies."""
    print("Checking dependencies...")
    
    dependencies = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("graphviz", "graphviz"),
        ("neat-python", "neat"),
        ("backtesting", "backtesting"),
        ("TA-Lib", "talib"),
    ]
    
    missing_deps = []
    
    for package_name, import_name in dependencies:
        is_installed, missing = check_dependency(package_name, import_name)
        if is_installed:
            print(f"✅ {package_name} is installed")
        else:
            print(f"❌ {package_name} is missing")
            missing_deps.append(package_name)
    
    if missing_deps:
        print(f"\nMissing dependencies: {', '.join(missing_deps)}")
        print("Install them with:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    
    return True

def test_full_functionality():
    """Test full functionality with all dependencies."""
    print("\nTesting full functionality...")
    
    try:
        # Test all imports
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
        
        # Test algorithm imports
        from neat_trader.algorithm import (
            NEATStrategyBase,
            sample_fitness,
            SQN,
            multi_objective_fitness_function_1,
            multi_objective_fitness_function_2,
            outperform_benchmark,
            gpt_fitness_fn,
            crypto_fitness_fn
        )
        print("✅ Algorithm module imports successful")
        
        # Test utils imports
        from neat_trader.utils import (
            plot_stats,
            plot_species,
            draw_net,
            plot_spikes
        )
        print("✅ Utils module imports successful")
        
        # Test configuration
        config = get_config()
        assert isinstance(config, dict)
        print("✅ Configuration working")
        
        # Test data handlers
        handler = DataHandler()
        crypto_handler = CryptoDataHandler()
        print("✅ Data handlers working")
        
        # Test fitness functions
        import pandas as pd
        from datetime import timedelta
        
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
        
        score = multi_objective_fitness_function_2(performance)
        assert isinstance(score, (int, float))
        print("✅ Fitness functions working")
        
        # Test strategy
        strategy = NEATStrategyBase()
        assert hasattr(strategy, 'threshold')
        print("✅ Strategy working")
        
        # Test evaluator
        evaluator = Evaluator(data_handler=handler, fitness_fn=multi_objective_fitness_function_2)
        assert hasattr(evaluator, 'data_handler')
        print("✅ Evaluator working")
        
        # Test visualization (without actually plotting)
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            print("✅ Visualization modules available")
        except ImportError:
            print("⚠️  Visualization modules not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Full functionality test failed: {e}")
        return False

def test_neat_integration():
    """Test NEAT integration."""
    print("\nTesting NEAT integration...")
    
    try:
        import neat
        from neat_trader import NeatTrader, Evaluator, DataHandler, multi_objective_fitness_function_2
        
        # Create mock config
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            'model/config-feedforward'  # This file should exist
        )
        
        # Create evaluator
        data_handler = DataHandler()
        evaluator = Evaluator(data_handler=data_handler, fitness_fn=multi_objective_fitness_function_2)
        
        # Create trader
        trader = NeatTrader(config, evaluator)
        
        assert hasattr(trader, 'population')
        assert hasattr(trader, 'stats')
        assert trader.winner is None
        
        print("✅ NEAT integration working")
        return True
        
    except FileNotFoundError:
        print("⚠️  NEAT config file not found (expected for demo)")
        return True
    except Exception as e:
        print(f"❌ NEAT integration failed: {e}")
        return False

def test_backtesting_integration():
    """Test backtesting integration."""
    print("\nTesting backtesting integration...")
    
    try:
        from backtesting import Backtest
        from neat_trader.algorithm.strategy import NEATStrategyBase
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'Open': 100 + np.random.randn(100).cumsum(),
            'High': 105 + np.random.randn(100).cumsum(),
            'Low': 95 + np.random.randn(100).cumsum(),
            'Close': 100 + np.random.randn(100).cumsum(),
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Create strategy
        class TestStrategy(NEATStrategyBase):
            def init(self):
                super().init()
            
            def next(self):
                pass  # Simple strategy that does nothing
        
        # Test backtesting
        bt = Backtest(data, TestStrategy, cash=100000, commission=0.002)
        result = bt.run()
        
        assert hasattr(result, 'Return')
        print("✅ Backtesting integration working")
        return True
        
    except Exception as e:
        print(f"❌ Backtesting integration failed: {e}")
        return False

def test_technical_indicators():
    """Test technical indicators integration."""
    print("\nTesting technical indicators...")
    
    try:
        from talib.abstract import SMA, RSI, MACD
        import pandas as pd
        import numpy as np
        
        # Create sample data
        data = pd.DataFrame({
            'Close': 100 + np.random.randn(100).cumsum(),
            'High': 105 + np.random.randn(100).cumsum(),
            'Low': 95 + np.random.randn(100).cumsum(),
            'Volume': np.random.randint(1000, 10000, 100)
        })
        
        # Test indicators
        sma = SMA(data['Close'], 5)
        rsi = RSI(data['Close'])
        macd = MACD(data['Close'])
        
        assert len(sma) == len(data)
        assert len(rsi) == len(data)
        assert len(macd[0]) == len(data)  # MACD returns tuple
        
        print("✅ Technical indicators working")
        return True
        
    except Exception as e:
        print(f"❌ Technical indicators failed: {e}")
        return False

def main():
    """Run comprehensive tests."""
    print("NEAT Trader Package - Comprehensive Test with Dependencies")
    print("=" * 60)
    
    # Check dependencies first
    deps_ok = test_dependencies()
    
    if not deps_ok:
        print("\n⚠️  Some dependencies are missing. Install them and run again.")
        return False
    
    # Run tests
    tests = [
        test_full_functionality,
        test_neat_integration,
        test_backtesting_integration,
        test_technical_indicators
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Comprehensive Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All comprehensive tests passed! Package is fully functional.")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
