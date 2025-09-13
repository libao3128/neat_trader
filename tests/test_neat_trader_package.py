#!/usr/bin/env python3
"""
Comprehensive test script for the refactored NEAT Trader package.

This script tests all components of the refactored package to ensure:
1. Proper package structure and imports
2. Configuration system functionality
3. Data handling capabilities
4. Fitness function implementations
5. Strategy and evaluation logic
6. Visualization capabilities
7. Error handling
8. Integration between components

Usage:
    python test_neat_trader_package.py
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import sqlite3
from datetime import datetime, timedelta

# Add the package to path for testing
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class TestNEATTraderPackage(unittest.TestCase):
    """Comprehensive test suite for NEAT Trader package."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data = self._create_test_data()
        self.test_performance = self._create_test_performance()
        
    def _create_test_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'Open': 100 + np.random.randn(100).cumsum(),
            'High': 105 + np.random.randn(100).cumsum(),
            'Low': 95 + np.random.randn(100).cumsum(),
            'Close': 100 + np.random.randn(100).cumsum(),
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        return data
    
    def _create_test_performance(self):
        """Create sample backtesting performance data."""
        return pd.Series({
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

class TestPackageImports(TestNEATTraderPackage):
    """Test package import functionality."""
    
    def test_main_package_imports(self):
        """Test main package imports."""
        try:
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
            self.assertTrue(True, "Main package imports successful")
        except ImportError as e:
            self.fail(f"Main package import failed: {e}")
    
    def test_algorithm_module_imports(self):
        """Test algorithm module imports."""
        try:
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
            self.assertTrue(True, "Algorithm module imports successful")
        except ImportError as e:
            self.fail(f"Algorithm module import failed: {e}")
    
    def test_utils_module_imports(self):
        """Test utils module imports."""
        try:
            from neat_trader.utils import (
                DataHandler,
                CryptoDataHandler,
                plot_stats,
                plot_species,
                draw_net,
                plot_spikes
            )
            self.assertTrue(True, "Utils module imports successful")
        except ImportError as e:
            self.fail(f"Utils module import failed: {e}")
    
    def test_config_module_imports(self):
        """Test config module imports."""
        try:
            from neat_trader.config import (
                DEFAULT_DB_PATH,
                DEFAULT_CRYPTO_DB_PATH,
                DEFAULT_TRADING_PERIOD_LENGTH,
                DEFAULT_TOTAL_GENERATION,
                FITNESS_WEIGHTS,
                RISK_THRESHOLDS,
                NODE_NAMES
            )
            self.assertTrue(True, "Config module imports successful")
        except ImportError as e:
            self.fail(f"Config module import failed: {e}")
    
    def test_exceptions_module_imports(self):
        """Test exceptions module imports."""
        try:
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
            self.assertTrue(True, "Exceptions module imports successful")
        except ImportError as e:
            self.fail(f"Exceptions module import failed: {e}")

class TestConfiguration(TestNEATTraderPackage):
    """Test configuration system."""
    
    def test_get_config(self):
        """Test configuration retrieval."""
        from neat_trader.config import get_config
        
        config = get_config()
        self.assertIsInstance(config, dict)
        self.assertIn('database', config)
        self.assertIn('trading', config)
        self.assertIn('strategy', config)
        self.assertIn('visualization', config)
        self.assertIn('fitness', config)
        self.assertIn('risk', config)
        self.assertIn('node_names', config)
    
    def test_node_names(self):
        """Test node names mapping."""
        from neat_trader.config import NODE_NAMES
        
        self.assertIsInstance(NODE_NAMES, dict)
        self.assertGreater(len(NODE_NAMES), 0)
        
        # Check for expected keys
        expected_keys = [-1, -2, -3, -4, -5, 0, 1, 2]
        for key in expected_keys:
            self.assertIn(key, NODE_NAMES)
    
    def test_fitness_weights(self):
        """Test fitness weights configuration."""
        from neat_trader.config import FITNESS_WEIGHTS
        
        self.assertIsInstance(FITNESS_WEIGHTS, dict)
        self.assertIn('relative_ror_weight', FITNESS_WEIGHTS)
        self.assertIn('max_drawdown_weight', FITNESS_WEIGHTS)
        self.assertIn('trade_freq_weight', FITNESS_WEIGHTS)
        
        # Check that weights are numeric
        for key, value in FITNESS_WEIGHTS.items():
            self.assertIsInstance(value, (int, float))

class TestDataHandlers(TestNEATTraderPackage):
    """Test data handling functionality."""
    
    def test_data_handler_initialization(self):
        """Test DataHandler initialization."""
        from neat_trader.utils.data_handler import DataHandler
        
        # Test with default path
        handler = DataHandler()
        self.assertEqual(handler.db_path, 'data/mydatabase.db')
        
        # Test with custom path
        test_db_path = os.path.join(tempfile.gettempdir(), 'test.db')
        custom_handler = DataHandler(test_db_path)
        self.assertEqual(custom_handler.db_path, test_db_path)
    
    def test_crypto_data_handler_initialization(self):
        """Test CryptoDataHandler initialization."""
        from neat_trader.utils.data_handler import CryptoDataHandler, DataHandler
        
        handler = CryptoDataHandler()
        self.assertIsInstance(handler, DataHandler)
        self.assertEqual(handler.db_path, r'data\binance.db')
        
        # Test with custom path
        crypto_test_db_path = os.path.join(tempfile.gettempdir(), 'crypto_test.db')
        custom_handler = CryptoDataHandler(crypto_test_db_path)
        self.assertEqual(custom_handler.db_path, crypto_test_db_path)
    
    def test_data_handler_with_mock_database(self):
        """Test DataHandler with mock database."""
        from neat_trader.utils.data_handler import DataHandler
        from neat_trader.exceptions import DatabaseError, InsufficientDataError
        
        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, 'mock_test_database.db')
        
        try:
            # Create test database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create Price table
            cursor.execute('''
                CREATE TABLE Price (
                    Datetime TEXT,
                    Ticker TEXT,
                    Open REAL,
                    High REAL,
                    Low REAL,
                    Close REAL,
                    Volume INTEGER
                )
            ''')
            
            # Insert test data
            test_data = [
                ('2023-01-01', 'TEST', 100.0, 105.0, 95.0, 102.0, 1000),
                ('2023-01-02', 'TEST', 102.0, 107.0, 97.0, 104.0, 1100),
                ('2023-01-03', 'TEST', 104.0, 109.0, 99.0, 106.0, 1200),
            ]
            cursor.executemany('INSERT INTO Price VALUES (?, ?, ?, ?, ?, ?, ?)', test_data)
            conn.commit()
            conn.close()
            
            # Test DataHandler
            handler = DataHandler(db_path)
            
            # Test data retrieval
            data = handler.get_random_data(data_length=2)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertEqual(len(data), 2)
            self.assertIn('Close', data.columns)
        finally:
            # Clean up
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

class TestFitnessFunctions(TestNEATTraderPackage):
    """Test fitness functions."""
    
    def test_sample_fitness(self):
        """Test sample fitness function."""
        from neat_trader.algorithm.fitness_fn import sample_fitness
        
        score = sample_fitness(self.test_performance)
        self.assertEqual(score, 0.0)
    
    def test_sqn_fitness(self):
        """Test SQN fitness function."""
        from neat_trader.algorithm.fitness_fn import SQN
        
        score = SQN(self.test_performance)
        self.assertEqual(score, 2.1)
    
    def test_multi_objective_fitness_function_1(self):
        """Test multi-objective fitness function 1."""
        from neat_trader.algorithm.fitness_fn import multi_objective_fitness_function_1
        
        score = multi_objective_fitness_function_1(self.test_performance)
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)  # Should be non-negative for this test data
    
    def test_multi_objective_fitness_function_2(self):
        """Test multi-objective fitness function 2."""
        from neat_trader.algorithm.fitness_fn import multi_objective_fitness_function_2
        
        score = multi_objective_fitness_function_2(self.test_performance)
        self.assertIsInstance(score, (int, float))
    
    def test_outperform_benchmark(self):
        """Test outperform benchmark fitness function."""
        from neat_trader.algorithm.fitness_fn import outperform_benchmark
        
        score = outperform_benchmark(self.test_performance)
        self.assertIsInstance(score, (int, float))
    
    def test_gpt_fitness_fn(self):
        """Test GPT fitness function."""
        from neat_trader.algorithm.fitness_fn import gpt_fitness_fn
        
        score = gpt_fitness_fn(self.test_performance)
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)
    
    def test_crypto_fitness_fn(self):
        """Test crypto fitness function."""
        from neat_trader.algorithm.fitness_fn import crypto_fitness_fn
        
        score = crypto_fitness_fn(self.test_performance)
        self.assertIsInstance(score, (int, float))

class TestStrategy(TestNEATTraderPackage):
    """Test trading strategy."""
    
    def test_strategy_initialization(self):
        """Test NEATStrategyBase initialization."""
        from neat_trader.algorithm.strategy import NEATStrategyBase
        
        # Test class attributes without instantiation
        self.assertEqual(NEATStrategyBase.n1, 5)
        self.assertEqual(NEATStrategyBase.n2, 12)
        self.assertEqual(NEATStrategyBase.n3, 26)
        self.assertEqual(NEATStrategyBase.threshold, 0.5)
        self.assertIsNone(NEATStrategyBase.model)
        
        # Test that the class has required methods
        self.assertTrue(hasattr(NEATStrategyBase, 'init'))
        self.assertTrue(hasattr(NEATStrategyBase, 'next'))
        self.assertTrue(hasattr(NEATStrategyBase, 'data_preprocessed'))
    
    def test_data_preprocessed(self):
        """Test data preprocessing."""
        from neat_trader.algorithm.strategy import NEATStrategyBase
        
        # Test that the data_preprocessed method exists and is callable
        self.assertTrue(hasattr(NEATStrategyBase, 'data_preprocessed'))
        self.assertTrue(callable(getattr(NEATStrategyBase, 'data_preprocessed')))
        
        # Test method signature
        import inspect
        from typing import Tuple
        sig = inspect.signature(NEATStrategyBase.data_preprocessed)
        self.assertEqual(len(sig.parameters), 1)  # Only 'self' parameter
        self.assertEqual(sig.return_annotation, Tuple[float, ...])

class TestEvaluator(TestNEATTraderPackage):
    """Test evaluator functionality."""
    
    def test_evaluator_initialization(self):
        """Test Evaluator initialization."""
        from neat_trader.algorithm.evaluate import Evaluator
        from neat_trader.utils.data_handler import DataHandler
        
        # Test with default parameters
        evaluator = Evaluator()
        self.assertIsInstance(evaluator.data_handler, DataHandler)
        self.assertIsNotNone(evaluator.fitness_fn)
        self.assertIsNotNone(evaluator.StrategyBaseClass)
    
    def test_evaluator_with_custom_parameters(self):
        """Test Evaluator with custom parameters."""
        from neat_trader.algorithm.evaluate import Evaluator
        from neat_trader.utils.data_handler import DataHandler
        from neat_trader.algorithm.fitness_fn import sample_fitness
        
        test_db_path = os.path.join(tempfile.gettempdir(), 'test.db')
        data_handler = DataHandler(test_db_path)
        evaluator = Evaluator(
            data_handler=data_handler,
            fitness_fn=sample_fitness
        )
        
        self.assertEqual(evaluator.data_handler.db_path, test_db_path)
        self.assertEqual(evaluator.fitness_fn, sample_fitness)
    
    def test_eval_genome(self):
        """Test genome evaluation."""
        from neat_trader.algorithm.evaluate import Evaluator
        from neat_trader.exceptions import EvaluationError
        
        evaluator = Evaluator()
        
        # Mock genome and config
        mock_genome = Mock()
        mock_config = Mock()
        
        # Mock backtest method
        evaluator.backtest = Mock(return_value=(self.test_performance, Mock()))
        
        # Test evaluation
        score = evaluator.eval_genome(mock_genome, mock_config)
        self.assertIsInstance(score, (int, float))

class TestVisualization(TestNEATTraderPackage):
    """Test visualization functionality."""
    
    def test_plot_stats(self):
        """Test statistics plotting."""
        from neat_trader.utils.visualize import plot_stats
        
        # Mock statistics object
        mock_stats = Mock()
        mock_stats.most_fit_genomes = [Mock(fitness=1.0), Mock(fitness=1.5)]
        mock_stats.get_fitness_mean = Mock(return_value=[1.0, 1.2])
        mock_stats.get_fitness_stdev = Mock(return_value=[0.1, 0.2])
        
        # Mock matplotlib to avoid Tkinter issues
        with patch('neat_trader.utils.visualize.plt') as mock_plt:
            mock_plt.figure.return_value = Mock()
            mock_plt.plot.return_value = Mock()
            mock_plt.title.return_value = Mock()
            mock_plt.xlabel.return_value = Mock()
            mock_plt.ylabel.return_value = Mock()
            mock_plt.grid.return_value = Mock()
            mock_plt.legend.return_value = Mock()
            mock_plt.savefig.return_value = Mock()
            mock_plt.close.return_value = Mock()
            
            # Test plotting (should not raise exception)
            try:
                filename = os.path.join(tempfile.gettempdir(), 'test_stats.svg')
                plot_stats(mock_stats, filename=filename)
                self.assertTrue(True, "Plot stats successful")
            except Exception as e:
                self.fail(f"Plot stats failed: {e}")
    
    def test_plot_species(self):
        """Test species plotting."""
        from neat_trader.utils.visualize import plot_species
        
        # Mock statistics object
        mock_stats = Mock()
        mock_stats.get_species_sizes = Mock(return_value=[[5, 10], [8, 12]])
        
        # Mock matplotlib to avoid Tkinter issues
        with patch('neat_trader.utils.visualize.plt') as mock_plt:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            mock_plt.figure.return_value = Mock()
            mock_plt.plot.return_value = Mock()
            mock_plt.title.return_value = Mock()
            mock_plt.xlabel.return_value = Mock()
            mock_plt.ylabel.return_value = Mock()
            mock_plt.legend.return_value = Mock()
            mock_plt.savefig.return_value = Mock()
            mock_plt.close.return_value = Mock()
            
            # Test plotting
            try:
                filename = os.path.join(tempfile.gettempdir(), 'test_species.svg')
                plot_species(mock_stats, filename=filename)
                self.assertTrue(True, "Plot species successful")
            except Exception as e:
                self.fail(f"Plot species failed: {e}")
    
    def test_draw_net(self):
        """Test network drawing."""
        from neat_trader.utils.visualize import draw_net
        
        # Mock config and genome
        mock_config = Mock()
        mock_config.genome_config.input_keys = [-1, -2]
        mock_config.genome_config.output_keys = [0, 1]
        
        mock_genome = Mock()
        mock_genome.nodes = {0: Mock(), 1: Mock()}
        mock_genome.connections = {}
        
        # Test network drawing
        try:
            filename = os.path.join(tempfile.gettempdir(), 'test_net')
            result = draw_net(mock_config, mock_genome, filename=filename)
            self.assertTrue(True, "Draw net successful")
        except Exception as e:
            self.fail(f"Draw net failed: {e}")

class TestErrorHandling(TestNEATTraderPackage):
    """Test error handling."""
    
    def test_custom_exceptions(self):
        """Test custom exception hierarchy."""
        from neat_trader.exceptions import (
            NEATTraderError,
            DataError,
            DatabaseError,
            InsufficientDataError,
            EvaluationError,
            BacktestError,
            FitnessError,
            VisualizationError,
            CheckpointError
        )
        
        # Test exception inheritance
        self.assertTrue(issubclass(DataError, NEATTraderError))
        self.assertTrue(issubclass(DatabaseError, DataError))
        self.assertTrue(issubclass(InsufficientDataError, DataError))
        self.assertTrue(issubclass(EvaluationError, NEATTraderError))
        self.assertTrue(issubclass(BacktestError, EvaluationError))
        self.assertTrue(issubclass(FitnessError, NEATTraderError))
        self.assertTrue(issubclass(VisualizationError, NEATTraderError))
        self.assertTrue(issubclass(CheckpointError, NEATTraderError))
    
    def test_exception_raising(self):
        """Test exception raising."""
        from neat_trader.exceptions import DataError, DatabaseError
        
        # Test DataError
        with self.assertRaises(DataError):
            raise DataError("Test data error")
        
        # Test DatabaseError
        with self.assertRaises(DatabaseError):
            raise DatabaseError("Test database error")

class TestIntegration(TestNEATTraderPackage):
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
        
        self.assertIsNotNone(trader.config)
        self.assertIsNotNone(trader.evaluator)
        self.assertIsNotNone(trader.population)
        self.assertIsNotNone(trader.stats)
        self.assertIsNone(trader.winner)  # Should be None initially

def run_tests():
    """Run all tests and display results."""
    print("=" * 60)
    print("NEAT Trader Package Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_classes = [
        TestPackageImports,
        TestConfiguration,
        TestDataHandlers,
        TestFitnessFunctions,
        TestStrategy,
        TestEvaluator,
        TestVisualization,
        TestErrorHandling,
        TestIntegration
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
