"""
Pytest configuration and shared fixtures for NEAT Trader tests.

This module provides common test fixtures and configuration
for all test modules in the package.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import sqlite3
import shutil
from unittest.mock import Mock


@pytest.fixture(scope="session")
def temp_test_dir():
    """Create a unified temporary directory for all test files."""
    temp_dir = tempfile.mkdtemp(prefix="neat_trader_test_")
    yield temp_dir
    
    # Clean up the entire directory after all tests
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_market_data():
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


@pytest.fixture
def sample_performance_data():
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


@pytest.fixture
def temp_database(temp_test_dir):
    """Create a temporary database for testing."""
    db_path = os.path.join(temp_test_dir, 'test_database.db')
    
    # Create test database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create Price table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Price (
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
        ('2023-01-04', 'TEST', 106.0, 111.0, 101.0, 108.0, 1300),
        ('2023-01-05', 'TEST', 108.0, 113.0, 103.0, 110.0, 1400),
    ]
    cursor.executemany('INSERT INTO Price VALUES (?, ?, ?, ?, ?, ?, ?)', test_data)
    conn.commit()
    conn.close()
    
    yield db_path


@pytest.fixture
def mock_neat_config():
    """Create a mock NEAT configuration."""
    class MockConfig:
        def __init__(self):
            self.genome_config = Mock()
            self.genome_config.input_keys = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11]
            self.genome_config.output_keys = [0, 1, 2]
    
    return MockConfig()


@pytest.fixture
def mock_genome():
    """Create a mock NEAT genome."""
    class MockGenome:
        def __init__(self):
            self.fitness = 0.0
            self.nodes = {0: Mock(), 1: Mock()}
            self.connections = {}
    
    return MockGenome()


@pytest.fixture
def mock_statistics():
    """Create mock NEAT statistics."""
    class MockStatistics:
        def __init__(self):
            self.most_fit_genomes = [Mock(fitness=1.0), Mock(fitness=1.5)]
        
        def get_fitness_mean(self):
            return [1.0, 1.2]
        
        def get_fitness_stdev(self):
            return [0.1, 0.2]
        
        def get_species_sizes(self):
            return [[5, 10], [8, 12]]
    
    return MockStatistics()
