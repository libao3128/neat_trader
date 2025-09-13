"""
Test data handling functionality.

This module tests the DataHandler and CryptoDataHandler classes
and their data retrieval capabilities.
"""

import pytest
import sys
import os
import pandas as pd

# Add the package to path for testing
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestDataHandlers:
    """Test data handling functionality."""
    
    def test_data_handler_initialization(self, temp_test_dir):
        """Test DataHandler initialization."""
        from neat_trader.utils.data_handler import DataHandler
        
        # Test with default path
        handler = DataHandler()
        assert handler.db_path == 'data/mydatabase.db'
        
        # Test with custom path
        test_db_path = os.path.join(temp_test_dir, 'test.db')
        custom_handler = DataHandler(test_db_path)
        assert custom_handler.db_path == test_db_path
    
    def test_crypto_data_handler_initialization(self, temp_test_dir):
        """Test CryptoDataHandler initialization."""
        from neat_trader.utils.data_handler import CryptoDataHandler, DataHandler
        
        handler = CryptoDataHandler()
        assert isinstance(handler, DataHandler)
        assert handler.db_path == r'data\binance.db'
        
        # Test with custom path
        crypto_test_db_path = os.path.join(temp_test_dir, 'crypto_test.db')
        custom_handler = CryptoDataHandler(crypto_test_db_path)
        assert custom_handler.db_path == crypto_test_db_path
    
    def test_data_handler_with_temp_database(self, temp_database):
        """Test DataHandler with temporary database."""
        from neat_trader.utils.data_handler import DataHandler
        from neat_trader.exceptions import DatabaseError, InsufficientDataError
        
        handler = DataHandler(temp_database)
        
        # Test data retrieval
        data = handler.get_random_data(data_length=2)
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2
        assert 'Close' in data.columns
        assert 'Open' in data.columns
        assert 'High' in data.columns
        assert 'Low' in data.columns
        assert 'Volume' in data.columns
    
    def test_uniform_sampling(self):
        """Test uniform sampling functionality."""
        from neat_trader.utils.data_handler import DataHandler
        
        handler = DataHandler()
        
        # Test with sufficient data
        dates = pd.Series(pd.date_range('2023-01-01', periods=100, freq='D'))
        start_date, end_date = handler._uniform_sample(dates, 10)
        
        assert start_date is not None
        assert end_date is not None
        assert isinstance(start_date, pd.Timestamp)
        assert isinstance(end_date, pd.Timestamp)
        
        # Test with insufficient data
        short_dates = pd.Series(pd.date_range('2023-01-01', periods=5, freq='D'))
        start_date, end_date = handler._uniform_sample(short_dates, 10)
        
        assert start_date is None
        assert end_date is None
    
    def test_database_connection_error(self, temp_test_dir):
        """Test database connection error handling."""
        from neat_trader.utils.data_handler import DataHandler
        from neat_trader.exceptions import DatabaseError
        
        # Test with non-existent database
        nonexistent_db_path = os.path.join(temp_test_dir, 'nonexistent.db')
        handler = DataHandler(nonexistent_db_path)
        
        with pytest.raises(DatabaseError):
            handler.get_random_data()
    
    def test_crypto_data_handler_methods(self):
        """Test CryptoDataHandler specific methods."""
        from neat_trader.utils.data_handler import CryptoDataHandler
        
        handler = CryptoDataHandler()
        
        # Test that it has the same interface as DataHandler
        assert hasattr(handler, 'get_random_data')
        assert hasattr(handler, '_connect')
        assert hasattr(handler, '_uniform_sample')
