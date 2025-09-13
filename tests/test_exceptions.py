"""
Test custom exceptions.

This module tests the custom exception hierarchy
and error handling functionality.
"""

import pytest
import sys
import os

# Add the package to path for testing
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestExceptions:
    """Test custom exceptions."""
    
    def test_exception_hierarchy(self):
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
        assert issubclass(DataError, NEATTraderError)
        assert issubclass(DatabaseError, DataError)
        assert issubclass(InsufficientDataError, DataError)
        assert issubclass(EvaluationError, NEATTraderError)
        assert issubclass(BacktestError, EvaluationError)
        assert issubclass(FitnessError, NEATTraderError)
        assert issubclass(VisualizationError, NEATTraderError)
        assert issubclass(CheckpointError, NEATTraderError)
    
    def test_exception_raising(self):
        """Test exception raising."""
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
        
        # Test DataError
        with pytest.raises(DataError):
            raise DataError("Test data error")
        
        # Test DatabaseError
        with pytest.raises(DatabaseError):
            raise DatabaseError("Test database error")
        
        # Test InsufficientDataError
        with pytest.raises(InsufficientDataError):
            raise InsufficientDataError("Test insufficient data error")
        
        # Test EvaluationError
        with pytest.raises(EvaluationError):
            raise EvaluationError("Test evaluation error")
        
        # Test BacktestError
        with pytest.raises(BacktestError):
            raise BacktestError("Test backtest error")
        
        # Test FitnessError
        with pytest.raises(FitnessError):
            raise FitnessError("Test fitness error")
        
        # Test VisualizationError
        with pytest.raises(VisualizationError):
            raise VisualizationError("Test visualization error")
        
        # Test CheckpointError
        with pytest.raises(CheckpointError):
            raise CheckpointError("Test checkpoint error")
    
    def test_exception_messages(self):
        """Test exception messages."""
        from neat_trader.exceptions import DataError, DatabaseError
        
        # Test DataError message
        error_msg = "Custom data error message"
        with pytest.raises(DataError) as exc_info:
            raise DataError(error_msg)
        assert str(exc_info.value) == error_msg
        
        # Test DatabaseError message
        db_error_msg = "Custom database error message"
        with pytest.raises(DatabaseError) as exc_info:
            raise DatabaseError(db_error_msg)
        assert str(exc_info.value) == db_error_msg
    
    def test_exception_inheritance_behavior(self):
        """Test exception inheritance behavior."""
        from neat_trader.exceptions import DataError, DatabaseError, NEATTraderError
        
        # Test that DatabaseError can be caught as DataError
        with pytest.raises(DataError):
            raise DatabaseError("Database error")
        
        # Test that DataError can be caught as NEATTraderError
        with pytest.raises(NEATTraderError):
            raise DataError("Data error")
        
        # Test that DatabaseError can be caught as NEATTraderError
        with pytest.raises(NEATTraderError):
            raise DatabaseError("Database error")
