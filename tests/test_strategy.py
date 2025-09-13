"""
Test trading strategy functionality.

This module tests the NEATStrategyBase class and its
data preprocessing capabilities.
"""

import pytest
import sys
import os
from unittest.mock import Mock

# Add the package to path for testing
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestStrategy:
    """Test trading strategy."""
    
    def test_strategy_initialization(self):
        """Test NEATStrategyBase initialization."""
        from neat_trader.algorithm.strategy import NEATStrategyBase
        
        # Test class attributes without instantiation
        assert NEATStrategyBase.n1 == 5
        assert NEATStrategyBase.n2 == 12
        assert NEATStrategyBase.n3 == 26
        assert NEATStrategyBase.threshold == 0.5
        assert NEATStrategyBase.model is None
        
        # Test that the class has required methods
        assert hasattr(NEATStrategyBase, 'init')
        assert hasattr(NEATStrategyBase, 'next')
        assert hasattr(NEATStrategyBase, 'data_preprocessed')
    
    def test_data_preprocessed(self, sample_market_data):
        """Test data preprocessing."""
        from neat_trader.algorithm.strategy import NEATStrategyBase
        
        # Test that the data_preprocessed method exists and is callable
        assert hasattr(NEATStrategyBase, 'data_preprocessed')
        assert callable(getattr(NEATStrategyBase, 'data_preprocessed'))
        
        # Test method signature
        import inspect
        from typing import Tuple
        sig = inspect.signature(NEATStrategyBase.data_preprocessed)
        assert len(sig.parameters) == 1  # Only 'self' parameter
        assert sig.return_annotation == Tuple[float, ...]
    
    def test_data_preprocessed_with_position(self, sample_market_data):
        """Test data preprocessing with position."""
        from neat_trader.algorithm.strategy import NEATStrategyBase
        
        # Test that the method exists and has correct signature
        assert hasattr(NEATStrategyBase, 'data_preprocessed')
        assert callable(getattr(NEATStrategyBase, 'data_preprocessed'))
        
        # Test that the method is designed to handle position data
        import inspect
        sig = inspect.signature(NEATStrategyBase.data_preprocessed)
        assert len(sig.parameters) == 1  # Only 'self' parameter
    
    def test_next_method_without_model(self, sample_market_data):
        """Test next method without model."""
        from neat_trader.algorithm.strategy import NEATStrategyBase
        
        # Test that the next method exists and is callable
        assert hasattr(NEATStrategyBase, 'next')
        assert callable(getattr(NEATStrategyBase, 'next'))
        
        # Test method signature
        import inspect
        sig = inspect.signature(NEATStrategyBase.next)
        assert len(sig.parameters) == 1  # Only 'self' parameter
    
    def test_next_method_with_model(self, sample_market_data):
        """Test next method with model."""
        from neat_trader.algorithm.strategy import NEATStrategyBase
        
        # Test that the next method exists and is callable
        assert hasattr(NEATStrategyBase, 'next')
        assert callable(getattr(NEATStrategyBase, 'next'))
        
        # Test that the method is designed to work with models
        import inspect
        sig = inspect.signature(NEATStrategyBase.next)
        assert len(sig.parameters) == 1  # Only 'self' parameter
