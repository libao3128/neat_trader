"""
Test configuration system.

This module tests the centralized configuration system
and all configuration-related functionality.
"""

import pytest
import sys
import os

# Add the package to path for testing
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestConfiguration:
    """Test configuration system."""
    
    def test_get_config(self):
        """Test configuration retrieval."""
        from neat_trader.config import get_config
        
        config = get_config()
        assert isinstance(config, dict)
        assert 'database' in config
        assert 'trading' in config
        assert 'strategy' in config
        assert 'visualization' in config
        assert 'fitness' in config
        assert 'risk' in config
        assert 'node_names' in config
    
    def test_node_names(self):
        """Test node names mapping."""
        from neat_trader.config import NODE_NAMES
        
        assert isinstance(NODE_NAMES, dict)
        assert len(NODE_NAMES) > 0
        
        # Check for expected keys
        expected_keys = [-1, -2, -3, -4, -5, 0, 1, 2]
        for key in expected_keys:
            assert key in NODE_NAMES
    
    def test_fitness_weights(self):
        """Test fitness weights configuration."""
        from neat_trader.config import FITNESS_WEIGHTS
        
        assert isinstance(FITNESS_WEIGHTS, dict)
        assert 'relative_ror_weight' in FITNESS_WEIGHTS
        assert 'max_drawdown_weight' in FITNESS_WEIGHTS
        assert 'trade_freq_weight' in FITNESS_WEIGHTS
        
        # Check that weights are numeric
        for key, value in FITNESS_WEIGHTS.items():
            assert isinstance(value, (int, float))
    
    def test_risk_thresholds(self):
        """Test risk thresholds configuration."""
        from neat_trader.config import RISK_THRESHOLDS
        
        assert isinstance(RISK_THRESHOLDS, dict)
        assert 'max_drawdown_warning' in RISK_THRESHOLDS
        assert 'max_drawdown_critical' in RISK_THRESHOLDS
        assert 'max_drawdown_fatal' in RISK_THRESHOLDS
        
        # Check that thresholds are numeric
        for key, value in RISK_THRESHOLDS.items():
            assert isinstance(value, (int, float))
    
    def test_default_values(self):
        """Test default configuration values."""
        from neat_trader.config import (
            DEFAULT_DB_PATH,
            DEFAULT_CRYPTO_DB_PATH,
            DEFAULT_TRADING_PERIOD_LENGTH,
            DEFAULT_TOTAL_GENERATION,
            DEFAULT_NUM_PROCESS,
            DEFAULT_CASH,
            DEFAULT_COMMISSION
        )
        
        assert isinstance(DEFAULT_DB_PATH, str)
        assert isinstance(DEFAULT_CRYPTO_DB_PATH, str)
        assert isinstance(DEFAULT_TRADING_PERIOD_LENGTH, int)
        assert isinstance(DEFAULT_TOTAL_GENERATION, int)
        assert isinstance(DEFAULT_NUM_PROCESS, int)
        assert isinstance(DEFAULT_CASH, int)
        assert isinstance(DEFAULT_COMMISSION, float)
