"""
Configuration module for NEAT Trader package.

This module contains constants, default values, and configuration settings
used throughout the package.
"""

import os
from typing import Dict, Any

# Default database paths
DEFAULT_DB_PATH = 'data/mydatabase.db'
DEFAULT_CRYPTO_DB_PATH = 'data/binance.db'

# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = 'checkpoint'

# Default trading parameters
DEFAULT_TRADING_PERIOD_LENGTH = 90
DEFAULT_TOTAL_GENERATION = 200
DEFAULT_NUM_PROCESS = 16
DEFAULT_CASH = 1000000
DEFAULT_COMMISSION = 0.002

# Default strategy parameters
DEFAULT_THRESHOLD = 0.5
DEFAULT_N1 = 5
DEFAULT_N2 = 12
DEFAULT_N3 = 26

# Node names mapping for NEAT networks
NODE_NAMES = {
    -1: 'long_position', 
    -2: 'short_position', 
    -3: 'sma5', 
    -4: 'sma10',
    -5: 'slowk', 
    -6: 'slowd', 
    -7: 'macdhist_diff', 
    -8: 'cci', 
    -9: 'willr',
    -10: 'rsi', 
    -11: 'adosc', 
    0: 'buy', 
    1: 'sell', 
    2: 'volume'
}

# Graphviz path (Windows specific - should be configurable)
GRAPHVIZ_PATH = r'C:/Users/USER/OneDrive - 國立陽明交通大學/文件/academic/大四上/演化計算/Project/Graphviz/bin'

# Default visualization settings
DEFAULT_VIZ_FORMAT = 'svg'
DEFAULT_VIZ_VIEW = False

# Fitness function weights and parameters
FITNESS_WEIGHTS = {
    'relative_ror_weight': 1.5,
    'max_drawdown_weight': 0.5,
    'trade_freq_weight': 0.0005,
    'trade_duration_penalty_weight': 0.001,
    'crypto_trade_freq_weight': 0.0001
}

# Risk management thresholds
RISK_THRESHOLDS = {
    'max_drawdown_warning': 30,
    'max_drawdown_critical': 50,
    'max_drawdown_fatal': 70,
    'min_trade_duration_days': 14
}

def get_config() -> Dict[str, Any]:
    """
    Get the complete configuration dictionary.
    
    Returns:
        Dict[str, Any]: Configuration dictionary containing all settings
    """
    return {
        'database': {
            'default_path': DEFAULT_DB_PATH,
            'crypto_path': DEFAULT_CRYPTO_DB_PATH
        },
        'trading': {
            'period_length': DEFAULT_TRADING_PERIOD_LENGTH,
            'total_generation': DEFAULT_TOTAL_GENERATION,
            'num_process': DEFAULT_NUM_PROCESS,
            'cash': DEFAULT_CASH,
            'commission': DEFAULT_COMMISSION
        },
        'strategy': {
            'threshold': DEFAULT_THRESHOLD,
            'n1': DEFAULT_N1,
            'n2': DEFAULT_N2,
            'n3': DEFAULT_N3
        },
        'visualization': {
            'format': DEFAULT_VIZ_FORMAT,
            'view': DEFAULT_VIZ_VIEW,
            'graphviz_path': GRAPHVIZ_PATH
        },
        'fitness': FITNESS_WEIGHTS,
        'risk': RISK_THRESHOLDS,
        'node_names': NODE_NAMES
    }
