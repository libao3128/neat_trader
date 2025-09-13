"""
Fitness functions for NEAT Trader package.

This module contains various fitness functions used to evaluate the performance
of trading strategies evolved by the NEAT algorithm.
"""

import pandas as pd
import datetime
import numpy as np
from typing import Union
from ..config import FITNESS_WEIGHTS, RISK_THRESHOLDS

def sample_fitness(performance: pd.DataFrame) -> float:
    """
    Sample fitness function - placeholder implementation.
    
    Args:
        performance: Backtesting performance DataFrame
        
    Returns:
        Fitness score (currently returns 0)
    """
    # The fitness function should be based on backtesting report and return a score to
    # evaluate trading strategy performance.
    score = 0.0
    return score

def SQN(performance: pd.DataFrame) -> float:
    """
    Fitness function based on System Quality Number (SQN).
    
    Args:
        performance: Backtesting performance DataFrame
        
    Returns:
        SQN value from performance metrics
    """
    return performance['SQN']

def multi_objective_fitness_function_1(performance: pd.DataFrame) -> float:
    """
    Multi-objective fitness function with discrete scoring.
    
    Calculates fitness based on:
    - Return on Investment (RoR) reward
    - Relative RoR compared to buy & hold
    - Maximum drawdown penalty
    
    Args:
        performance: Backtesting performance DataFrame
        
    Returns:
        Fitness score based on discrete scoring system
    """
    fitness = 0.0

    # RoR reward
    ror = performance['Return [%]']
    if ror > 0:
        fitness += 1
    
    # Relative RoR reward
    bh_ror = performance['Buy & Hold Return [%]']
    relative_ror = (ror - bh_ror)
    if relative_ror > 0:
        fitness += 3
    if relative_ror > 20:
        fitness += 1
    
    # Max drawdown penalty
    max_drawdown = performance['Max. Drawdown [%]']
    if max_drawdown > RISK_THRESHOLDS['max_drawdown_warning']:
        fitness -= 1
    if max_drawdown > RISK_THRESHOLDS['max_drawdown_critical']:
        fitness -= 1
    if max_drawdown > RISK_THRESHOLDS['max_drawdown_fatal']:
        fitness -= 10

    return fitness

def multi_objective_fitness_function_2(performance: pd.DataFrame) -> float:
    """
    Advanced multi-objective fitness function with continuous scoring.
    
    Calculates fitness based on:
    - Return on Investment (RoR) score
    - Relative RoR compared to buy & hold (weighted)
    - Maximum drawdown penalty
    - Trade frequency reward
    - Average trade duration penalty
    
    Args:
        performance: Backtesting performance DataFrame
        
    Returns:
        Continuous fitness score
    """
    ror = performance['Return [%]']  # The RoR of strategy
    bh_ror = performance['Buy & Hold Return [%]']  # The RoR of Benchmark
    relative_ror = (ror - bh_ror)
    
    relative_ror_score = relative_ror / 100 * FITNESS_WEIGHTS['relative_ror_weight']
    ror_score = ror / 100

    # Max drawdown score
    max_drawdown = performance['Max. Drawdown [%]']
    if max_drawdown is None:
        max_drawdown = 0
    max_drawdown_score = max_drawdown / 100 * FITNESS_WEIGHTS['max_drawdown_weight']

    # Trade frequency score
    num_trade = performance['# Trades']
    trade_freq_score = num_trade * FITNESS_WEIGHTS['trade_freq_weight']

    # Average trade duration penalty
    avg_trade_duration = performance['Avg. Trade Duration']
    trade_duration_penalty = 0.0
    if num_trade > 0 and avg_trade_duration.days > RISK_THRESHOLDS['min_trade_duration_days']:
        trade_duration_penalty = -FITNESS_WEIGHTS['trade_duration_penalty_weight'] * avg_trade_duration.days
    
    return ror_score + relative_ror_score + max_drawdown_score + trade_freq_score + trade_duration_penalty

def outperform_benchmark(performance: pd.DataFrame) -> float:
    """
    Fitness function focused on outperforming benchmark (buy & hold).
    
    Args:
        performance: Backtesting performance DataFrame
        
    Returns:
        Fitness score emphasizing benchmark outperformance
    """
    ror = performance['Return [%]']
    bh_ror = performance['Buy & Hold Return [%]']
    relative_ror_reward = (ror - bh_ror) / 100
    
    # Return reward
    return_reward = ror / 100 * 0.1

    # Encourage the strategy to trade more
    num_trade = performance['# Trades']
    trade_freq_reward = num_trade * 0.0001

    # Risk management
    max_drawdown = performance['Max. Drawdown [%]']
    if max_drawdown == 0:
        max_drawdown_penalty = 0
    else:
        max_drawdown_penalty = 1 / (1 - (max_drawdown / 100))
    
    return relative_ror_reward + return_reward + trade_freq_reward - max_drawdown_penalty

def gpt_fitness_fn(performance: pd.DataFrame) -> float:
    """
    Advanced fitness function using Sharpe ratio and trade frequency optimization.
    
    Args:
        performance: Backtesting performance DataFrame
        
    Returns:
        Fitness score based on Sharpe ratio and optimal trade frequency
    """
    def trade_freq_function(t: int, t_target: float) -> float:
        """Gaussian function for trade frequency scoring."""
        sigma = 5
        return np.exp(-(t - t_target) ** 2 / sigma ** 2)
    
    if performance['# Trades'] == 0:
        return 0.0
    
    sharpe_ratio = performance['Sharpe Ratio']
    if sharpe_ratio is None:
        sharpe_ratio = 1.0
    
    annualized_return = performance['Return (Ann.) [%]'] / 100
    max_drawdown = performance['Max. Drawdown [%]'] / 100
    if max_drawdown is None:
        max_drawdown = 0.0
        
    duration = performance['Duration']
    duration_days = duration.days if isinstance(duration, datetime.timedelta) else duration
    t_target = duration_days / 30 * 3
    trade_freq = performance['# Trades']
    trade_freq_score = trade_freq_function(trade_freq, t_target)
    
    score = (sharpe_ratio * annualized_return) / (1 + max_drawdown) * trade_freq_score * 100
    return score

def crypto_fitness_fn(performance: pd.DataFrame) -> float:
    """
    Fitness function optimized for cryptocurrency trading.
    
    Simplified version focusing on:
    - Return on Investment (RoR)
    - Maximum drawdown penalty
    - Trade frequency reward
    
    Args:
        performance: Backtesting performance DataFrame
        
    Returns:
        Fitness score optimized for crypto trading
    """
    ror = performance['Return [%]']  # The RoR of strategy
    ror_score = ror / 100

    # Max drawdown penalty
    max_drawdown = performance['Max. Drawdown [%]']
    if max_drawdown is None:
        max_drawdown = 0
    max_drawdown_score = max_drawdown / 100 * FITNESS_WEIGHTS['max_drawdown_weight']

    # Trade frequency score
    num_trade = performance['# Trades']
    trade_freq_score = num_trade * FITNESS_WEIGHTS['crypto_trade_freq_weight']

    return ror_score + max_drawdown_score + trade_freq_score 