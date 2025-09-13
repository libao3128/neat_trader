"""
Trading strategy implementation for NEAT Trader package.

This module contains the base trading strategy class that integrates with NEAT
neural networks for automated trading decisions.
"""

from typing import Tuple, Optional, Any
import numpy as np
from backtesting import Backtest, Strategy
from talib.abstract import STOCH as KD, MACD, CCI, SMA, WILLR, RSI, ADOSC

from ..config import DEFAULT_THRESHOLD, DEFAULT_N1, DEFAULT_N2, DEFAULT_N3

class NEATStrategyBase(Strategy):
    """
    Base trading strategy class that integrates NEAT neural networks.
    
    This class provides the foundation for NEAT-based trading strategies,
    handling technical indicators and neural network decision making.
    
    Attributes:
        n1, n2, n3: Technical indicator parameters
        model: NEAT neural network model
        threshold: Decision threshold for buy/sell signals
    """
    n1 = DEFAULT_N1
    n2 = DEFAULT_N2
    n3 = DEFAULT_N3

    model: Optional[Any] = None
    threshold: float = DEFAULT_THRESHOLD

    def init(self) -> None:
        """
        Initialize technical indicators for the strategy.
        
        Sets up all required technical indicators including:
        - Stochastic Oscillator (KD)
        - MACD
        - Commodity Channel Index (CCI)
        - Simple Moving Averages (SMA)
        - Williams %R
        - Relative Strength Index (RSI)
        - Accumulation/Distribution Oscillator (ADOSC)
        """
        self.kd = self.I(KD, self.data.High, self.data.Low, self.data.Close, name='KD')
        self.macd = self.I(MACD, self.data.Close, name='MACD')
        self.cci = self.I(CCI, self.data.High, self.data.Low, self.data.Close, name='CCI')
        self.sma5 = self.I(SMA, self.data.Close, 5, name='SMA5')
        self.sma10 = self.I(SMA, self.data.Close, 10, name='SMA10')
        self.willr = self.I(WILLR, self.data.High, self.data.Low, self.data.Close, name='WILLR')
        self.rsi = self.I(RSI, self.data.Close, name='RSI')
        self.adosc = self.I(ADOSC, np.array(self.data.High, dtype=float), 
                            np.array(self.data.Low, dtype=float), 
                            np.array(self.data.Close, dtype=float), 
                            np.array(self.data.Volume, dtype=float), name='ADOSC')

    def next(self) -> None:
        """
        Execute trading logic for each time step.
        
        This method is called for each bar in the backtest and:
        1. Preprocesses market data into input features
        2. Activates the NEAT neural network model
        3. Makes trading decisions based on model output
        4. Executes buy/sell orders
        """
        if self.model is None:
            return
            
        # Preprocess the data to get the input features for the model
        input_data = self.data_preprocessed()
        
        # Activate the model with the preprocessed data to get buy, sell signals and volume
        buy, sell, vol = self.model.activate(input_data)
        
        # Determine the action to take based on the buy and sell signals
        action: Optional[int] = None
        if buy > self.threshold and sell > self.threshold:
            action = np.argmax([buy, sell])  # Choose the stronger signal
        elif buy > self.threshold:
            action = 0  # Buy signal
        elif sell > self.threshold:
            action = 1  # Sell signal

        # Calculate the volume to trade based on the equity and current price
        vol_size = int(self.equity * vol / self.data.df.Close.iloc[-1])

        # Execute the trade based on the determined action
        if action == 0 and vol_size > 0:
            self.buy(size=vol_size)
        elif action == 1 and vol_size > 0:
            self.sell(size=vol_size)

    def data_preprocessed(self) -> Tuple[float, ...]:
        """
        Preprocess market data into input features for the NEAT model.
        
        Returns:
            Tuple of 11 input features:
            - long_position: Current long position size
            - short_position: Current short position size  
            - sma5_ratio: Price to SMA5 ratio
            - sma10_ratio: Price to SMA10 ratio
            - slowk: Stochastic %K value
            - slowd: Stochastic %D value
            - macdhist_diff: MACD histogram difference
            - cci: Commodity Channel Index
            - willr: Williams %R
            - rsi: Relative Strength Index
            - adosc: Accumulation/Distribution Oscillator
        """
        data = self.data.df
        price = data.Close.iloc[-1]
        length = len(data)
        
        indicators = {
            'slowk': self.kd[0][:length],
            'slowd': self.kd[1][:length],
            'macdhist_diff': (self.macd[2][:length][-1] - self.macd[2][:length][-2]) / price,
            'cci': self.cci[:length],
            'willr': self.willr[:length],
            'rsi': self.rsi[:length],
            'adosc': self.adosc[:length],
            'price_sma5': price / self.sma5[:length] - 1,
            'price_sma10': price / self.sma10[:length] - 1
        }
        
        long_position, short_position = 0.0, 0.0
        if self.position:
            position_size = self.position.size * price / self.equity
            if self.position.is_long:
                long_position = position_size
            else:
                short_position = abs(position_size)

        return (
            long_position, short_position,
            indicators['price_sma5'][-1], indicators['price_sma10'][-1],
            indicators['slowk'][-1], indicators['slowd'][-1],
            indicators['macdhist_diff'],
            indicators['cci'][-1],
            indicators['willr'][-1],
            indicators['rsi'][-1],
            indicators['adosc'][-1]
        )