"""
Data handling utilities for NEAT Trader package.

This module provides classes for handling market data from SQLite databases,
including both traditional stock data and cryptocurrency data.
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from ..exceptions import DatabaseError, InsufficientDataError

class DataHandler:
    """
    DataHandler class for handling stock price data from a SQLite database.
    
    This class provides methods to fetch random market data for backtesting
    NEAT trading strategies.
    
    Attributes:
        db_path: Path to the SQLite database file
    """
    
    def __init__(self, db_path: str = 'data/mydatabase.db'):
        """
        Initialize DataHandler.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        """
        Establish connection to SQLite database.
        
        Returns:
            SQLite database connection
            
        Raises:
            DatabaseError: If connection fails
        """
        try:
            return sqlite3.connect(self.db_path)
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to connect to database {self.db_path}: {str(e)}")

    def get_random_data(self, is_test: bool = False, data_length: int = 365) -> pd.DataFrame:
        """
        Fetch random stock price data for a given number of days.

        Args:
            is_test: If True, fetches data for the entire date range of the ticker
            data_length: Number of days of data to fetch

        Returns:
            DataFrame containing the stock price data
            
        Raises:
            DatabaseError: If database query fails
            InsufficientDataError: If insufficient data is available
        """
        con = self._connect()
        
        try:
            # Get a random ticker
            tickers = pd.read_sql('SELECT DISTINCT Ticker FROM Price', con)
            if tickers.empty:
                raise InsufficientDataError("No tickers found in database")
                
            ticker = np.random.choice(tickers['Ticker'].tolist())
            
            if is_test:
                query = f'SELECT * FROM Price WHERE Ticker="{ticker}"'
                data = pd.read_sql(query, con)
                if data.empty:
                    raise InsufficientDataError(f"No data found for ticker {ticker}")
                data.index = pd.DatetimeIndex(data['Datetime'])
                return data
            
            # Get a random date range
            dates = pd.read_sql(f'SELECT Datetime FROM Price WHERE Ticker="{ticker}" ORDER BY Datetime', con)
            if dates.empty:
                raise InsufficientDataError(f"No dates found for ticker {ticker}")

            start_date, end_date = self._uniform_sample(dates['Datetime'], data_length)
            if start_date is None:
                return self.get_random_data(is_test, data_length)

            query = f'''
                SELECT * FROM Price WHERE Ticker="{ticker}"
                AND Datetime>="{start_date}"
                AND Datetime<="{end_date}"
                ORDER BY Datetime
            '''

            data = pd.read_sql(query, con)
            if data.empty:
                raise InsufficientDataError(f"No data found for ticker {ticker} in date range")
                
            data.index = pd.DatetimeIndex(data['Datetime'])
            return data
            
        except Exception as e:
            if isinstance(e, (DatabaseError, InsufficientDataError)):
                raise
            raise DatabaseError(f"Failed to fetch data: {str(e)}")
        finally:
            con.close()
    
    def _uniform_sample(self, available_dates: pd.Series, data_length: int) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """
        Uniformly sample a random date range from available dates.

        Args:
            available_dates: Available dates for sampling
            data_length: Length of the data to sample

        Returns:
            Tuple of (start_date, end_date) or (None, None) if insufficient data
        """
        if len(available_dates) < data_length:
            return None, None
        start_index = np.random.randint(0, len(available_dates) - data_length)
        return available_dates.iloc[start_index], available_dates.iloc[start_index + data_length - 1]

class CryptoDataHandler(DataHandler):
    """
    DataHandler for cryptocurrency data from a SQLite database.
    
    Inherits from DataHandler and provides specialized methods for
    handling cryptocurrency market data.
    """
    
    def __init__(self, db_path: str = r'data\binance.db'):
        """
        Initialize CryptoDataHandler.
        
        Args:
            db_path: Path to the cryptocurrency database file
        """
        super().__init__(db_path)
        
    def get_random_data(self, is_test: bool = False, data_length: int = 365) -> pd.DataFrame:
        """
        Fetch random cryptocurrency data for a given number of periods.

        Args:
            is_test: If True, fetches data for the entire date range
            data_length: Number of periods of data to fetch

        Returns:
            DataFrame containing the cryptocurrency price data
            
        Raises:
            DatabaseError: If database query fails
            InsufficientDataError: If insufficient data is available
        """
        con = self._connect()
        
        try:
            # Get time range
            timestamps = pd.read_sql('SELECT timestamp FROM BTC_USDT_15m ORDER BY timestamp', con)
            if timestamps.empty:
                raise InsufficientDataError("No timestamps found in crypto database")
                
            if is_test:
                start_time, end_time = timestamps['timestamp'].min(), timestamps['timestamp'].max()
            else:
                start_time, end_time = self._uniform_sample(timestamps['timestamp'], data_length)
                if start_time is None:
                    return self.get_random_data(is_test, data_length)
            
            # Get data
            query = f'''
                SELECT timestamp, open AS Open, close as Close, high as High, low as Low, volume as Volume 
                FROM BTC_USDT_15m 
                WHERE timestamp>="{start_time}" AND timestamp<="{end_time}" 
                ORDER BY timestamp
                '''
            data = pd.read_sql(query, con)
            if data.empty:
                raise InsufficientDataError("No crypto data found in specified time range")
                
            data.index = pd.DatetimeIndex(data['timestamp'])
            data = data.drop(columns=['timestamp'])
            
            return data
            
        except Exception as e:
            if isinstance(e, (DatabaseError, InsufficientDataError)):
                raise
            raise DatabaseError(f"Failed to fetch crypto data: {str(e)}")
        finally:
            con.close()
        
if __name__ == '__main__':
    handler = CryptoDataHandler()
    print(handler.get_random_data())