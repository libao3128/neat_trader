import sqlite3
import pandas as pd
import numpy as np
import random
import datetime

class DataHandler:
    '''
    DataHandler class for handling stock price data from a SQLite database.
    Attributes:
        db_path (str): Path to the SQLite database file.
    Methods:
        __init__(db_path='data/mydatabase.db'):
            Initializes the DataHandler with the specified database path.
        _connect():
            Establishes a connection to the SQLite database.
        get_random_data(is_test=False, num_date=365):
    '''
    def __init__(self, db_path='data/mydatabase.db'):
        self.db_path = db_path

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def get_random_data(self, is_test=False, data_length=365):
        """
        Fetches random stock price data for a given number of days.

        Args:
            is_test (bool, optional): If True, fetches data for the entire date range of the ticker. Defaults to False.
            num_date (int, optional): Number of days of data to fetch. Defaults to 365.

        Returns:
            pd.DataFrame: DataFrame containing the stock price data.
        """
        con = self._connect()
        
        # get a random ticker
        tickers = pd.read_sql('SELECT DISTINCT Ticker FROM Price', con)
        ticker = np.random.choice(tickers['Ticker'].tolist())
        
        if is_test:
            query = f'SELECT * FROM Price WHERE Ticker="{ticker}"'
            data = pd.read_sql(query, con)
            data.index = pd.DatetimeIndex(data['Datetime'])
            con.close()
            return data
        
        # get a random date
        dates = pd.read_sql(f'SELECT Datetime FROM Price WHERE Ticker="{ticker}" ORDER BY Datetime', con)

        start_date, end_date = self._uniform_sample(dates['Datetime'], data_length)
        if start_date is None:
            return self.get_random_data(is_test, data_length)

        query = f'''
            SELECT * FROM Price WHERE Ticker="{ticker}"
            AND Datetime>=DATETIME({start_date}, "unixepoch")
            AND Datetime<=DATETIME({end_date}, "unixepoch")
            ORDER BY Datetime
        '''

        data = pd.read_sql(query, con)
        data.index = pd.DatetimeIndex(data['Datetime'])
        con.close()
        return data
    
    def _uniform_sample(self, available_dates, data_length):
        """
        Uniformly samples a random date from the available dates.

        Args:
            available_dates (pd.DatetimeIndex): Available dates for sampling.
            data_length (int): Length of the data to sample.

        Returns:
            pd.Timestamp: Randomly sampled date.
        """
        if len(available_dates) < data_length:
            return None, None
        start_index = np.random.randint(0, len(available_dates) - data_length)
        return available_dates[start_index], available_dates[start_index + data_length - 1]

class CryptoDataHandler(DataHandler):
    """
    A class to handle cryptocurrency data from a SQLite database.
    Inherits from DataHandler.
    """
    def __init__(self, db_path=r'data\binance.db'):
        super().__init__(db_path)
        
    def get_random_data(self, is_test=False, data_length=365):
        con = self._connect()
        
        # get random start time
        timestamps = pd.read_sql('SELECT timestamp FROM BTC_USDT_15m ORDER BY timestamp', con)
        start_time, end_time = self._uniform_sample(timestamps['timestamp'], data_length)
        
        # get data
        query = f'''
            SELECT timestamp, open AS Open, close as Close, high as High, low as Low, volume as Volume 
            FROM BTC_USDT_15m 
            WHERE timestamp>="{start_time}"AND timestamp<="{end_time}" 
            ORDER BY timestamp
            '''
        data = pd.read_sql(query, con)
        data.index = pd.DatetimeIndex(data['timestamp'])
        data = data.drop(columns=['timestamp'])
        
        con.close()
        return data
        
if __name__ == '__main__':
    handler = CryptoDataHandler()
    print(handler.get_random_data())