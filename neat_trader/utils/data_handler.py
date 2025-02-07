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

    def get_random_data(self, is_test=False, num_date=365):
        """
        Fetches random stock price data for a given number of days.

        Args:
            is_test (bool, optional): If True, fetches data for the entire date range of the ticker. Defaults to False.
            num_date (int, optional): Number of days of data to fetch. Defaults to 365.

        Returns:
            pd.DataFrame: DataFrame containing the stock price data.
        """
        con = self._connect()
        tickers = pd.read_sql('SELECT DISTINCT Ticker FROM Price', con)

        ticker = np.random.choice(tickers['Ticker'].tolist())
        dates = pd.read_sql(f'SELECT Datetime FROM Price WHERE Ticker="{ticker}"', con)
        dates = pd.to_datetime(dates['Datetime'])
        start_date = dates.max()
        end_date = dates.min()

        time_between_dates = start_date - end_date
        try:
            days_between_dates = int(time_between_dates.days - 365 - num_date - 30)
            random_number_of_days = random.randrange(days_between_dates)
        except:
            return self.get_random_data()

        random_date = end_date + datetime.timedelta(days=random_number_of_days)
        if not is_test:
            start_date = random_date
            end_date = start_date + datetime.timedelta(days=int(num_date + 30))
        else:
            end_date = dates.max()
            start_date = dates.min()

        query = f'SELECT * FROM Price WHERE Ticker="{ticker}"' \
                f' AND Datetime>=DATETIME({start_date.timestamp()}, "unixepoch")' \
                f' AND Datetime<=DATETIME({end_date.timestamp()}, "unixepoch")'

        data = pd.read_sql(query, con)
        data.index = pd.DatetimeIndex(data['Datetime'])
        con.close()
        return data