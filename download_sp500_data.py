#!/usr/bin/env python3
"""
S&P 500 Data Download Script

This script downloads S&P 500 stock price data from yfinance and saves it
to a SQLite database with proper schema for the NEAT Trader package.

Usage:
    python download_sp500_data.py [--db-path DATABASE_PATH] [--period PERIOD] [--interval INTERVAL]
"""

import argparse
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys
from typing import List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SP500DataDownloader:
    """Downloads and stores S&P 500 stock data."""
    
    def __init__(self, db_path: str = "data/mydatabase.db"):
        """
        Initialize the data downloader.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.ensure_database_exists()
    
    def ensure_database_exists(self):
        """Create database and tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create Price table with proper schema
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Price (
                    Datetime TEXT NOT NULL,
                    Ticker TEXT NOT NULL,
                    Open REAL NOT NULL,
                    High REAL NOT NULL,
                    Low REAL NOT NULL,
                    Close REAL NOT NULL,
                    Volume INTEGER NOT NULL,
                    PRIMARY KEY (Datetime, Ticker)
                )
            ''')
            
            # Create index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_price_ticker_datetime 
                ON Price (Ticker, Datetime)
            ''')
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    def get_sp500_tickers(self) -> List[str]:
        """
        Get list of S&P 500 ticker symbols from local CSV file.
        
        Returns:
            List of ticker symbols
        """
        try:
            # Try to read from local CSV file first
            csv_path = "s-and-p-500-companies/data/constituents.csv"
            if os.path.exists(csv_path):
                logger.info(f"Reading S&P 500 tickers from local CSV: {csv_path}")
                df = pd.read_csv(csv_path)
                
                # Check common column names for ticker symbols
                ticker_column = None
                for col in ['Symbol', 'symbol', 'Ticker', 'ticker', 'TICKER']:
                    if col in df.columns:
                        ticker_column = col
                        break
                
                if ticker_column:
                    tickers = df[ticker_column].tolist()
                    logger.info(f"Found {len(tickers)} tickers in CSV file")
                else:
                    logger.warning(f"Could not find ticker column in CSV. Available columns: {df.columns.tolist()}")
                    # Use first column if no standard ticker column found
                    tickers = df.iloc[:, 0].tolist()
                    logger.info(f"Using first column as tickers: {len(tickers)} entries")
                
                # Clean up tickers (remove dots, replace with dashes for yfinance)
                clean_tickers = []
                for ticker in tickers:
                    if pd.notna(ticker):  # Skip NaN values
                        clean_ticker = str(ticker).replace('.', '-')
                        clean_tickers.append(clean_ticker)
                
                logger.info(f"Successfully loaded {len(clean_tickers)} S&P 500 tickers from CSV")
                return clean_tickers
            else:
                logger.warning(f"CSV file not found at {csv_path}")
                raise FileNotFoundError(f"CSV file not found at {csv_path}")
                
        except Exception as e:
            logger.error(f"Failed to get S&P 500 tickers from CSV: {e}")
            
            # Fallback: Try Wikipedia method
            try:
                logger.info("Trying Wikipedia as fallback...")
                url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
                tables = pd.read_html(url)
                sp500_table = tables[0]
                tickers = sp500_table['Symbol'].tolist()
                
                clean_tickers = []
                for ticker in tickers:
                    clean_ticker = ticker.replace('.', '-')
                    clean_tickers.append(clean_ticker)
                
                logger.info(f"Fallback successful: Found {len(clean_tickers)} S&P 500 tickers from Wikipedia")
                return clean_tickers
                
            except Exception as e2:
                logger.error(f"Wikipedia fallback also failed: {e2}")
                # Final fallback to a smaller set of major tickers
                fallback_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'UNH', 'JNJ']
                logger.warning(f"Using minimal fallback list: {len(fallback_tickers)} tickers")
                return fallback_tickers
    
    def download_ticker_data(self, ticker: str, period: str = "2y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Download data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with stock data or None if failed
        """
        import time
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period=period, interval=interval)
                
                if data.empty:
                    logger.warning(f"No data found for {ticker}")
                    return None
            
                # Reset index to get Datetime as a column
                data = data.reset_index()
                
                # Rename columns to match our schema
                data = data.rename(columns={
                    'Date': 'Datetime',
                    'Open': 'Open',
                    'High': 'High', 
                    'Low': 'Low',
                    'Close': 'Close',
                    'Volume': 'Volume'
                })
                
                # Add ticker column
                data['Ticker'] = ticker
                
                # Convert Datetime to string format
                data['Datetime'] = data['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Select only required columns
                data = data[['Datetime', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
                
                logger.info(f"Downloaded {len(data)} records for {ticker}")
                return data
                
            except Exception as e:
                if "Too Many Requests" in str(e) or "Rate limited" in str(e):
                    wait_time = (attempt + 1) * 5  # Exponential backoff
                    logger.warning(f"Rate limited for {ticker}, attempt {attempt + 1}/{max_retries}, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Failed to download data for {ticker} (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        return None
                    time.sleep(2)
        
        return None
    
    def save_data_to_db(self, data: pd.DataFrame, ticker: str):
        """
        Save data to database.
        
        Args:
            data: DataFrame with stock data
            ticker: Stock ticker symbol
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Delete existing data for this ticker to avoid duplicates
                cursor = conn.cursor()
                cursor.execute('DELETE FROM Price WHERE Ticker = ?', (ticker,))
                
                # Insert new data
                data.to_sql('Price', conn, if_exists='append', index=False)
                conn.commit()
                
                logger.info(f"Saved {len(data)} records for {ticker} to database")
                
        except Exception as e:
            logger.error(f"Failed to save data for {ticker}: {e}")
    
    def download_all_data(self, period: str = "2y", interval: str = "1d", max_tickers: Optional[int] = None):
        """
        Download data for all S&P 500 tickers.
        
        Args:
            period: Data period
            interval: Data interval
            max_tickers: Maximum number of tickers to download (for testing)
        """
        tickers = self.get_sp500_tickers()
        
        if max_tickers:
            tickers = tickers[:max_tickers]
            logger.info(f"Limiting to first {max_tickers} tickers for testing")
        
        successful_downloads = 0
        failed_downloads = 0
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Processing {ticker} ({i}/{len(tickers)})")
            
            data = self.download_ticker_data(ticker, period, interval)
            
            if data is not None:
                self.save_data_to_db(data, ticker)
                successful_downloads += 1
            else:
                failed_downloads += 1
            
            # Add delay to avoid rate limiting
            import time
            time.sleep(2)  # Increased delay to 2 seconds
        
        logger.info(f"Download complete: {successful_downloads} successful, {failed_downloads} failed")
    
    def get_database_stats(self):
        """Get statistics about the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total records
                cursor.execute('SELECT COUNT(*) FROM Price')
                total_records = cursor.fetchone()[0]
                
                # Get unique tickers
                cursor.execute('SELECT COUNT(DISTINCT Ticker) FROM Price')
                unique_tickers = cursor.fetchone()[0]
                
                # Get date range
                cursor.execute('SELECT MIN(Datetime), MAX(Datetime) FROM Price')
                date_range = cursor.fetchone()
                
                logger.info(f"Database Statistics:")
                logger.info(f"  Total records: {total_records:,}")
                logger.info(f"  Unique tickers: {unique_tickers}")
                logger.info(f"  Date range: {date_range[0]} to {date_range[1]}")
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Download S&P 500 stock data')
    parser.add_argument('--db-path', default='data/mydatabase.db', 
                       help='Path to SQLite database file')
    parser.add_argument('--period', default='2y', 
                       choices=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'],
                       help='Data period')
    parser.add_argument('--interval', default='1d',
                       choices=['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'],
                       help='Data interval')
    parser.add_argument('--max-tickers', type=int, default=None,
                       help='Maximum number of tickers to download (for testing)')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only show database statistics')
    
    args = parser.parse_args()
    
    downloader = SP500DataDownloader(args.db_path)
    
    if args.stats_only:
        downloader.get_database_stats()
    else:
        logger.info(f"Starting download to {args.db_path}")
        logger.info(f"Period: {args.period}, Interval: {args.interval}")
        
        downloader.download_all_data(args.period, args.interval, args.max_tickers)
        downloader.get_database_stats()


if __name__ == "__main__":
    main()
