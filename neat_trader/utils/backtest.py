from backtesting import Backtest, Strategy
from talib.abstract import STOCH as KD, MACD, CCI, SMA, WILLR, RSI, ADOSC
import pandas as pd
import numpy as np
from multiprocessing import Pool
from typing import List, Tuple
from tqdm import tqdm

class NEATStrategy(Strategy):
    n1 = 5
    n2 = 12
    n3 = 26

    model = None
    threshold = 0.5

    def init(self):
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

    def next(self):
        # Preprocess the data to get the input features for the model
        input_data = self.data_preprocessed()
        
        # Activate the model with the preprocessed data to get buy, sell signals and volume
        buy, sell, vol = self.model.activate(input_data)
        
        # Determine the action to take based on the buy and sell signals
        action = None
        if buy > self.threshold and sell > self.threshold:
            action = np.argmax([buy, sell])  # Choose the stronger signal
        elif buy > self.threshold:
            action = 0  # Buy signal
        elif sell > self.threshold:
            action = 1  # Sell signal

        # Calculate the volume to trade based on the equity and current price
        vol = int(self.equity * vol / self.data.df.Close.iloc[-1])

        # Execute the trade based on the determined action
        if action == 0 and vol > 0:
            self.buy(size=vol)
        elif action == 1 and vol > 0:
            self.sell(size=vol)

    def data_preprocessed(self):
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
        
        long_position, short_position = 0, 0
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

def backtest(model, data):
    NEATStrategy.model = model
    bt = Backtest(data, NEATStrategy, cash=1000000, commission=.002, exclusive_orders=False)
    output = bt.run()
    return output, bt

def test_single_net(args):
    model, data = args
    return backtest(model, data)

def multi_process_backtest(models, data, num_processes=None) -> Tuple[List[pd.DataFrame], List[Backtest]]:
    if not isinstance(models, list):
        models = [models]*len(data)
    if not isinstance(data, list):
        data = [data]*len(models)
    
    args = [(model, d) for model, d in zip(models, data)]
    
    with Pool(processes=num_processes) as pool:
        results = list(pool.imap(test_single_net, args))
    return zip(*results)

if __name__ == '__main__':
    import yfinance as yf
    from neat_trader.utils.tool import test


    df = yf.download('AAPL', interval='1d', period='360d')
    df.index = pd.to_datetime(df.index)
    df['Ticker'] = 'AAPL'

    performance, bt = test(r'checkpoint\1210_1827\winner/')
    print(performance._trades)