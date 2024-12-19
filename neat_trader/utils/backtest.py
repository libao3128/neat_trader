from backtesting import Backtest, Strategy
from talib.abstract import STOCH as KD, MACD, CCI, SMA, WILLR, RSI, ADOSC
import pandas as pd
import numpy as np
from multiprocessing import Pool
from typing import List, Tuple

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
        input_data = self.data_preprocessed()
        buy, sell, vol = self.model.activate(input_data)
        action = None
        if buy > self.threshold:
            action = 0
        if sell > self.threshold:
            action = 1
        if sell > self.threshold and buy > self.threshold:
            action = np.argmax([buy, sell])

        vol = int(self.equity * vol / self.data.df.Close.iloc[-1])

        if action == 0 and vol != 0:
            self.buy(size=vol)
        elif action == 1 and vol != 0:
            self.sell(size=vol)

    def data_preprocessed(self):
        high, low, close = self.data.df.High, self.data.df.Low, self.data.df.Close
        price = close.iloc[-1]
        length = len(self.data.df.High)
        slowk, slowd = tuple(self.kd[:length])
        macd, macdsignal, macdhist = tuple(self.macd[:length])
        macdhist_diff = (macdhist[-1] - macdhist[-2]) / price
        cci = self.cci[:length]
        willr = self.willr[:length]
        rsi = self.rsi[:length]
        adosc = self.adosc[:length]
        sma5 = self.sma5[:length]
        price_sma5 = price / sma5 - 1
        sma10 = self.sma10[:length]
        price_sma10 = price / sma10 - 1
        long_position, short_position = 0, 0
        if self.position:
            if self.position.is_long:
                long_position = self.position.size * price / self.equity
            else:
                short_position = abs(self.position.size * price / self.equity)

        return (
            long_position, short_position,
            price_sma5[-1], price_sma10[-1],
            slowk[-1], slowd[-1],
            macdhist_diff,
            cci[-1],
            willr[-1],
            rsi[-1],
            adosc[-1]
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
        
    with Pool(processes=num_processes) as pool:
        args = [(model, d) for model, d in zip(models, data)]
        results = pool.map(test_single_net, args)
    return zip(*results)

if __name__ == '__main__':
    import yfinance as yf
    from neat_trader.utils.tool import test

    df = yf.download('AAPL', interval='1d', period='360d')
    df.index = pd.to_datetime(df.index)
    df['Ticker'] = 'AAPL'

    performance, bt = test(r'checkpoint\1210_1827\winner/')
    print(performance._trades)