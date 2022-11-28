from backtesting import Backtest, Strategy
import backtesting
from backtesting.lib import crossover
from talib.abstract import STOCH as KD
from talib.abstract import MACD
from talib import SMA, EMA, WMA, RSI, MOM, CCI, WILLR, AD
import pandas as pd

class NEAT_strategy(Strategy): #交易策略命名為SmaClass，使用backtesting.py的Strategy功能
    n1 = 5
    n2 = 12
    n3 = 26

    model = None
    threshold = 0.5

    def init(self):
    
        self.kd = self.I(KD, self.data.High, self.data.Low, self.data.Close, name='KD')
        self.macd = self.I(MACD, self.data.Close, name='MACD')
        #print(123)
    def next(self):
        input = self.data_preprocessed()
        #print(input)
        trade, vol = self.model.activate(input)
        vol = abs(vol)
        if vol>=1:
            vol = 0.999
        
        if trade>self.threshold and vol!=0:
            self.buy(size=vol)
        elif trade<-self.threshold and vol!=0:
            self.sell(size=vol)
    
    def data_preprocessed(self):
        # Proceed only with out-of-sample data. Prepare some variables
        high, low, close = self.data.df.High, self.data.df.Low, self.data.df.Close
        current_time = self.data.index[-1]
        #print('data:',len(self.data.df))
        slowk, slowd = KD(high, low, close)
        macd, macdsignal, macdhist = MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        #print(KD(high, low, close))

        if self.position:
            position = self.position.pl_pct
        else:
            position = 0
        
        return (position ,slowk[-1], slowd[-1], macdhist[-1])

def backtest(model, data):
    new_strategy = NEAT_strategy
    new_strategy.model = model
    bt = Backtest(data, new_strategy,
                  cash=10000, commission=.002,
                  exclusive_orders=True)
    output = bt.run()
    return output, bt

class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()

if __name__ == '__main__':
    backtesting.set_bokeh_output(notebook=False)
    new_stra = SmaCross
    new_stra.model = 123
    import yfinance as yf
    df = yf.download('AAPL', interval='15m', period='60d')
    
    df.index = pd.to_datetime(df.index)
    bt = Backtest(df, new_stra,
                  cash=10000, commission=.002,
                  exclusive_orders=True)

    output = bt.run()
    print(output)
    print(output.keys())