from backtesting import Backtest, Strategy
import backtesting
from backtesting.lib import crossover
from talib.abstract import STOCH as KD
from talib.abstract import MACD
from talib import SMA, EMA, WMA, RSI, MOM, CCI, WILLR, ADOSC
import pandas as pd
import numpy as np


class NEAT_strategy(Strategy): #交易策略命名為SmaClass，使用backtesting.py的Strategy功能
    n1 = 5
    n2 = 12
    n3 = 26

    model = None
    threshold = 0.5
    

    def init(self):
        #print('data',self.data)
        self.kd = self.I(KD, self.data.High, self.data.Low, self.data.Close, name='KD')
        #print(self.kd)
        self.macd = self.I(MACD, self.data.Close, name='MACD')
        self.cci = self.I(CCI, self.data.High, self.data.Low, self.data.Close, name='CCI')
        self.sma5 = self.I(SMA, self.data.Close, 5, name='SMA5')
        self.sma10 = self.I(SMA, self.data.Close, 10, name='SMA10')
        self.willr = self.I(WILLR, self.data.High, self.data.Low, self.data.Close, name='WILLR')
        self.rsi = self.I(RSI, self.data.Close, name='RSI')
        #print(self.data.Volume)
        self.adosc = self.I(
            ADOSC, np.array(self.data.High,dtype=float), 
            np.array(self.data.Low,dtype=float), np.array(self.data.Close,dtype=float), 
            np.array(self.data.Volume,dtype=float), name='ADOSC')
        
    def next(self):
        
        input = self.data_preprocessed()
       # print('input',input)
        buy, sell, vol = self.model.activate(input)
        #print('output',(buy, sell, vol))
        #print(trade, vol)
        action=None
        if buy>self.threshold:
            
            action=0
        if sell>self.threshold:
            action=1
        if sell>self.threshold and buy>self.threshold:
            action=np.argmax([buy,sell])
        
        vol = int(self.equity*vol/self.data.df.Close.iloc[-1])
        
        if action==0 and vol!=0:
            #print('buy')
            self.buy(size=vol)
        elif action==1 and vol!=0:
            #print('sell')
            self.sell(size=vol)
        
        #print(self.position)
    def data_preprocessed(self):
        # Proceed only with out-of-sample data. Prepare some variables
        high, low, close = self.data.df.High, self.data.df.Low, self.data.df.Close
        price = close.iloc[-1]
        length = len(self.data.df.High)
        #print('price1',price)
        #print('len',len(self.data.df.High))
        
        #current_time = self.data.index[-1]
        #print('data:',len(self.data.df))
        #print(self.kd[:][:self.index])
        slowk, slowd = tuple(self.kd[:length]) 
        
        macd, macdsignal, macdhist = tuple(self.macd[:length]) 
        macdhist_diff = (macdhist[-1] - macdhist[-2])/price
        
        cci = self.cci[:length]

        willr = self.willr[:length]

        rsi = self.rsi[:length]

        adosc = self.adosc[:length]
        
        sma5 = self.sma5[:length]
        price_sma5 = price/sma5-1
        sma10 = self.sma10[:length]
        price_sma10 = price/sma10-1
        #print(KD(high, low, close))
        long_position, short_position = 0, 0
        if self.position:
            if self.position.is_long:
                long_position = self.position.size*price/self.equity
            else:
                short_position = abs(self.position.size*price/self.equity)
            
        
        return (
            long_position, short_position,
            price_sma5[-1], price_sma10[-1],
            slowk[-1], slowd[-1], 
            macdhist_diff,
            cci[-1],
            willr[-1],
            rsi[-1],
            adosc[-1])

def backtest(model, data=None):
    new_strategy = NEAT_strategy
    new_strategy.model = model
    

    bt = Backtest(data, new_strategy,
                  cash=1000000, commission=.002,exclusive_orders=False
                  )
    output = bt.run()
    return output, bt



class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)
        self.index=1

    def next(self):
        
        if self.index==1:
            print('buy')
            self.buy()
        elif self.index==2:
            self.buy()
            pass
            #self.buy(size=-0.99)
        elif self.index==3:
            print('sell')
            self.sell()
        elif self.index==4:
            #self.sell(size=-0.99)
            pass
        #print(self.tra)
        print(self.orders)
        print(self.equity)
        self.index+=1

if __name__ == '__main__':
    
    #backtesting.set_bokeh_output(notebook=False)
    #new_stra = NEAT_strategy
    #import neat
    #config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
    #                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
    #                    'config-feedforward')
    #p = neat.Population(config)
    #
    #new_stra.model = neat.nn.FeedForwardNetwork.create(p.population[1], config) 
    import yfinance as yf
    df = yf.download('AAPL', interval='1d', period='360d')
    ##
    df.index = pd.to_datetime(df.index)
    df['Ticker'] = 'AAPL'
    ##print(len(KD(df.High,df.Low,df.Close)[0]))
    ##print(len(df))
    #bt = Backtest(df, new_stra,
    #              cash=100000000, commission=.002,exclusive_orders=True
    #              )
    ##bt.optimize(maximize='Retrun [%]')
    #output = bt.run()
    #print(output)
    ##print()
    ##print(output.keys())
    ##bt.plot(plot_width=1400)
    #print(output.keys()
    from tool import test
    performance,bt = test(r'checkpoint\1210_1827\winner/')
    print(performance._trades)