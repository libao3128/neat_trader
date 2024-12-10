import os
import pandas as pd
import yfinance as yf
def get_price(period, interval):
    file_name = 'data/price_sp500_'+interval+'_'+period+'.xlsx'
    if os.path.exists(file_name):
        price = pd.read_excel(file_name)
    else:
        sp500 = pd.read_excel('data/sp500.xlsx', header=2, index_col=0)
        price = yf.download(list(sp500['Symbol']), period=period, interval=interval)
        price.to_excel(file_name)
    return price

if __name__ == '__main__':
    price = get_price(period='20y', interval='1d')
    
