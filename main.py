import os

import neat
import visualize

from backtest import NEAT_strategy, backtest, SmaCross

import yfinance as yf

import pandas as pd
from tool import run, test, evaluate
from backtesting import Backtest, Strategy
import backtesting

if __name__ == '__main__':
    eval = evaluate(None)
    run('config-feedforward', eval, max_generation=200, num_process=2)