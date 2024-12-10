from neat_trader.algorithm.fitness_fn import multi_objective_fitness_function_2
from neat_trader.utils.backtest import backtest

import neat
import sqlite3
import random, datetime
import pandas as pd
import numpy as np



# A class that do the evaluation on genome.
class Evaluater:
    def __init__(self, backtesting_data=None, fitness_fn=None):
        self.backtesting_data = backtesting_data
        
        # set default fitness function if not assigned
        if fitness_fn is None:
            self.fitness_fn = multi_objective_fitness_function_2
        
    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            performance, _ = backtest(net, self.backtesting_data)
            genome.fitness = self.fitness_fn(performance)
            print(genome.fitness)

    def eval_genome(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        if self.backtesting_data is None:
            performance, _ = backtest(net, self._generate_random_data())
        else:
            performance, _ = backtest(net, self.backtesting_data)
    
        return self.fitness_fn(performance)
        
    def _generate_random_data(self, is_test=False, num_date=365):
        
        con = sqlite3.connect('data/mydatabase.db')
        tickers = pd.read_sql('SELECT DISTINCT Ticker FROM Price', con)

        ticker = np.random.choice(tickers['Ticker'].tolist())
        dates = pd.read_sql('SELECT Datetime FROM Price WHERE Ticker="'+ticker+'"', con)
        dates = pd.to_datetime(dates['Datetime'])
        start_date = dates.max()
        end_date = dates.min()

        time_between_dates = start_date-end_date
        try:
            days_between_dates = int(time_between_dates.days-365-num_date-30)
            
            random_number_of_days = random.randrange(days_between_dates)
        except:
            return self._generate_random_data()
    
            
        
        random_date = end_date + datetime.timedelta(days=random_number_of_days)
        if not is_test:
            start_date = random_date
            end_date = start_date + datetime.timedelta(days=int(num_date+30))
        else:
            end_date = dates.max()
            start_date = dates.min()

        query = 'SELECT * FROM Price WHERE Ticker="'+ticker+'"'\
                +' AND Datetime>=DATETIME('+str(start_date.timestamp())+', "unixepoch")'\
                +' AND Datetime<=DATETIME('+str(end_date.timestamp())+', "unixepoch")'

        data =  pd.read_sql(query, con)
        data.index = pd.DatetimeIndex(data['Datetime']) 
        con.close()
        return data
