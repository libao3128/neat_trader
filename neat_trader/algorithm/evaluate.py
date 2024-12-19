from neat_trader.algorithm.fitness_fn import multi_objective_fitness_function_2
from neat_trader.utils.backtest import backtest
from neat_trader.utils.data_handler import DataHandler

import neat
import pandas as pd

# A class that do the evaluation on genome.
class Evaluater:
    def __init__(self, data_handler=None, backtesting_data=None, fitness_fn=None):
        if data_handler is None:
            self.data_handler = DataHandler()
        else:
            self.data_handler = data_handler
        
        self.backtesting_data = backtesting_data
        
        # set default fitness function if not assigned
        if fitness_fn is None:
            self.fitness_fn = multi_objective_fitness_function_2
        else:
            self.fitness_fn = fitness_fn
        
    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            performance, _ = backtest(net, self.backtesting_data)
            genome.fitness = self.fitness_fn(performance)
            print(genome.fitness)

    def eval_genome(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        if self.backtesting_data is None:
            performance, _ = backtest(net, self.data_handler.get_random_data())
        else:
            performance, _ = backtest(net, self.backtesting_data)
    
        return self.fitness_fn(performance)
