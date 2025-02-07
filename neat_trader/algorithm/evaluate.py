from neat_trader.algorithm.fitness_fn import multi_objective_fitness_function_2
from neat_trader.utils.backtest import backtest
from neat_trader.utils.data_handler import DataHandler

import neat

class Evaluator:
    """
    A class to evaluate genomes using a specified fitness function and backtesting data.
    Attributes:
        data_handler (DataHandler): An instance of DataHandler to handle data operations.
        backtesting_data (Any): The data used for backtesting the genomes.
        fitness_fn (Callable): The fitness function used to evaluate the performance of genomes.
    Methods:
        eval_genomes(genomes, config):
            Evaluates a list of genomes and assigns fitness values to each genome.
        eval_genome(genome, config):
            Evaluates a single genome and returns its fitness value.
    """
    def __init__(self, config, data_handler=None, backtesting_data=None, fitness_fn=None):
        self.config = config
        self.data_handler = data_handler or DataHandler()
        self.backtesting_data = backtesting_data
        self.fitness_fn = fitness_fn or multi_objective_fitness_function_2

    def eval_genomes(self, genomes):
        fitness_scores = []
        
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, self.config)
            performance, _ = backtest(net, self.backtesting_data)
            genome.fitness = self.fitness_fn(performance)
            fitness_scores.append(genome.fitness)
            
        return fitness_scores

    def eval_genome(self, genome):
        net = neat.nn.FeedForwardNetwork.create(genome, self.config)
        if self.backtesting_data is None:
            self.backtesting_data = self.data_handler.get_random_data()
        performance, _ = backtest(net, self.backtesting_data)
        score = self.fitness_fn(performance)
        return score
