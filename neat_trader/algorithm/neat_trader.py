from neat_trader.utils.backtest import multi_process_backtest

import neat
from neat import Population, ParallelEvaluator, Checkpointer, StatisticsReporter, StdOutReporter
from neat.nn.feed_forward import FeedForwardNetwork
import pickle
import os, time
from tqdm import tqdm
import pandas as pd
from neat_trader.utils import visualize

class NeatTrader:
    """
    A class to represent a NEAT algorithm-based stock trading strategy.
    Attributes
    ----------
    config : object
        Configuration object for the NEAT algorithm.
    evaluater : object
        Evaluater object to evaluate genomes.
    checkpoint_path : str
        Path to save checkpoints and results.
    population : Population
        Population object representing the NEAT population.
    stats : object
        Statistics object to report the progress of the NEAT algorithm.
    winner : object
        The best genome found during the evolution process.
    Methods
    -------
    _initialize_folder_structure():
        Initializes the folder structure for saving checkpoints and results.
    evolve(trading_period_length=90, total_generation=200, num_process=16):
        Evolves the population over a specified number of generations.
    generate_report():
        Generates a report of the best genome and visualizes the results.
    initialize_population_from_checkpoint(checkpoint_path):
        Initializes the population from a checkpoint.
    node_names():
        Returns a dictionary mapping node IDs to their names.
    """
    def __init__(self, config_path, evaluator):
        self.config = self._load_configuration(config_path)
        self.evaluator = evaluator
        
        self.checkpoint_path = self._initialize_folder_structure()
        
        self.population = Population(self.config)
        self.stats = self._initialize_reporters()
        
        self.winner = None
        
    def _load_configuration(self, config_file_path):
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_file_path)
    
    def _initialize_folder_structure(self):
        time_str = time.strftime("%m%d_%H%M", time.localtime())
        folder = f'checkpoint/{time_str}/'
        if not os.path.exists(f'{folder}winner'):
            os.makedirs(f'{folder}winner')
        return folder
        
    def _initialize_reporters(self):
        self.population.add_reporter(StdOutReporter(True))
        stats = StatisticsReporter()
        self.population.add_reporter(stats)
        self.population.add_reporter(Checkpointer(5, filename_prefix=f'{self.checkpoint_path}neat-checkpoint-'))
        return stats
    
    def evolve(self, trading_period_length=90, total_generation=200, num_process=16):
        """
        Evolves the population of genomes over a specified number of generations.
        Args:
            trading_period_length (int, optional): The length of the trading period in days. Defaults to 90.
            total_generation (int, optional): The total number of generations for the evolution process. Defaults to 200.
            num_process (int, optional): The number of processes to use for parallel evaluation. Defaults to 16.
        Returns:
            Genome: The winning genome after the evolution process.
        Saves:
            The winning genome and node names to the specified checkpoint path.
        """

        if num_process > 1:
            for _ in range(int(total_generation / 5)):
                self.evaluator.backtesting_data = self.evaluator.data_handler.get_random_data(num_date=trading_period_length)
                pe = ParallelEvaluator(num_process, self.evaluator.eval_genome)
                winner = self.population.run(pe.evaluate, 5)
        else:
            winner = self.population.run(self.evaluator.eval_genomes, total_generation)
            
        self.winner = winner
            
        with open(f'{self.checkpoint_path}winner/winner.pkl', 'wb') as handle:
            pickle.dump(winner, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{self.checkpoint_path}winner/nodes_name.pkl', 'wb') as handle:
            pickle.dump(self.node_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        return winner
    
    def test_population(self, episode=10, date_length=90):
        """
        Tests the population by running multiple backtesting episodes.
    
        Args:
            episode (int, optional): The number of backtesting episodes to run. Defaults to 10.
            date_length (int, optional): The number of dates to use for backtesting data. Defaults to 90.
    
        Returns:
            tuple: A tuple containing the performances DataFrame and the backtesting data.
        """
        nets = [FeedForwardNetwork.create(individual, self.config) for individual in self.population.population.values()]
        performances = pd.DataFrame(columns=['individual_index', 'test_case', 'performance', 'bt'])
        backtesting_data = []
        
        for i in tqdm(range(episode), desc='Testing population'):
            data = self.evaluator.data_handler.get_random_data(num_date=date_length)
            performance, bts = multi_process_backtest(nets, data)
            performance_df = pd.DataFrame({
                'individual_index': list(self.population.population.keys()),
                'test_case': i,
                'performance': performance,
                'bt': bts
            })
            performances = pd.concat([performances, performance_df], ignore_index=True)
            backtesting_data.append(data)
            
        return performances, backtesting_data
    
    def generate_report(self):
        if self.winner is None:
            print('No winner found. Please run the evolve method first.')
            return
        winner = self.winner
        print(f'\nBest genome:\n{winner}')
        print('\nOutput:')
        if not os.path.exists(f'{self.checkpoint_path}graph'):
            os.makedirs(f'{self.checkpoint_path}graph')

        visualize.draw_net(self.config, winner, True, node_names=self.node_names, filename=f'{self.checkpoint_path}graph/winner_net')
        visualize.plot_stats(self.stats, ylog=False, view=True, filename=f'{self.checkpoint_path}graph/avg_fitness.svg')
        visualize.plot_species(self.stats, view=True, filename=f'{self.checkpoint_path}graph/speciation.svg')
    
    def initialize_population_from_checkpoint(self, checkpoint_path):
        self.population = Checkpointer.restore_checkpoint(checkpoint_path)
        
    @property
    def node_names(self):
        return {
            -1: 'long_position', -2: 'short_position', -3: 'sma5', -4: 'sma10',
            -5: 'slowk', -6: 'slowd', -7: 'macdhist_diff', -8: 'cci', -9: 'willr',
            -10: 'rsi', -11: 'adosc', 0: 'buy', 1: 'sell', 2: 'volume'
        }