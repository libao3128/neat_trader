from typing import Optional, Callable, Any, Tuple
import neat
from backtesting import Backtest

from ..utils.data_handler import DataHandler
from .strategy import NEATStrategyBase
from ..exceptions import EvaluationError, BacktestError

class Evaluator:
    """
    A class to evaluate genomes using a specified fitness function and backtesting data.
    
    Attributes:
        data_handler (DataHandler): An instance of DataHandler to handle data operations.
        backtesting_data (Any): The data used for backtesting the genomes.
        fitness_fn (Callable): The fitness function used to evaluate the performance of genomes.
        StrategyBaseClass: The base strategy class for creating trading strategies.
    """
    
    def __init__(self, data_handler: Optional[DataHandler] = None, 
                 backtesting_data: Optional[Any] = None, 
                 fitness_fn: Optional[Callable] = None):
        """
        Initialize the Evaluator.
        
        Args:
            data_handler: DataHandler instance for data operations
            backtesting_data: Data for backtesting genomes
            fitness_fn: Fitness function for evaluating genome performance
        """
        self.data_handler = data_handler or DataHandler()
        self.backtesting_data = backtesting_data
        self.fitness_fn = fitness_fn
        
        # Import fitness function here to avoid circular imports
        if self.fitness_fn is None:
            from .fitness_fn import multi_objective_fitness_function_2
            self.fitness_fn = multi_objective_fitness_function_2
        
        self.StrategyBaseClass = NEATStrategyBase

    def eval_genomes(self, genomes: list, config: neat.Config) -> list:
        """
        Evaluate a list of genomes and assign fitness values.
        
        Args:
            genomes: List of (genome_id, genome) tuples
            config: NEAT configuration object
            
        Returns:
            List of fitness scores
        """
        fitness_scores = []
        
        for genome_id, genome in genomes:
            try:
                performance, _ = self.backtest(genome, config, self.backtesting_data)
                genome.fitness = self.fitness_fn(performance)
                fitness_scores.append(genome.fitness)
            except Exception as e:
                raise EvaluationError(f"Failed to evaluate genome {genome_id}: {str(e)}")
            
        return fitness_scores

    def eval_genome(self, genome: neat.DefaultGenome, config: neat.Config) -> float:
        """
        Evaluate a single genome and return its fitness value.
        
        Args:
            genome: NEAT genome to evaluate
            config: NEAT configuration object
            
        Returns:
            Fitness score of the genome
        """
        if self.backtesting_data is None:
            self.backtesting_data = self.data_handler.get_random_data()
        
        try:
            performance, _ = self.backtest(genome, config, self.backtesting_data)
            score = self.fitness_fn(performance)
            return score
        except Exception as e:
            raise EvaluationError(f"Failed to evaluate genome: {str(e)}")
    
    def backtest(self, genome: neat.DefaultGenome, config: neat.Config, data: Any) -> Tuple[Any, Backtest]:
        """
        Run backtest for a given genome.
        
        Args:
            genome: NEAT genome to test
            config: NEAT configuration object
            data: Market data for backtesting
            
        Returns:
            Tuple of (backtest results, backtest object)
        """
        try:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            neat_strategy = self._create_strategy_class(net)
            
            from ..config import DEFAULT_CASH, DEFAULT_COMMISSION
            bt = Backtest(data, neat_strategy, cash=DEFAULT_CASH, 
                         commission=DEFAULT_COMMISSION, exclusive_orders=False)
            output = bt.run()
            return output, bt
        except Exception as e:
            raise BacktestError(f"Backtest failed: {str(e)}")
    
    def _create_strategy_class(self, model: neat.nn.FeedForwardNetwork):
        """
        Create a custom strategy class with the given model.
        
        Args:
            model: NEAT neural network model
            
        Returns:
            Custom strategy class
        """
        class CustomNEATStrategy(self.StrategyBaseClass):
            def init(self):
                self.model = model  # Use external model dynamically
                super().init()

        return CustomNEATStrategy
