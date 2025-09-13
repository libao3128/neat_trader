"""
Main NEAT Trader implementation.

This module contains the core NeatTrader class that orchestrates the NEAT
algorithm evolution process for trading strategy optimization.
"""

import os
import time
import pickle
from typing import Optional, Dict, Any
import neat
from neat import Population, ParallelEvaluator, Checkpointer, StatisticsReporter, StdOutReporter
from neat.nn.feed_forward import FeedForwardNetwork

from ..utils import visualize
from ..config import DEFAULT_CHECKPOINT_DIR, DEFAULT_TRADING_PERIOD_LENGTH, DEFAULT_TOTAL_GENERATION, DEFAULT_NUM_PROCESS, NODE_NAMES
from ..exceptions import CheckpointError, VisualizationError

class NeatTrader:
    """
    Main class for NEAT algorithm-based stock trading strategy.
    
    This class orchestrates the entire NEAT evolution process, including
    population management, evolution, checkpointing, and reporting.
    
    Attributes:
        config: NEAT configuration object
        evaluator: Evaluator object for genome evaluation
        checkpoint_path: Path for saving checkpoints and results
        population: NEAT population object
        stats: Statistics reporter for evolution progress
        winner: Best genome found during evolution
    """
    
    def __init__(self, config: neat.Config, evaluator: Any):
        """
        Initialize NeatTrader.
        
        Args:
            config: NEAT configuration object
            evaluator: Evaluator instance for genome evaluation
        """
        self.config = config
        self.evaluator = evaluator
        
        self.checkpoint_path = self._initialize_folder_structure()
        
        self.population = Population(self.config)
        self.stats = self._initialize_reporters()
        
        self.winner: Optional[neat.DefaultGenome] = None
        
    def _initialize_folder_structure(self) -> str:
        """
        Initialize folder structure for checkpoints and results.
        
        Returns:
            Path to the checkpoint directory
        """
        time_str = time.strftime("%m%d_%H%M", time.localtime())
        folder = f'{DEFAULT_CHECKPOINT_DIR}/{time_str}/'
        winner_dir = f'{folder}winner'
        
        try:
            if not os.path.exists(winner_dir):
                os.makedirs(winner_dir)
        except OSError as e:
            raise CheckpointError(f"Failed to create checkpoint directory: {str(e)}")
            
        return folder
        
    def _initialize_reporters(self) -> StatisticsReporter:
        """
        Initialize NEAT reporters for statistics and checkpointing.
        
        Returns:
            StatisticsReporter instance
        """
        self.population.add_reporter(StdOutReporter(True))
        stats = StatisticsReporter()
        self.population.add_reporter(stats)
        self.population.add_reporter(Checkpointer(5, filename_prefix=f'{self.checkpoint_path}neat-checkpoint-'))
        return stats
    
    def evolve(self, trading_period_length: int = DEFAULT_TRADING_PERIOD_LENGTH, 
               total_generation: int = DEFAULT_TOTAL_GENERATION, 
               num_process: int = DEFAULT_NUM_PROCESS) -> neat.DefaultGenome:
        """
        Evolve the population of genomes over specified generations.
        
        Args:
            trading_period_length: Length of trading period in days
            total_generation: Total number of generations for evolution
            num_process: Number of processes for parallel evaluation
            
        Returns:
            The winning genome after evolution
            
        Raises:
            CheckpointError: If saving results fails
        """
        try:
            if num_process > 1:
                # Parallel evolution with periodic data refresh
                for _ in range(int(total_generation / 5)):
                    self.evaluator.backtesting_data = self.evaluator.data_handler.get_random_data(data_length=trading_period_length)
                    pe = ParallelEvaluator(num_process, self.evaluator.eval_genome)
                    winner = self.population.run(pe.evaluate, 5)
            else:
                # Sequential evolution
                self.evaluator.backtesting_data = self.evaluator.data_handler.get_random_data(data_length=trading_period_length)
                winner = self.population.run(self.evaluator.eval_genomes, total_generation)
                
            self.winner = winner
            
            # Save results
            self._save_results(winner)
            
            return winner
            
        except Exception as e:
            raise CheckpointError(f"Evolution failed: {str(e)}")
    
    def _save_results(self, winner: neat.DefaultGenome) -> None:
        """
        Save winning genome and node names to checkpoint directory.
        
        Args:
            winner: The winning genome to save
        """
        try:
            with open(f'{self.checkpoint_path}winner/winner.pkl', 'wb') as handle:
                pickle.dump(winner, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f'{self.checkpoint_path}winner/nodes_name.pkl', 'wb') as handle:
                pickle.dump(self.node_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise CheckpointError(f"Failed to save results: {str(e)}")
       
    def generate_report(self) -> None:
        """
        Generate comprehensive report of the best genome and evolution statistics.
        
        Creates visualizations including:
        - Neural network topology
        - Fitness evolution over generations
        - Species evolution
        
        Raises:
            VisualizationError: If visualization generation fails
        """
        if self.winner is None:
            print('No winner found. Please run the evolve method first.')
            return
            
        winner = self.winner
        print(f'\nBest genome:\n{winner}')
        print('\nGenerating visualizations...')
        
        try:
            # Create graph directory
            graph_dir = f'{self.checkpoint_path}graph'
            if not os.path.exists(graph_dir):
                os.makedirs(graph_dir)

            # Generate visualizations
            visualize.draw_net(self.config, winner, True, 
                            node_names=self.node_names, 
                            filename=f'{graph_dir}/winner_net')
            visualize.plot_stats(self.stats, ylog=False, view=True, 
                              filename=f'{graph_dir}/avg_fitness.svg')
            visualize.plot_species(self.stats, view=True, 
                                filename=f'{graph_dir}/speciation.svg')
                                
            print(f'Report generated successfully in {self.checkpoint_path}')
            
        except Exception as e:
            raise VisualizationError(f"Failed to generate report: {str(e)}")
    
    def initialize_population_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Initialize population from a saved checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Raises:
            CheckpointError: If checkpoint restoration fails
        """
        try:
            self.population = Checkpointer.restore_checkpoint(checkpoint_path)
        except Exception as e:
            raise CheckpointError(f"Failed to restore checkpoint: {str(e)}")
        
    @property
    def node_names(self) -> Dict[int, str]:
        """
        Get node names mapping for NEAT networks.
        
        Returns:
            Dictionary mapping node IDs to descriptive names
        """
        return NODE_NAMES