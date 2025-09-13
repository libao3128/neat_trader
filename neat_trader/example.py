"""
Example usage of the refactored NEAT Trader package.

This module demonstrates how to use the reorganized package structure
following Python package principles.
"""

import neat
from neat_trader import (
    NeatTrader, 
    Evaluator, 
    DataHandler, 
    CryptoDataHandler,
    multi_objective_fitness_function_2,
    get_config,
    NODE_NAMES
)
from neat_trader.exceptions import NEATTraderError, DataError


def main():
    """Demonstrate the refactored NEAT Trader package usage."""
    
    try:
        # Load NEAT configuration
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            'model/config-feedforward')
        
        # Initialize data handler
        data_handler = DataHandler('data/mydatabase.db')
        
        # Initialize evaluator with fitness function
        evaluator = Evaluator(
            data_handler=data_handler,
            fitness_fn=multi_objective_fitness_function_2
        )
        
        # Initialize NEAT trader
        trader = NeatTrader(config, evaluator)
        
        print("NEAT Trader initialized successfully!")
        print(f"Node names: {NODE_NAMES}")
        print(f"Configuration: {get_config()}")
        
        # Run evolution (commented out for demo)
        # winner = trader.evolve(total_generation=50, num_process=4)
        # trader.generate_report()
        
        print("Example completed successfully!")
        
    except DataError as e:
        print(f"Data error: {e}")
    except NEATTraderError as e:
        print(f"NEAT Trader error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
