# NEAT Trader Package

A professionally refactored Python package implementing NEAT (NeuroEvolution of Augmenting Topologies) algorithm for automated stock trading strategies.

## Package Structure

The package follows Python package principles with clear separation of concerns:

```
neat_trader/
├── __init__.py              # Main package exports
├── config.py                # Configuration and constants
├── exceptions.py            # Custom exception classes
├── example.py               # Usage examples
├── algorithm/               # Core NEAT implementation
│   ├── __init__.py
│   ├── neat_trader.py       # Main NeatTrader class
│   ├── evaluate.py          # Genome evaluation logic
│   ├── fitness_fn.py        # Fitness functions
│   └── strategy.py           # Trading strategy base class
└── utils/                   # Utility modules
    ├── __init__.py
    ├── data_handler.py      # Data handling utilities
    └── visualize.py         # Visualization functions
```

## Key Improvements

### 1. **Proper Package Structure**
- Clear module organization with logical subpackages
- Proper `__init__.py` files with controlled exports
- Separation of concerns between algorithm, utilities, and configuration

### 2. **Type Hints**
- Comprehensive type annotations throughout the codebase
- Better IDE support and code documentation
- Improved maintainability and debugging

### 3. **Error Handling**
- Custom exception hierarchy for better error management
- Graceful error handling with informative messages
- Proper exception propagation

### 4. **Configuration Management**
- Centralized configuration in `config.py`
- Easy customization of parameters
- Environment-specific settings

### 5. **Documentation**
- Comprehensive docstrings for all classes and functions
- Module-level documentation
- Usage examples and API documentation

### 6. **Import Structure**
- Eliminated circular imports
- Clean, hierarchical import structure
- Proper dependency management

## Usage

### Basic Usage

```python
import neat
from neat_trader import NeatTrader, Evaluator, DataHandler, multi_objective_fitness_function_2

# Load configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    'model/config-feedforward')

# Initialize components
data_handler = DataHandler('data/mydatabase.db')
evaluator = Evaluator(data_handler=data_handler, fitness_fn=multi_objective_fitness_function_2)
trader = NeatTrader(config, evaluator)

# Run evolution
winner = trader.evolve(total_generation=100)
trader.generate_report()
```

### Advanced Usage

```python
from neat_trader import get_config, NODE_NAMES
from neat_trader.exceptions import DataError, EvaluationError

# Access configuration
config_dict = get_config()
print(f"Node names: {NODE_NAMES}")

# Error handling
try:
    data = data_handler.get_random_data(data_length=365)
except DataError as e:
    print(f"Data error: {e}")
```

## Configuration

The package uses a centralized configuration system:

```python
from neat_trader.config import get_config

config = get_config()
# Access trading parameters, fitness weights, risk thresholds, etc.
```

## Fitness Functions

Multiple fitness functions are available:

- `multi_objective_fitness_function_2`: Advanced multi-objective function
- `multi_objective_fitness_function_1`: Discrete scoring system
- `outperform_benchmark`: Benchmark-focused function
- `gpt_fitness_fn`: Sharpe ratio optimization
- `crypto_fitness_fn`: Cryptocurrency-optimized function

## Data Handlers

- `DataHandler`: Traditional stock data from SQLite
- `CryptoDataHandler`: Cryptocurrency data from Binance database

## Visualization

Comprehensive visualization tools:

- `plot_stats()`: Evolution statistics
- `plot_species()`: Species evolution
- `draw_net()`: Neural network topology
- `plot_spikes()`: Spiking neuron visualization

## Error Handling

Custom exception hierarchy:

- `NEATTraderError`: Base exception
- `DataError`: Data-related errors
- `EvaluationError`: Evaluation failures
- `VisualizationError`: Visualization issues
- `CheckpointError`: Checkpoint operations

## Dependencies

- `neat-python`: NEAT algorithm implementation
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `matplotlib`: Visualization
- `graphviz`: Network visualization
- `backtesting`: Backtesting framework
- `talib`: Technical analysis library

## Migration from Old Structure

The refactored package maintains backward compatibility while providing a cleaner API:

```python
# Old way (still works)
from neat_trader.algorithm.neat_trader import NeatTrader
from neat_trader.algorithm.evaluate import Evaluator

# New way (recommended)
from neat_trader import NeatTrader, Evaluator
```

## Contributing

When contributing to this package:

1. Follow the established package structure
2. Add type hints to all new functions
3. Include comprehensive docstrings
4. Use the custom exception classes
5. Update configuration in `config.py` for new constants
6. Add tests for new functionality

## License

This package is part of the NEAT Algorithm-based Stock Trading Strategy project.
