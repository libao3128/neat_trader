# NEAT Algorithm-based Stock Trading Strategy

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NEAT-Python](https://img.shields.io/badge/NEAT-python--neat-green.svg)](https://github.com/CodeReclaimers/neat-python)

A sophisticated Python package implementing **NEAT (NeuroEvolution of Augmenting Topologies)** algorithm for automated stock trading strategies using multiple technical indicators. This project combines evolutionary computation with financial markets to evolve neural network-based trading strategies.

## 🚀 Features

### Core Capabilities
- **NEAT Algorithm Implementation**: Evolves neural network topologies for trading decisions
- **Multiple Technical Indicators**: Integrates 11+ technical indicators (SMA, RSI, MACD, CCI, Williams %R, etc.)
- **Multi-Objective Fitness Functions**: Advanced evaluation metrics including Sharpe ratio, drawdown, and return optimization
- **Dual Market Support**: Works with both traditional stocks (SQLite) and cryptocurrency (Binance) data
- **Comprehensive Backtesting**: Full backtesting integration with performance analytics
- **Professional Package Structure**: Clean, modular design following Python best practices

### Advanced Features
- **Parallel Processing**: Multi-process evolution for faster training
- **Checkpoint System**: Save and resume training sessions
- **Visualization Tools**: Network topology visualization and performance plotting
- **Risk Management**: Built-in risk thresholds and position sizing
- **Type Safety**: Comprehensive type hints throughout the codebase
- **Error Handling**: Robust exception handling with custom error classes

## 📊 Technical Indicators

The system integrates multiple technical indicators as neural network inputs:

| Indicator | Description | Node ID |
|-----------|-------------|---------|
| SMA5 | 5-period Simple Moving Average | -3 |
| SMA10 | 10-period Simple Moving Average | -4 |
| SlowK | Stochastic Oscillator %K | -5 |
| SlowD | Stochastic Oscillator %D | -6 |
| MACD Histogram | MACD Histogram Difference | -7 |
| CCI | Commodity Channel Index | -8 |
| Williams %R | Williams Percent Range | -9 |
| RSI | Relative Strength Index | -10 |
| ADOSC | Accumulation/Distribution Oscillator | -11 |
| Long Position | Current long position state | -1 |
| Short Position | Current short position state | -2 |

**Outputs:**
- Buy Signal (Node 0)
- Sell Signal (Node 1) 
- Volume Signal (Node 2)

## 🛠 Installation

### Prerequisites

1. **Python 3.7+** - [Download Python](https://www.python.org/downloads/)
2. **TA-Lib** - [Install here](https://github.com/TA-Lib/ta-lib-python.git)
3. **Graphviz** - For network visualization
   - [Windows](https://graphviz.org/download/)
   - [Linux](https://graphviz.org/download/)
   - [Mac](https://graphviz.org/download/)

### Package Installation

```bash
# Clone the repository
git clone https://github.com/libao3128/NEAT-Algorithm-based-Stock-Trading-Strategy.git
cd NEAT-Algorithm-based-Stock-Trading-Strategy

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

```python
import neat
from neat_trader import NeatTrader, Evaluator, DataHandler, multi_objective_fitness_function_2

# Load NEAT configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    'model/config-feedforward')

# Initialize components
data_handler = DataHandler('data/mydatabase.db')
evaluator = Evaluator(data_handler=data_handler, fitness_fn=multi_objective_fitness_function_2)
trader = NeatTrader(config, evaluator)

# Run evolution
winner = trader.evolve(total_generation=100, num_process=4)
trader.generate_report()
```

### Advanced Usage
See [neat_trading_strategy_development.ipynb] for more example.

## 📁 Project Structure

```
NEAT-Algorithm-based-Stock-Trading-Strategy/
├── neat_trader/                 # Main package
│   ├── __init__.py             # Package exports
│   ├── config.py               # Configuration and constants
│   ├── exceptions.py           # Custom exception classes
│   ├── example.py              # Usage examples
│   ├── algorithm/              # Core NEAT implementation
│   │   ├── neat_trader.py      # Main NeatTrader class
│   │   ├── evaluate.py         # Genome evaluation logic
│   │   ├── fitness_fn.py       # Fitness functions
│   │   └── strategy.py         # Trading strategy base class
│   └── utils/                  # Utility modules
│       ├── data_handler.py     # Data handling utilities
│       └── visualize.py        # Visualization functions
├── model/                      # NEAT configuration files
│   ├── config-feedforward      # Main configuration
│   └── config-test            # Test configuration
├── tests/                      # Test suite
└── README.md                  # This file
```

## ⚙️ Configuration

### NEAT Configuration

The main configuration file `model/config-feedforward` controls:

- **Population Size**: 256 individuals
- **Input Nodes**: 11 technical indicators
- **Output Nodes**: 3 trading signals
- **Activation Functions**: ReLU, Sigmoid
- **Mutation Rates**: Connection (80%), Node (50%), Weight (80%)
- **Compatibility Threshold**: 3.0

### Trading Parameters

```python
from neat_trader.config import get_config

config = get_config()
# Access trading parameters, fitness weights, risk thresholds
```

Key parameters:
- **Initial Cash**: $1,000,000
- **Commission**: 0.2%
- **Trading Period**: 90 days
- **Risk Thresholds**: Configurable drawdown limits
- **Fitness Weights**: Multi-objective optimization weights

## 🧪 Fitness Functions

### Available Fitness Functions

1. **`multi_objective_fitness_function_2`** (Recommended)
   - Advanced multi-objective optimization
   - Considers return, Sharpe ratio, and drawdown
   - Weighted scoring system

2. **`multi_objective_fitness_function_1`**
   - Discrete scoring system
   - Return vs. buy-and-hold comparison
   - Drawdown penalties

3. **`outperform_benchmark`**
   - Benchmark-focused optimization
   - Market outperformance tracking

4. **`gpt_fitness_fn`**
   - Sharpe ratio optimization
   - Risk-adjusted returns

### Custom Fitness Functions

```python
def custom_fitness(performance: pd.DataFrame) -> float:
    """Custom fitness function example."""
    total_return = performance['Return [%]'].iloc[-1]
    max_drawdown = abs(performance['Drawdown [%]'].min())
    sharpe_ratio = performance['Sharpe Ratio'].iloc[-1]
    
    # Custom scoring logic
    score = total_return * 0.5 + sharpe_ratio * 0.3 - max_drawdown * 0.2
    return max(0, score)  # Ensure non-negative
```

## 📊 Data Handling

### Stock Data (SQLite)

```python
from neat_trader import DataHandler

# Initialize with stock data
data_handler = DataHandler('data/mydatabase.db')

# Get random data for training
data = data_handler.get_random_data(data_length=365)

# Get specific symbol data
aapl_data = data_handler.get_data_by_symbol('AAPL')
```

## 📈 Performance & Results

### Evolution Statistics

The system tracks multiple performance metrics:

- **Fitness Evolution**: Population fitness over generations
- **Species Diversity**: Number of species and their fitness
- **Network Complexity**: Node and connection counts
- **Trading Performance**: Returns, Sharpe ratio, drawdown

### Visualization

```python
from neat_trader.utils import plot_stats, plot_species, draw_net

# Plot evolution statistics
plot_stats(stats)

# Plot species evolution
plot_species(stats)

# Draw winning network
draw_net(config, winner)
```

### Sample Results

Typical evolution results:
- **Training Generations**: 100-200
- **Population Size**: 256 individuals
- **Best Fitness**: 50-150 (varies by market conditions)
- **Network Complexity**: 10-50 nodes, 20-100 connections
- **Trading Performance**: 5-25% annual returns (backtested)

## 🔧 Customization

### Adding New Technical Indicators

```python
# 1. Update NODE_NAMES in config.py
NODE_NAMES[-12] = 'new_indicator'

# 2. Update NEAT config file
num_inputs = 12  # Increase from 11 to 12

# 3. Implement indicator in data handler
def calculate_new_indicator(self, data):
    # Your indicator calculation
    return indicator_values
```

### Custom Trading Strategies

```python
from neat_trader.algorithm import NEATStrategyBase

class CustomStrategy(NEATStrategyBase):
    def __init__(self, genome, config):
        super().__init__(genome, config)
        # Custom initialization
    
    def calculate_signals(self, data):
        # Custom signal calculation
        return buy_signal, sell_signal, volume_signal
```

## 🚨 Error Handling

### Custom Exceptions

```python
from neat_trader.exceptions import (
    NEATTraderError, DataError, EvaluationError, 
    VisualizationError, CheckpointError
)

try:
    winner = trader.evolve(total_generation=100)
except DataError as e:
    print(f"Data issue: {e}")
except EvaluationError as e:
    print(f"Evaluation failed: {e}")
except NEATTraderError as e:
    print(f"General error: {e}")
```

## 📚 API Reference

### Core Classes

#### `NeatTrader`
Main class for running NEAT evolution.

```python
trader = NeatTrader(config, evaluator)
winner = trader.evolve(total_generation=100, num_process=4)
trader.generate_report()
```

#### `Evaluator`
Evaluates genome fitness using backtesting.

```python
evaluator = Evaluator(data_handler=data_handler, fitness_fn=fitness_function)
fitness = evaluator.evaluate_genome(genome)
```

#### `DataHandler`
Handles data loading and preprocessing.

```python
data_handler = DataHandler('data/mydatabase.db')
data = data_handler.get_random_data(data_length=365)
```

### Configuration Functions

#### `get_config()`
Returns current configuration dictionary.

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Follow the package structure**: Maintain the modular design
4. **Add type hints**: Include comprehensive type annotations
5. **Write tests**: Add tests for new functionality
6. **Update documentation**: Update README and docstrings
7. **Submit a pull request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/NEAT-Algorithm-based-Stock-Trading-Strategy.git

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov

# Run tests
python run_tests.py --all

# Check code coverage
pytest --cov=neat_trader tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **NEAT-Python**: Core NEAT algorithm implementation
- **TA-Lib**: Technical analysis indicators
- **Backtesting**: Backtesting framework
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib & Graphviz**: Visualization tools

## Disclaimer

This software is for educational and research purposes only. It is not intended for live trading without proper testing and risk management. Past performance does not guarantee future results. Always do your own research and consider consulting with financial professionals before making investment decisions.

---

**Happy Trading with NEAT! 🧬📈**