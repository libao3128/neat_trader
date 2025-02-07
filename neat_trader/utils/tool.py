import os
import pickle
import shutil
import time
from multiprocessing import Pool

import neat
from neat.nn.feed_forward import FeedForwardNetwork
import pandas as pd
from tqdm import tqdm

from neat_trader.utils import visualize
from neat_trader.utils.backtest import backtest, multi_process_backtest
from neat_trader.algorithm.evaluate import Evaluater
from neat_trader.utils.data_handler import DataHandler
from neat_trader.neat.tools import load_configuration, initialize_population, add_reporters

def create_folder_structure(time_str):
    folder = f'checkpoint/{time_str}/'
    if not os.path.exists(f'{folder}winner'):
        os.makedirs(f'{folder}winner')
    return folder

def run_experiment(config, evaluater, max_generation=200, num_date=90, num_process=16, checkpoint_path=None):
    time_str = time.strftime("%m%d_%H%M", time.localtime())
    folder = create_folder_structure(time_str)
    config.save(f'{folder}winner/config-feedforward')

    population = initialize_population(config, checkpoint_path)
    stats = add_reporters(population, folder)

    if __name__ == '__main__':
        winner = population.run(evaluater.eval_genomes, max_generation)
    else:
        for _ in range(int(max_generation / 5)):
            evaluater.backtesting_data = evaluater.data_handler.get_random_data(num_date=num_date)
            pe = neat.ParallelEvaluator(num_process, evaluater.eval_genome)
            winner = population.run(pe.evaluate, 5)

    print(f'\nBest genome:\n{winner}')
    print('\nOutput:')
    node_names = {
        -1: 'long_position', -2: 'short_position', -3: 'sma5', -4: 'sma10',
        -5: 'slowk', -6: 'slowd', -7: 'macdhist_diff', -8: 'cci', -9: 'willr',
        -10: 'rsi', -11: 'adosc', 0: 'buy', 1: 'sell', 2: 'volume'
    }
    if not os.path.exists(f'{folder}graph'):
        os.makedirs(f'{folder}graph')

    visualize.draw_net(config, winner, True, node_names=node_names, filename=f'{folder}graph/winner_net')
    visualize.plot_stats(stats, ylog=False, view=True, filename=f'{folder}graph/avg_fitness.svg')
    visualize.plot_species(stats, view=True, filename=f'{folder}graph/speciation.svg')

    with open(f'{folder}winner/winner.pkl', 'wb') as handle:
        pickle.dump(winner, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{folder}winner/nodes_name.pkl', 'wb') as handle:
        pickle.dump(node_names, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return population, winner, stats

def test_population(population, config, evaluater, episode=10, date_length=90):
    nets = [FeedForwardNetwork.create(individual, config) for individual in population.population.values()]
    performances = pd.DataFrame(columns=['individual_index', 'test_case', 'performance', 'bt'])
    backtesting_data = []

    for i in tqdm(range(episode), desc='Generating backtesting data'):
        data = evaluater.data_handler.get_random_data(num_date=date_length)
        performance, bts = multi_process_backtest(nets, data)
        performance_df = pd.DataFrame({
            'individual_index': list(population.population.keys()),
            'test_case': i,
            'performance': performance,
            'bt': bts
        })
        performances = pd.concat([performances, performance_df])
        backtesting_data.append(data)

    return performances, backtesting_data


if __name__ == '__main__':
    performances = test_population('VERY_GOOD', 1979, 20, 365)
    pass