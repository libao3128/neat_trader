import os

import neat
import visualize
import backtesting
import pandas as pd
from backtest import NEAT_strategy, backtest
import pickle
import shutil
import time

import yfinance as yf

def fitness_fn(performance):
    #print(performance)
    ror = performance['Return [%]']
    #print(ror)
    return ror



def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        performance, _ = backtest(net, backtesting_data)
        genome.fitness = fitness_fn(performance)
        #print(genome.fitness)
        #print(performance)
class evaluate:
    
    def __init__(self, backtesting_data):
        self.backtesting_data = backtesting_data

    def eval_genome(self, genome, config):
        #print(genome)
        #global backtesting_data
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        performance, _ = backtest(net, self.backtesting_data)
        #genome.fitness = fitness_fn(performance)
        return fitness_fn(performance)
        #self.stop()

def run(config_file, evaluate, max_generation = 200):
    os.environ["PATH"] += os.pathsep + r'C:/Users/USER/OneDrive - 國立陽明交通大學/文件/academic/大四上/演化計算/Project/Graphviz/bin'
    

    t = time.localtime()
    time_str = time.strftime("%m%d_%H%M", t)
    

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    folder = 'checkpoint/'+time_str+'/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    p.add_reporter(neat.Checkpointer(5, filename_prefix=folder+'neat-checkpoint-'))

    # Run for up to 300 generations.
    pe = neat.ParallelEvaluator(20, evaluate.eval_genome)
    winner = p.run(pe.evaluate, max_generation)
    
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    performance, bt = backtest(winner_net, evaluate.backtesting_data)
    print(performance)
    bt.plot(plot_width=1500)


    node_names = {
        -1: 'position', 
        -2: 'slowk',
        -3: 'slowd',
        -4: 'macdhist',

        0: 'trade',
        1: 'volume'
        }
    if not os.path.exists(folder+'graph'):
        os.makedirs(folder+'graph')

    visualize.draw_net(config, winner, True
        , node_names=node_names,filename=folder+'graph/winner_net'
        )
    #visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True, filename=folder+'graph/avg_fitness.svg')
    visualize.plot_species(stats, view=True, filename=folder+'graph/speciation.svg')

    if not os.path.exists(folder+'winner'):
        os.makedirs(folder+'winner')
    with open(folder+'winner/winner.pkl', 'wb') as handle:
        pickle.dump(winner, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder+'winner/nodes_name.pkl', 'wb') as handle:
        pickle.dump(node_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #pickle.dump(node_names)
        
    shutil.copyfile(config_file, 'checkpoint/'+time_str+'/winner/config-feedforward')


def test( file_name):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         file_name+'config-feedforward')
    with open(file_name+'winner.pkl', 'rb') as f:
        winner = pickle.load(f)
    with open(file_name+'nodes_name.pkl', 'rb') as f:
        node_names = pickle.load(f)
    #winner = p.run(eval_genomes, 1)

    net = neat.nn.FeedForwardNetwork.create(winner, config)
    performance, bt = backtest(net, backtesting_data)
    print(performance)
    bt.plot(plot_width=1500)

if __name__ == '__main__':
    backtesting.set_bokeh_output(notebook=False)
    backtesting_data = yf.download('AAPL', interval='15m', period='60d')
    backtesting_data.index = pd.to_datetime(backtesting_data.index)
    
    run('config-feedforward', 5)
    #test('checkpoint/1128_1507\winner/' )