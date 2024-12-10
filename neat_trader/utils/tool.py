import os
from neat_trader.utils import visualize

import neat
import pandas as pd
from neat_trader.utils.backtest import backtest
import pickle
import shutil
import time
from tqdm import tqdm

from multiprocessing import Pool

def run(config_file_path, evaluater, max_generation = 200, num_date=90, num_process = 16, checkpoint_path=None) -> neat.Population:
    '''
    The script to run the experiment
    Input:
        config_file_path: The path of config file of NEAT algorithm
        evaluater: The evaluater to eavluate the performance of genome
        max_generation: The maximum generation to evolve the population
        num_process: The number of process to run the training step in multi processing
        checkpoint_path: The path to the ckeckpoint to start the training
    Return:
        The trained population in last generation
    '''    
    t = time.localtime()
    time_str = time.strftime("%m%d_%H%M", t)
    folder = 'checkpoint/'+time_str+'/'

    if not os.path.exists(folder+'winner'):
        os.makedirs(folder+'winner')
    
    shutil.copyfile(config_file_path, 'checkpoint/'+time_str+'/winner/config-feedforward')
    
    
    

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file_path)

    # Create the population, which is the top-level object for a NEAT run.
    if checkpoint_path is None:
        p = neat.Population(config)
    else:
        p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    p.add_reporter(neat.Checkpointer(5, filename_prefix=folder+'neat-checkpoint-'))

    # Run for up to 300 generations.
    if __name__ == '__main__':
        winner = p.run(evaluater.eval_genomes, max_generation)
    else:
        for i in range(int(max_generation/5)):
            evaluater.backtesting_data = evaluater._generate_random_data(num_date=num_date)
            pe = neat.ParallelEvaluator(num_process, evaluater.eval_genome)
            winner = p.run(pe.evaluate, 5)
    
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    
    node_names = {
        -1: 'long_position', 
        -2: 'short_position',
        -3: 'sma5',
        -4: 'sma10',
        -5: 'slowk',
        -6: 'slowd',
        -7: 'macdhist_diff',
        -8: 'cci',
        -9: 'willr',
        -10: 'rsi',
        -11: 'adosc',
    
        0: 'buy',
        1: 'sell',
        
        2: 'volume'
        }
    if not os.path.exists(folder+'graph'):
        os.makedirs(folder+'graph')

    visualize.draw_net(config, winner, True
        , node_names=node_names,filename=folder+'graph/winner_net'
        )
    #visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True, filename=folder+'graph/avg_fitness.svg')
    visualize.plot_species(stats, view=True, filename=folder+'graph/speciation.svg')

    with open(folder+'winner/winner.pkl', 'wb') as handle:
        pickle.dump(winner, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder+'winner/nodes_name.pkl', 'wb') as handle:
        pickle.dump(node_names, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return p

def test( file_name, backtesting_data=None, plot=True):
    from neat_trader.algorithm.evaluate import Evaluater
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         file_name+'config-feedforward')
    with open(file_name+'winner.pkl', 'rb') as f:
        winner = pickle.load(f)
    with open(file_name+'nodes_name.pkl', 'rb') as f:
        node_names = pickle.load(f)
    #winner = p.run(eval_genomes, 1)
    if backtesting_data is None:
        backtesting_data=Evaluater._generate_random_data(is_test=True)
    #print(backtesting_data)
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    performance, bt = backtest(net, backtesting_data)
    if plot:
        visualize.draw_net(config, winner, True
        , node_names=node_names
        )
        print(performance)
        bt.plot(plot_width=1500, filename=file_name+'/Performance_{}.html'.format(backtesting_data['Ticker'][0]))
    return performance, bt

def task(input):
    net, ind, eval, i = input
    
    perfor, bt = backtest(net, eval.backtesting_data[i])
    perfor['individual_index'] = ind
    perfor['test_case'] = i
    perfor['bt'] = bt
    
    return perfor

def test_population(checkpoint_folder_path, checkpoint, episode=10, date_length=90):
    from neat_trader.algorithm.evaluate import Evaluater
    
    checkpoint_path = os.path.join(checkpoint_path, f'neat-checkpoint-{checkpoint}')
    config_path = os.path.join(checkpoint_folder_path, 'winner', 'config-feedforward')

    eval = Evaluater()
    p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    config = neat.Config(   
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path)
    

    eval.backtesting_data = []
    for _ in tqdm(range(episode)):
        eval.backtesting_data.append(eval.generate_random_data(num_date=date_length))
    performances = pd.DataFrame()
    input = []
    for ind in tqdm(p.population) :
        net = neat.nn.RecurrentNetwork.create(p.population[ind],config)
        for i in range(episode):
            input.append((net, ind, eval, i))
    #        task(input[-1])
    with Pool(16) as pool:
        output=pool.map(task, input)
    

    
    performances = pd.DataFrame(output)

    
            
    
    return performances
    
def task2(input):
    net,num_date = input
    
    perfor, bt = backtest(net, evaluate().generate_random_data(num_date = num_date))
    
    return perfor, bt

def test_individual(fold, checkpoint, individual_id,  episode=10, date_length=90):
    checkpoint_path = f'checkpoint/{fold}/neat-checkpoint-{checkpoint}'
    config_path = f'checkpoint\{fold}\winner/'+'config-feedforward'


    eval = evaluate()
    p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    config = neat.Config(   
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path)

    individual = p.population[individual_id]
    bts=[]

    draw_net(fold, individual)
    eval.backtesting_data = []

    
    
    net = neat.nn.RecurrentNetwork.create(individual,config)
    
    with Pool(16) as pool:
        output = pool.map(task2, [(net,date_length)]*episode)  
    for i, (perfor, bt) in enumerate(output):
        bts.append(bt)
        if i == 0:
            performances = perfor
        else:
            performances = pd.concat([performances,perfor],axis=1)
    performances = performances.transpose()
    performances['relative return']=performances['Return [%]']-performances['Buy & Hold Return [%]']
    performances['relative return']=pd.Series(performances['relative return'], dtype=float)
    return performances, bts
 
def draw_net(fold, genomes):
    node_names = {
        -1: 'long_position', 
        -2: 'short_position',
        -3: 'sma5',
        -4: 'sma10',
        -5: 'slowk',
        -6: 'slowd',
        -7: 'macdhist_diff',
        -8: 'cci',
        -9: 'willr',
        -10: 'rsi',
        -11: 'adosc',
    
        0: 'buy',
        1: 'sell',
        
        2: 'volume'
        }
    config_path = f'checkpoint\{fold}\winner/'+'config-feedforward'
    config = neat.Config(   
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path)

    #node_name_path = f'checkpoint\{fold}\winner/'+'nodes_name.pkl'
    #with open(node_name_path) as f:
    #    node_names = pickle.load(f)

    visualize.draw_net(config, genomes, node_names=node_names, view=True)


if __name__ == '__main__':
    #performances = test_population('VERY_GOOD', 1979, 100, 365)
    '''
    config = neat.Config(   
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        'config-test')
    node_names = {
        -1: 'long_position', 
        -2: 'short_position',
        -3: 'sma5',
        -4: 'sma10',
        -5: 'slowk',
        -6: 'slowd',
        -7: 'macdhist_diff',
        -8: 'cci',
        -9: 'willr',
        -10: 'rsi',
        -11: 'adosc',
    
        0: 'buy',
        1: 'sell',
        
        2: 'volume'
        }
    p = neat.Population(config)
    print(p.population)
    
    '''
    
    performances = test_individual('VERY_GOOD', 1979,23554, 1, 365)
    pass