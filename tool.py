import os

import neat
import visualize
import backtesting
import pandas as pd
from backtest import NEAT_strategy, backtest
import pickle
import shutil
import time
import sqlite3
import numpy as np
import random
import datetime
import io

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
        if backtesting_data is None:
            self.random_generate = True
        else:
            self.random_generate = False
        self.cnt = 0
        

    def eval_genome(self, genome, config):
        #print(genome)
        self.cnt += 1
        print(self.cnt)
        #global backtesting_data
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        if self.backtesting_data is None:
            performance, _ = backtest(net, self.generate_random_data())
        else:
            performance, _ = backtest(net, self.backtesting_data)
        #genome.fitness = fitness_fn(performance)
        return fitness_fn(performance)
        #self.stop()

    def generate_random_data(self):
        con = sqlite3.connect('mydatabase.db')
        tickers = pd.read_sql('SELECT DISTINCT Ticker FROM Price', con)

        ticker = np.random.choice(tickers['Ticker'].tolist())
        dates = pd.read_sql('SELECT Datetime FROM Price WHERE Ticker="'+ticker+'"', con)
        dates = pd.to_datetime(dates['Datetime'])
        start_date = dates.max()
        end_date = dates.min()

        #
        time_between_dates = start_date-end_date
        try:
            days_between_dates = int(time_between_dates.days-365*2.5)
            random_number_of_days = random.randrange(days_between_dates)
        except:
            return self.generate_random_data()
    
            
        
        random_date = end_date + datetime.timedelta(days=random_number_of_days)

        start_date = random_date
        end_date = start_date + datetime.timedelta(days=int(365*1.5))

        data =  pd.read_sql(
            'SELECT * FROM Price WHERE Ticker="'+ticker+'"'
            +' AND Datetime>=DATETIME('+str(start_date.timestamp())+', "unixepoch")'
            +' AND Datetime<=DATETIME('+str(end_date.timestamp())+', "unixepoch")'
            , con)

        con.close()
        return data

    def read_sql_inmem_uncompressed(self, query, conn):
        copy_sql = "COPY ({query}) TO STDOUT WITH CSV {head}".format(
           query=query, head="HEADER"
        )
        
        cur = conn.cursor()
        store = io.StringIO()
        cur.copy_expert(copy_sql, store)
        store.seek(0)
        df = pd.read_csv(store)
        return df

def run(config_file, evaluate, max_generation = 200):
    os.environ["PATH"] += os.pathsep + r'C:/Users/USER/OneDrive - 國立陽明交通大學/文件/academic/大四上/演化計算/Project/Graphviz/bin'
    shutil.copyfile(config_file, 'checkpoint/'+time_str+'/winner/config-feedforward')

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
    test(folder+'winner/')
    


def test( file_name, backtesting_data=None):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         file_name+'config-feedforward')
    with open(file_name+'winner.pkl', 'rb') as f:
        winner = pickle.load(f)
    with open(file_name+'nodes_name.pkl', 'rb') as f:
        node_names = pickle.load(f)
    #winner = p.run(eval_genomes, 1)
    if backtesting_data is None:
        backtesting_data=evaluate.generate_random_data(None)
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    performance, bt = backtest(net, backtesting_data)
    print(performance)
    bt.plot(plot_width=1500, filename=file_name+'/Performance_{}.html'.format(backtesting_data['Ticker'][0]))

if __name__ == '__main__':
    from tool import run, test, evaluate
    #eval = evaluate(None)
    #run('config-feedforward', eval, max_generation=200)
    #backtesting.set_bokeh_output(notebook=False)
    #backtesting_data = yf.download('AAPL', interval='15m', period='60d')
    #backtesting_data.index = pd.to_datetime(backtesting_data.index)

    #evaluate(None).generate_random_data()
    test('checkpoint/1129_0032\winner/')
    #run('config-feedforward', 5)
    #test('checkpoint/1128_1507\winner/' )
    pass