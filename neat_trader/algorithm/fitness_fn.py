import pandas as pd
import datetime

def sample_fitness(performance: pd.DataFrame):
    # The fitness function should based on backtesting report and return a score to
    # evaluate trading strategy performance.

    score = 0
    
    return score

def SQN(performance: pd.DataFrame):
    # A fitness function based on System Quality Number(SQN).
    return performance['SQN']

def multi_objective_fitness_function_1(performance: pd.DataFrame):
    # Calculate the fitness socre based on the following equation
    # Fitness = 𝑃𝑛𝐿+ 1.5×𝑃𝑛𝐿𝑟𝑒𝑙𝑎𝑡𝑖𝑣𝑒 − 0.5×𝑚𝑎𝑥(𝑑𝑟𝑎𝑤𝑑𝑜𝑤𝑛)
    
    fitness = 0

    # RoR reward
    ror = performance['Return [%]']
    if ror>0:
        fitness+=1
    
    # Relative RoR reward
    bh_ror = performance['Buy & Hold Return [%]']
    relative_ror = (ror - bh_ror)
    if relative_ror>0:
        fitness+=3
    if relative_ror>20:
        fitness+=1
    
    # Max drawdown penalty
    max_drawdown = performance['Max. Drawdown [%]']
    if max_drawdown>30:
        fitness-=1
    if max_drawdown>50:
        fitness-=1
    if max_drawdown>70:
        fitness-=10

    return fitness

def multi_objective_fitness_function_2(performance: pd.DataFrame):
    # Calculate the fitness socre based on the following equation
    # Fitness = 𝑃𝑛𝐿 + 1.5×𝑃𝑛𝐿𝑟𝑒𝑙𝑎𝑡𝑖𝑣𝑒 − 0.5×𝑚𝑎𝑥(𝑑𝑟𝑎𝑤𝑑𝑜𝑤𝑛) + 0.0005×#𝑡𝑟𝑎𝑑𝑒𝑠 − 𝑎𝑣𝑔(𝑑𝑢𝑟𝑎𝑡𝑖𝑜𝑛)
    
    ror = performance['Return [%]'] # The RoR of strategy
    bh_ror = performance['Buy & Hold Return [%]'] # The RoR of Benchmark
    relative_ror = (ror - bh_ror)
    
    relative_ror_score = relative_ror/100*1.5
    ror_score = ror/100

    # Max drawdown score
    max_drawdown = performance['Max. Drawdown [%]']
    if max_drawdown is None:
        max_drawdown = 0
    max_drawdown_score = max_drawdown/100*0.5

    # Trade frequency score
    num_trade = performance['# Trades']
    trade_freq_score = num_trade*0.0005

    # Average trade duration penalty
    avg_trade_duration = performance['Avg. Trade Duration']
    trade_duration_penalty = 0
    if num_trade>0 and avg_trade_duration.days>14:
        trade_duration_penalty = -0.001*avg_trade_duration.days
    
    return ror_score + relative_ror_score + max_drawdown_score + trade_freq_score + trade_duration_penalty