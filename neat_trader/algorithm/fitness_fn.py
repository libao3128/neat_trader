import pandas as pd
import datetime
import numpy as np

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

def outperform_benchmark(performance: pd.DataFrame):
    # A fitness function based on outperform benchmark.
    ror = performance['Return [%]']
    bh_ror = performance['Buy & Hold Return [%]']
    relative_ror_reward = (ror - bh_ror)/100
    
    # Return reward
    return_reward = ror/100*0.1

    # Encourage the strategy to trade more
    num_trade = performance['# Trades']
    trade_freq_reward = num_trade*0.0001

    # Risk management
    max_drawdown = performance['Max. Drawdown [%]']
    if max_drawdown==0:
        max_drawdown_penalty = 0
    else:
        max_drawdown_penalty = 1/(1- (max_drawdown/100))
    
    return relative_ror_reward + return_reward + trade_freq_reward - max_drawdown_penalty

def gpt_fitness_fn(performance: pd.DataFrame):
    def trade_freq_function(t, t_target):
        sigma = 5
        return np.exp(-(t-t_target)**2/sigma**2)
    if performance['# Trades'] == 0:
        return 0
    sharpe_ratio = performance['Sharpe Ratio']
    annualized_return = performance['Return (Ann.) [%]']/100
    max_drawdown = performance['Max. Drawdown [%]']/100
    if max_drawdown is None:
        max_drawdown = 0
        
    duration = performance['Duration']
    duration_days = duration.days if isinstance(duration, datetime.timedelta) else duration
    t_target = duration_days/30*3
    trade_freq = performance['# Trades']
    trade_freq_score = trade_freq_function(trade_freq, t_target)
    
    score = (sharpe_ratio*annualized_return)/(1+max_drawdown)*trade_freq_score * 100
    print(performance, score)
    return score