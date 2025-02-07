import neat
import os

def load_configuration(config_file_path):
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                       neat.DefaultSpeciesSet, neat.DefaultStagnation,
                       config_file_path)

def initialize_population(config, checkpoint_path):
    if checkpoint_path is None:
        return neat.Population(config)
    return neat.Checkpointer.restore_checkpoint(checkpoint_path)

def add_reporters(population, folder):
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    if not os.path.exists(folder):
        os.makedirs(folder)
    population.add_reporter(neat.Checkpointer(5, filename_prefix=f'{folder}neat-checkpoint-'))
    return stats