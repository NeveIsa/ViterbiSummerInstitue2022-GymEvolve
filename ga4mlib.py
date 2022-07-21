import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import spacegame
import multiprocessing.dummy as mp
from more_itertools import chunked

POOLSIZE=100
pool = mp.Pool(POOLSIZE)

def linearpolicy(obsv, particle):
    return particle @ obsv

def objectivefn(particles, render=False):
    particles = [particles]
    rewards = []
    for pchunks in list(chunked(particles,POOLSIZE)):
        orgs = list(map(lambda x:x.reshape(2,8),pchunks))
        r = pool.map(lambda o:spacegame.play(policy=linearpolicy, organism=o ,render=render), orgs)
        rewards += r

    return -np.array(rewards)

varbound = np.array([[-50,50]]*16)

algorithm_param = {'max_num_iteration': 20,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

def f(X):
    return np.sum(X)

model = ga(function = objectivefn, dimension = 16, variable_type = 'real', variable_boundaries = varbound, algorithm_parameters=algorithm_param)

x = model.run()
print('----')
best_org = model.best_variable
spacegame.play(policy=linearpolicy, organism=np.array(best_org).reshape(2,8) ,render=True)
