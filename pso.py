from operator import neg
import numpy as np
import spacegame
import pyswarms as ps
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
import multiprocessing.dummy as mp
from more_itertools import chunked
from datetime import datetime
import fire
import dill

POOLSIZE = 100
pool = mp.Pool(POOLSIZE)


def linearpolicy(obsv, particle):
    return particle @ obsv


def objectivefn(particles, render=False):
    # organism = np.array(particle).reshape(2,8)
    rewards = []
    for pchunks in list(chunked(particles, POOLSIZE)):
        orgs = list(map(lambda x: x.reshape(2, 8), pchunks))
        r = pool.map(
            lambda o: spacegame.play(policy=linearpolicy, organism=o, render=render),
            orgs,
        )
        rewards += r

    return -np.array(rewards)


def main(POPSIZE=100, GENS=10):

    outfile = f"store/pso/{datetime.now().isoformat().split('.')[0]}"

    options = {"c1": 0.8, "c2": 0.5, "w": 0.7}
    max_val = np.ones(16) * 50
    min_val = -max_val
    bounds = (min_val, max_val)
    optimizer = ps.single.GlobalBestPSO(
        n_particles=POPSIZE, dimensions=16, options=options, bounds=bounds
    )
    # Perform optimization
    for gen in range(GENS):
        negreward, bestparticle = optimizer.optimize(objectivefn, iters=1)
        reward = -negreward

        pos = optimizer.swarm.position
        costs = optimizer.swarm.current_cost
        print("top5:",sorted(costs, reverse=False)[:5])

        dill.dump({"position":pos, "cost":costs}, open(f"{outfile}-{gen}.dill", "wb"))

        # plot_cost_history(cost_history=optimizer.cost_history)
        # plt.show()

        # runreward = spacegame.play(policy=linearpolicy, organism=bestparticle.reshape(2, 8), render=True)
        # print('run_reward:',runreward)


    
if __name__ == "__main__":
    fire.Fire(main)

