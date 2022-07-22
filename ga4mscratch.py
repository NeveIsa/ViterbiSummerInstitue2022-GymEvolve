# import pathos.multiprocessing as mp
import multiprocessing.dummy as mp
import signal
import sys

import fire
import numpy as np
from more_itertools import chunked
from tqdm import tqdm
import dill
from datetime import datetime
import spacegame


def stopgame(signo, stackframe):
    sys.exit(-2)


signal.signal(signal.SIGINT, stopgame)


class Algo:
    def __init__(
        self, game, orgdims=16, popsize=100, userpolicy=None, nworkers=25, render=False
    ):
        self.game = game
        self.fitness = [0] * popsize
        self.survivors = []
        self.popsize = popsize
        self.render = render
        self.userpolicy = userpolicy
        self.nworkers = nworkers
        self.orgdims = orgdims
        self.pop = [self.getorganism() for i in range(popsize)]
        self.lastpop = [None for i in range(popsize)]
        
    def policy(self, observation, organism):
        if self.userpolicy:
            action = self.userpolicy(observation, organism)
        else:
            action = [np.random.rand() - 0.5, np.random.rand() - 0.5]

        return action

    def getorganism(self):
        return (np.random.rand(self.orgdims) - 0.5) * 100

    def mate(self, replace_fraction=0.5):

        nchildren = int(replace_fraction * self.popsize)

        # kill the unfit
        self.pop = self.pop[
            :-nchildren
        ]  # self.pop was already sorted in unnaturalselection

        for i in range(nchildren):
            indx1 = np.random.randint(len(self.survivors))
            indx2 = np.random.randint(len(self.survivors))
            org1 = self.survivors[indx1]
            org2 = self.survivors[indx2]
            child = 0.5 * (org1 + org2)

            self.pop.append(child)

    def evaluateorganism(self, organism, ntimes=1):
        rewards = []

        for i in range(ntimes):
            reward = self.game.play(self.policy, organism, render=self.render)
            rewards.append(reward)

        return rewards

    def unnaturalselection(self, survival_fraction=0.1):
        n_survivors = int(survival_fraction * self.popsize)
        n_survivors = max(n_survivors, 2)

        sorted_popfit = sorted(
            zip(self.pop, self.fitness), key=lambda x: x[1], reverse=True
        )
        sorted_pop = list(map(lambda x: x[0], sorted_popfit))
        self.pop = sorted_pop
        self.survivors = np.copy(
            sorted_pop[:n_survivors]
        )  # if we dond't copy, mutate will modify the survivors too

    def mutate(self, probability=0.1, deviation_fraction=0.2):
        for i in range(self.popsize):
            if np.random.rand() < probability:
                deviation_magnitude = np.mean(self.pop[i]) * deviation_fraction
                deviation = deviation_magnitude * np.random.rand(*self.pop[i].shape)
                self.pop[i] += deviation

    def evolve(self):
        def evalorgs(org):
            rewards = self.evaluateorganism(org)
            totalreward = sum(rewards)
            return totalreward

        pool = mp.Pool(self.nworkers)

        for orgids in tqdm(
            list(chunked(range(self.popsize), self.nworkers)),
            colour="green",
            leave=False,
        ):

            orgs = list(map(lambda i: self.pop[i], orgids))

            if not self.render:
                __rewards = pool.map(evalorgs, orgs)
            else:
                __rewards = list(map(evalorgs, orgs))

            for i, orgid in enumerate(orgids):
                self.fitness[orgid] = __rewards[i]

         # save population
        self.lastpop = [np.copy(org) for org in self.pop]  
        



        # select a few
        self.unnaturalselection()

        # mate them
        self.mate()

        # mutate
        self.mutate()


def main(POPSIZE=50, GENS=3, NWORKERS=25, RENDER=False):

    outfile = f"store/ga/{datetime.now().isoformat().split('.')[0]}"
    

    ########## LINEAR
    def linearpolicy(observation, organism):
        # return organism[:-2].reshape(2,8) @ np.array(observation) + organism[-2:]
        return organism.reshape(2, 8) @ np.array(observation)

    genealgo = Algo(
        # orgdims=18, popsize=POPSIZE, userpolicy=linearpolicy, game=spacegame, render=RENDER
        orgdims=16,
        popsize=POPSIZE,
        userpolicy=linearpolicy,
        game=spacegame,
        render=RENDER,
    )

    for generation in tqdm(range(GENS), colour="red", leave=True):
        genealgo.evolve()

        print("\n\ntop5:", sorted(genealgo.fitness, reverse=True)[:5], "\n\n")

        # finalreward = spacegame.play(
            # genealgo.policy, genealgo.survivors[0], render=True
        # )
        # print("\nLinear Reward :", finalreward)


        # store to file
        lastpop = np.array(genealgo.lastpop)   
        lastfitness = np.array(genealgo.fitness)
        dill.dump({"pop":lastpop, "fitness": lastfitness}, open(f"{outfile}-gen:{generation}.dill",'wb'))




    # ########### NONLINEEAR
    # def nonlinearpolicy(observation, organism):
        # # action = organism[:-2].reshape(2,8) @ np.array(observation) + organism[-2:]
        # action = organism.reshape(2, 8) @ np.array(observation)
        # return np.arctan(action) * 2 / np.pi
# 
    # genealgo = Algo(
        # # orgdims=18, popsize=POPSIZE, userpolicy=nonlinearpolicy, game=spacegame, render=RENDER
        # orgdims=16,
        # popsize=POPSIZE,
        # userpolicy=nonlinearpolicy,
        # game=spacegame,
        # render=RENDER,
    # )
# 
    # for generation in tqdm(range(GENS), colour="red", leave=True):
        # genealgo.evolve()
        # finalreward = spacegame.play(
            # genealgo.policy, genealgo.survivors[0], render=True
        # )
        # print("\n NonLinear Reward :", finalreward)
        # print("\n\ntop5:", sorted(genealgo.fitness, reverse=True)[:5], "\n\n")



    
    
if __name__ == "__main__":
    fire.Fire(main)
