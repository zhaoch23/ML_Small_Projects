import random
from deap import base, creator, tools, algorithms
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial
from tqdm import tqdm
from math import sqrt

import argparse

eps = 1e-3

parser = argparse.ArgumentParser(description='IDM model with genetic algorithm')

parser.add_argument('--pop', type=int, default=128,
                    help='population size')

parser.add_argument('--gen', type=int, default=2048,
                    help='generation size')

parser.add_argument('--car_id', type=int, default=2,
                    help='car id 1 < car_id < 4')

parser.add_argument('--dataset', type=str, default='./data/drivetest1.FCdata',
                    help='dataset path, seperated by comma')

parser.add_argument('--cxpb', type=float, default=0.5,
                    help='crossover probability')

parser.add_argument('--mutpb', type=float, default=0.2,
                    help='mutation probability')

parser.add_argument('--indpb', type=float, default=0.05,
                    help='individual mutation probability')

parser.add_argument('--tournsize', type=int, default=3,
                    help='tournament size in each iterate')

parser.add_argument('--init_low', type=float, default=eps,
                    help='initial value lower bound for parameters')

parser.add_argument('--init_high', type=int, default=1,
                    help='initial value upper bound for parameters')

parser.add_argument('--mate_alpha', type=float, default=0.5,
                    help='crossover alpha for parameters')

parser.add_argument('--mutate_mu', type=float, default=0.0,
                    help='mutation guassian mu for parameters')

parser.add_argument('--mutate_sigma', type=float, default=0.2,
                    help='mutation guassian sigma for parameters')

parser.add_argument('--mutate_indpb', type=float, default=0.05,
                    help='mutation probability for parameters')

parser.add_argument('--dt', type=float, default=0.1,
                    help='time step')

parser.add_argument('--num_previous_elites', type=int, default=8,
                    help='number of previous elites')

parser.add_argument('--plot', action='store_true', default=True,
                    help='plot the result')

def acc_idm(v, s, dv, a, s0, v0, T, b):
    """
    v: velocity
    s: distance between front car
    dv: velocity difference of front car (v - v_front)
    a: acceleration
    s0: minimum distance
    v0: desired velocity
    T: time headway
    b: safe deceleration
    """
    s_star_tmp = v * T + v * dv / (2 * max(eps, sqrt(a*b)))
    s_star_tmp[s_star_tmp < 0] = 0
    s_star = s0 + s_star_tmp
    s[s < eps] = eps
    return a * (1 - (v / max(eps, v0)) ** 4 - (s_star / s) ** 2)

def acc_idm_scalar(v, s, dv, a, s0, v0, T, b):
    s_star_tmp = v * T + v * dv / (2 * max(eps, sqrt(a*b)))
    s_star_tmp = max(eps, s_star_tmp)
    s_star = s0 + s_star_tmp
    s = max(eps, s)
    return a * (1 - (v / max(eps, v0)) ** 4 - (s_star / s) ** 2)

# argument: argmin(rmse(acc_pred, acc_label)) <- v, ds
def rmse(v, s, dv, acc, individual):
    a, s0, v0, T, b = individual
    acc_pred = acc_idm(v, s, dv, a, s0, v0, T, b)
    return ((acc_pred - acc) ** 2).mean(),

def rmse_scalar(y, y_pred):
    return sqrt(mean_squared_error(y, y_pred))

# To draw the graph, given the ics, iterate to get the results.
def euler_method(Ics, partial_acc_f, steps, dt):
    va, vb0, ds0 = Ics # initial conditions: Front car velocity, Self velocity, distance between them
    acc = []
    vb = [vb0]
    ds = [ds0]
    for i in range(steps - 1):
        acc.append(partial_acc_f(vb[i], ds[i], vb[i] - va[i]))
        vb.append(vb[i] + acc[i] * dt)
        ds.append(ds[i] + (va[i] - vb[i]) * dt)

    return vb, ds

def _mutate(individual, mu, sigma, indpb):
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            individual[i] += random.gauss(mu, sigma)
            individual[i] = max(eps, individual[i])
    return individual,

def _crosseover(ind1, ind2, alpha):
    for i in range(len(ind1)):
        ind1[i] = max(eps, (1-alpha)*ind1[i] + alpha*ind2[i])
        ind2[i] = max(eps, alpha*ind1[i] + (1-alpha)*ind2[i])
    return ind1, ind2

def main():
    args = parser.parse_args()
    print(args)

    # read data
    df = pd.read_csv(args.dataset, sep='\t', header=None)
    df.columns = ['v1', 'v2', 'v3', 'v4', 'd12', 'd23', 'd34']

    v = df['v{}'.format(args.car_id)].values
    v_front = df['v{}'.format(args.car_id - 1)].values
    ds = df['d{}{}'.format(args.car_id - 1, args.car_id)].values
    dv = (v - v_front)[:-1]
    acc = np.diff(v) / args.dt
    v = v[:-1]
    v_front = v_front[:-1]
    ds = ds[:-1]

   # Setup DEAP
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, args.init_low, args.init_high)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", partial(rmse, v, ds, dv, acc))
    toolbox.register("mate", _crosseover, alpha=args.mate_alpha)
    toolbox.register("mutate", _mutate, mu=args.mutate_mu, sigma=args.mutate_sigma, indpb=args.mutate_indpb)
    toolbox.register("select", tools.selTournament, tournsize=args.tournsize)

    population = toolbox.population(n=args.pop)

    # Iterate
    pbar = tqdm(range(args.gen))
    for gen in pbar:
        offspring = algorithms.varAnd(population, toolbox, cxpb=args.cxpb, mutpb=args.mutpb)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        
        # Elitism
        #population = toolbox.select(offspring, k=len(population))
        selected = toolbox.select(offspring, k=len(population) - args.num_previous_elites)

        elite = tools.selBest(population, k=args.num_previous_elites)
        next_generation = elite + selected

        pbar.set_description(f'Best fitness: {tools.selBest(next_generation, k=1)[0].fitness.values[0]}')  

        population[:] = next_generation

    best_ind = tools.selBest(population, k=1)[0] # best individual

    v_self_pred, ds_pred = euler_method([v_front, v[0], ds[0]], partial(acc_idm_scalar, a=best_ind[0], s0=best_ind[1], v0=best_ind[2], T=best_ind[3], b=best_ind[4]), len(v_front), args.dt)
    print('Best individual: ', best_ind)
    print('RMSE dv/dt: ', rmse(acc, v, ds, dv, best_ind)[0])
    print(f'RMSE v: ', rmse_scalar(v, v_self_pred))
    print(f'RMSE ds: ', rmse_scalar(ds, ds_pred))

    if args.plot:
        plot_estimation(v, v_self_pred, ds, ds_pred, args.dt)

def plot_estimation(v, v_pred, ds, ds_pred, dt):
    # Plot two figure
    plt.figure(figsize=(20, 10))
    ax = plt.subplot(2, 1, 1)
    ts = np.arange(0, len(v) * dt, dt)
    plt.plot(ts, v, label='v')
    plt.plot(ts, v_pred, label='v_pred')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('velocity (m/s)')
    plt.legend()
   
    ax = plt.subplot(2, 1, 2)
    plt.plot(ts, ds, label='ds')
    plt.plot(ts, ds_pred, label='ds_pred')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('distance (m)')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()