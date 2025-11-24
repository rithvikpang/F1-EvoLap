import random
import numpy as np
from copy import deepcopy
from deap import base, creator, tools, algorithms

from evaluation.fitness_function import evaluate_trajectory
from evolution.trajectory import Trajectory

# def setup_deap(n_control, min_offset=-12.0, max_offset=12.0, seed=None):
def setup_deap(n_control, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # single objective minimization (lap time + penalties)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    # Attribute generator
    # toolbox.register("attr_offset", random.uniform, min_offset, max_offset)
    # Structure initializers
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_offset, n=n_control)
    toolbox.register("individual", tools.initRepeat, creator.Individual, random.random, n=n_control)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox

def ga_optimize(track, vehicle, lap_simulator,
                n_control=30, pop_size=40, n_gens=80,
                cxpb=0.5, mutpb=0.3, mut_scale=1.0,
                tournament_k=3, elitism=2, penalty_weights=None, verbose=True, seed=None):
    
    centerline = track.get_centerline() if hasattr(track, 'get_centerline') else np.column_stack((track.x_center, track.y_center))
    # bounds for offsets: use half track width (conservative)
    # if hasattr(track, "get_track_width"):
    #     half_width = float(np.mean(track.get_track_width()) / 2.0)
    #     min_off = -half_width
    #     max_off = half_width
    # else:
    #     min_off = -6.0
    #     max_off = 6.0

    # DEAP setup
    # toolbox = setup_deap(n_control, min_offset=min_off, max_offset=max_off, seed=seed)
    toolbox = setup_deap(n_control, seed=seed)

    # # Crossover: blend (alpha) implemented manually
    # def cx_blend(ind1, ind2, alpha=0.3):
    #     a = np.array(ind1)
    #     b = np.array(ind2)
    #     lam = np.random.uniform(-alpha, 1+alpha, size=a.shape)
    #     child1 = lam*a + (1-lam)*b
    #     child2 = lam*b + (1-lam)*a
    #     ind1[:] = list(np.clip(child1, min_off, max_off))
    #     ind2[:] = list(np.clip(child2, min_off, max_off))
    #     return ind1, ind2

    # # Mutation: gaussian on each gene
    # def mut_gauss(individual, mu=0.0, sigma=1.0, indpb=0.2):
    #     arr = np.array(individual)
    #     mask = np.random.rand(len(arr)) < indpb
    #     arr[mask] += np.random.normal(mu, sigma, size=mask.sum())
    #     arr = np.clip(arr, min_off, max_off)
    #     individual[:] = list(arr)
    #     return (individual,)

    # toolbox.register("mate", cx_blend, alpha=0.3)
    # toolbox.register("mutate", mut_gauss, sigma=mut_scale, indpb=0.2)
    toolbox.register("mate", tools.cxBlend, alpha=0.7)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=tournament_k)

    def eval_individual(ind):
        offsets = np.array(ind, dtype=float)
        traj = Trajectory(centerline, n_control, offsets)
        path = traj.get_path()
        res = evaluate_trajectory(path, track, vehicle, lap_simulator, penalty_weights=penalty_weights)
        fit = res.get("fitness", None)
        if fit is None or not np.isfinite(fit):
            fit = 1e6
        return (float(fit),)

    toolbox.register("evaluate", eval_individual)

    # create population
    pop = toolbox.population(n=pop_size)
    # ensure baseline individual (zero offsets = centerline) is present as first individual
    for i in range(n_control):
        pop[0][i] = 0.0

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    history = []
    hof = tools.HallOfFame(1, similar=np.array_equal)

    for gen in range(1, n_gens+1):
        offspring = toolbox.select(pop, len(pop) - elitism)
        offspring = list(map(toolbox.clone, offspring))

        for i in range(0, len(offspring), 2):
            if i+1 >= len(offspring):
                break
            if random.random() < cxpb:
                toolbox.mate(offspring[i], offspring[i+1])
                del offspring[i].fitness.values
                del offspring[i+1].fitness.values

        for i in range(len(offspring)):
            if random.random() < mutpb:
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        elites = tools.selBest(pop, elitism)
        pop = elites + offspring

        invalid = [ind for ind in pop if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        hof.update(pop)
        best = hof[0]
        history.append(best.fitness.values[0])

        if verbose and (gen % 10 == 0 or gen == n_gens):
            print(f"[GA] Gen {gen}/{n_gens} best fitness = {best.fitness.values[0]:.6f}")

    best_ind = hof[0]
    best_offsets = np.array(best_ind, dtype=float)
    best_traj = Trajectory(centerline, n_control, best_offsets)
    best_eval = evaluate_trajectory(best_traj.get_path(), track, vehicle, lap_simulator, penalty_weights=penalty_weights)

    return {
        "best_offsets": best_offsets,
        "best_traj": best_traj,
        "best_eval": best_eval,
        "history": history
    }