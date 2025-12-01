"""
Evolutionary Strategy (self-adaptive) to optimize a per-segment throttle multiplier
that increases/decreases effective engine power when simulating laps on a fixed trajectory.
This is a proxy for optimizing throttle/brake behavior.
"""


import numpy as np
import math
from copy import deepcopy
from evaluation.fitness_function import evaluate_trajectory
from physics.lap_simulator import LapSimulator

def es_optimize_throttle(traj_path, track, vehicle,
                        pop_size=20, n_gens=100, Np=30, verbose=True):
   """
   traj_path: (M,2) numpy array - path to follow (fixed geometry)
   We'll represent throttle profile as Np control points along the path (multipliers).
   """


   # initialize population: params ~ 1.0 +/- noise, sigma ~ 0.1
   attr = "max_power"
   pop = []
   for _ in range(pop_size):
       params = np.ones(Np) + np.random.normal(0, 0.05, size=Np)
       sigma = np.ones(Np) * 0.1
       pop.append((params, sigma))

   def evaluate_params(params):
       # create a temporary vehicle copy and scale its power if attribute exists
       try:
           vcopy = deepcopy(vehicle)
       except Exception:
           vcopy = vehicle

       # map params to full path
       M = traj_path.shape[0]
       full_mult = np.interp(np.linspace(0, M-1, M), np.linspace(0, M-1, Np), params)
       base = getattr(vcopy, attr)
       setattr(vcopy, attr, base * float(np.median(full_mult)))

       # Now run the lap simulator on traj_path
       res = LapSimulator(vcopy, traj_path)
       lap_time, _ = res.simulate()
       return lap_time


   # Evaluate initial pop
   scores = [evaluate_params(ind[0]) for ind in pop]
   hist = []
   for gen in range(n_gens):
       # (mu + lambda) style: create one offspring per parent with self-adaptive sigma
       offspring = []
       for params, sigma in pop:
           tau = 1 / math.sqrt(2*len(params))
           sigma_prime = sigma * np.exp(tau * np.random.normal(0,1,size=sigma.shape))
           params_prime = params + sigma_prime * np.random.normal(0,1,size=params.shape)
           params_prime = np.clip(params_prime, 0.4, 1.6)
           offspring.append((params_prime, sigma_prime))
       combined = pop + offspring
       combined_scores = [evaluate_params(ind[0]) for ind in combined]
       idx = np.argsort(combined_scores)
       pop = [combined[i] for i in idx[:pop_size]]
       best_score = combined_scores[idx[0]]
       hist.append(best_score)
       if verbose and (gen == 1 or gen % 5 == 0 or gen == n_gens):
           print(f"[ES] Gen {gen}/{n_gens} best lap time ≈ {best_score:.3f} s")


   # return best median profile and history
   best_params, best_sigma = pop[0]
   return best_params, hist