import random
import numpy as np
from copy import deepcopy
from deap import base, creator, tools

from physics.vehicle_model import VehicleModel
from physics.lap_simulator import LapSimulator
from track.track_loader import Track
from trajectory.trajectory_builder import TrajectoryBuilder
from evaluation.fitness_function import evaluate_trajectory

# ------------------------------
# Vehicle parameter ranges
# ------------------------------
# VEHICLE_PARAM_SPEC = [
#     ("CD", 0.85, 0.40, 1.6),
#     ("CL", 3.5, 0.50, 7.0),
#     ("frontal_area", 1.5, 1.0, 2.2),
#     ("downforce_mult", 1.0, 0.2, 1.6),
#     ("drag_mult", 1.0, 0.6, 1.4),
#     ("mu_base", 2.4, 1.2, 3.2),
#     ("front_ride_height", 0.03, 0.0, 0.08),
#     ("rear_ride_height", 0.05, 0.0, 0.12),
#     ("antiroll_front", 60000, 20000, 120000),
#     ("antiroll_rear", 55000, 20000, 120000),
# ]

VEHICLE_PARAM_SPEC = [
    # Aerodynamics (realistic 2022–2025 F1 values)
    ("CD",              0.85,   0.70,   1.10),   # total drag coefficient
    ("CL",              3.50,   2.50,   5.50),   # downforce coefficient
    ("frontal_area",    1.50,   1.35,   1.75),   # m^2
    ("aero_balance",    0.45,   0.40,   0.58),   # fraction DF front axle

    # scalars to allow GA flexibility
    ("drag_mult",       1.00,   0.80,   1.20),
    ("downforce_mult",  1.00,   0.80,   1.30),

    # Engine / Hybrid (MGU-K + MGU-H)
    ("engine_inertia",      0.25,   0.15,   0.40),       # kg*m^2
    ("shift_rpm",           12100, 11000, 12800),        # ICE shift strategy

    ("ers_max_deploy",      120_000,  80_000, 160_000),  # W (real limit 120kW)
    ("ers_max_regen",       240_000, 180_000, 300_000),  # regen ceiling
    ("ers_capacity",        4e6,     3e6,     5e6),      # total battery MJ
    ("ers_efficiency",      0.85,    0.75,    0.95),

    ("mguh_max_power",      80_000,  40_000, 120_000),   # MGU-H turbine gen
    ("mguh_to_battery_eff", 0.80,    0.60,    0.95),
    ("mguh_to_k_eff",       0.75,    0.60,    0.90),

    # Gearbox & Drivetrain
    # These should not vary too wildly—F1 gear ratios are narrow.
    ("final_drive",         3.60,   2.80,   4.20),
    ("drivetrain_efficiency", 0.92, 0.85,   0.95),

    # Brakes (Brembo F1 data)
    ("front_brake_force",   12_000,  10_000, 16_000),   # N
    ("rear_brake_force",    10_000,  8_000,  14_000),   # N
    ("brake_bias",          0.56,    0.50,   0.60),     # fraction front
    ("brake_fade_coeff",    0.01,    0.005,  0.03),     # fade per temp unit
    ("brake_cooling_efficiency", 0.80, 0.60, 0.95),

    # Tires (loaded μ varies between 1.8–3.5)
    ("mu_load_sensitivity",   0.0008, 0.0005, 0.0012),
    ("rolling_resistance",    0.015,  0.010,  0.030),
    ("optimal_slip_angle",    6.0,    4.0,    10.0),

    # Suspension & Ride Heights (2022 ground-effect F1)
    ("front_ride_height",    0.030,  0.020,  0.050),    # m
    ("rear_ride_height",     0.050,  0.030,  0.070),

    ("front_spring_rate",    95_000, 70_000, 130_000),  # N/m
    ("rear_spring_rate",     105_000, 80_000, 150_000),

    ("arb_front",            60_000, 40_000, 100_000),  # Nm/rad
    ("arb_rear",             55_000, 35_000,  90_000),

    ("heave_stiffness",      180_000, 150_000, 240_000),
    ("roll_stiffness",       145_000, 110_000, 190_000),
]

PARAM_NAMES = [p[0] for p in VEHICLE_PARAM_SPEC]
V_MIN = np.array([p[2] for p in VEHICLE_PARAM_SPEC])
V_MAX = np.array([p[3] for p in VEHICLE_PARAM_SPEC])
N_VEHICLE_PARAMS = len(VEHICLE_PARAM_SPEC)


# ------------------------------
# Convert vectors to objects
# ------------------------------

def build_vehicle(vec):
    params = {}
    for i, name in enumerate(PARAM_NAMES):
        params[name] = float(vec[i])
    return VehicleModel(params)

# ------------------------------
# Fitness function
# ------------------------------

def evaluate_individual(ind, track, n_control, penalty_weights):
    """
    ind: full genome
    structure = [vehicle_params | trajectory_control_values]
    """

    vehicle_vec = np.array(ind[:N_VEHICLE_PARAMS])
    ctrl_vec = np.array(ind[N_VEHICLE_PARAMS:])

    # Build objects
    vehicle = build_vehicle(vehicle_vec)
    tb = TrajectoryBuilder()
    traj = tb.build_trajectory(track, track.get_centerline(), ctrl_vec)

    res = evaluate_trajectory(traj, track, vehicle, penalty_weights=penalty_weights)
    fit = res.get("fitness", None)
    if fit is None or not np.isfinite(fit):
        fit = 1e6
    
    return (float(fit),)

# ------------------------------
# GA main function
# ------------------------------

def joint_optimize(track,
                   pop_size=40,
                   n_gens=40,
                   n_control=20,
                   cxpb=0.7,
                   mutpb=0.3,
                   mut_sigma_vehicle=0.05,
                   mut_sigma_line=1.5, # was 0.5
                   penalty_weights={"offtrack": 30.0, "smoothness": 5.0},
                   seed=None, 
                   verbose=True):

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    genome_len = N_VEHICLE_PARAMS + n_control

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # initialization
    def init_individual():
        # vehicle part
        v = [random.uniform(V_MIN[i], V_MAX[i]) for i in range(N_VEHICLE_PARAMS)]
        # trajectory controls (small initial displacement)
        c = [random.uniform(-0.2, 0.2) for _ in range(n_control)]
        return creator.Individual(v + c)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate",
                     lambda ind: evaluate_individual(ind, track, n_control, penalty_weights))

    toolbox.register("select", tools.selTournament, tournsize=3)

    # Crossover: blend only vehicle part, simulated binary on control part
    def crossover(ind1, ind2):
        # vehicle params
        for i in range(N_VEHICLE_PARAMS):
            if random.random() < 0.5:
                alpha = 0.5
                a = ind1[i]
                b = ind2[i]
                ind1[i] = alpha * a + (1 - alpha) * b
                ind2[i] = alpha * b + (1 - alpha) * a

        # trajectory params: swap segments
        cut = random.randint(N_VEHICLE_PARAMS, genome_len - 1)
        ind1[cut:], ind2[cut:] = ind2[cut:], ind1[cut:]
        return ind1, ind2

    toolbox.register("mate", crossover)

    # Mutation: different sigma for vehicle vs line
    def mutate(ind):
        # vehicle mutation
        for i in range(N_VEHICLE_PARAMS):
            if random.random() < 0.4:
                ind[i] += random.gauss(0, mut_sigma_vehicle)
                ind[i] = float(np.clip(ind[i], V_MIN[i], V_MAX[i]))

        # line mutation
        for j in range(N_VEHICLE_PARAMS, genome_len):
            if random.random() < 0.3:
                ind[j] += random.gauss(0, mut_sigma_line)

        return ind,

    toolbox.register("mutate", mutate)

    # Init population
    population = toolbox.population(n=pop_size)

    for gen in range(0, n_gens):
        # Evaluate
        invalid = [ind for ind in population if not ind.fitness.valid]
        fits = toolbox.map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit

        best = min(population, key=lambda x: x.fitness.values[0])
        if verbose and (gen % 5 == 0 or gen == n_gens):
            print(f"[GA] Gen {gen}/{n_gens} best fitness = {best.fitness.values[0]:.3f}")

        # select + clone
        offspring = toolbox.select(population, len(population))
        offspring = list(map(deepcopy, offspring))

        # crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        # mutation
        for mut in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mut)
                del mut.fitness.values

        population[:] = offspring

    # Final best
    # best = min(population, key=lambda x: x.fitness.values[0])
    best_vehicle = build_vehicle(np.array(ind[:N_VEHICLE_PARAMS]))
    print("\n===== Best Vehicle Parameters =====")
    for name, value in zip(PARAM_NAMES, best[:N_VEHICLE_PARAMS]):
        print(f"{name:20s} = {value:.6f}")

    print("\n===== Best Trajectory Control Offsets =====")
    print(best[N_VEHICLE_PARAMS:])
    return best, best_vehicle, best[N_VEHICLE_PARAMS:]
