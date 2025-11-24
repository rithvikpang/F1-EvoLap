"""Main script to run lap simulation on a racing track using a vehicle model."""

import logging
import numpy as np
from track.track_loader import Track
from track.track_visualizer import plot_track
from physics.vehicle_model import VehicleModel
from physics.lap_simulator import LapSimulator
from visualization.visualize_results import plot_lap_simulation_results
from evolution.ga_optimizer import ga_optimize
from evolution.es_optimizer import es_optimize_throttle
from evolution.ga_optimizer2 import joint_optimize

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading Silverstone track...")
    track = Track("../data/silverstone.csv")

    logger.info("\nTrack width stats:")
    logger.info(f"  Min: {track.track_width.min():.2f} m")
    logger.info(f"  Max: {track.track_width.max():.2f} m")
    logger.info(f"  Mean: {track.track_width.mean():.2f} m")
    
    racing_line = track.get_centerline()
    logger.info(f"\nRacing line points: {len(racing_line)}")
    
    # Quick track preview
    logger.info("\nShowing track layout...")
    plot_track(track, trajectory=racing_line)
    
    # Create vehicle with default parameters
    logger.info("\nCreating vehicle model...")
    vehicle_params = {}
    vehicle = VehicleModel(vehicle_params)
    
    logger.info("\n" + "="*60)
    logger.info("RUNNING LAP SIMULATION")
    logger.info("="*60)
    
    sim = LapSimulator(vehicle, racing_line)
    lap_time, times = sim.simulate()
    
    logger.info("\n" + "="*60)
    logger.info(f"LAP TIME: {lap_time:.3f} seconds ({lap_time/60:.3f} minutes)")
    logger.info("="*60)
    
    logger.info(f"\nSpeed statistics:")
    logger.info(f"  Max speed: {sim.v_final.max() * 3.6:.1f} km/h")
    logger.info(f"  Min speed: {sim.v_final.min() * 3.6:.1f} km/h")
    logger.info(f"  Average speed: {sim.v_final.mean() * 3.6:.1f} km/h")
    
    # Physics analysis
    logger.info("\nGenerating comprehensive visualization...")
    plot_lap_simulation_results(sim, track, racing_line)



    # 1) GA: find geometry (offsets)
    # ga_res = ga_optimize(track, vehicle, sim.simulate,
    #                  pop_size=12, n_gens=20, n_control=16,
    #                  mut_sigma=0.8, elitism=2, tournament_k=3,
    #                  penalty_weights={"offtrack":200.0, "smoothness":2.0},
    #                  verbose=True)
    # ga_res = ga_optimize(track, vehicle, sim.simulate,
    #             n_control=16, pop_size=30, n_gens=20,
    #             cxpb=0.8, mutpb=0.2, mut_scale=1.0,
    #             tournament_k=3, elitism=2, penalty_weights={"offtrack":50.0, "smoothness":5.0}, verbose=True, seed=None)
               
    # ga_res = ga_optimize(track, vehicle, sim.simulate,
    #             n_control=20, pop_size=40, n_gens=20,
    #             cxpb=0.7, mutpb=0.35, mut_scale=0.5,
    #             tournament_k=4, elitism=3, penalty_weights={"offtrack":200.0, "smoothness":2.0}, verbose=True, seed=None)

    # best_path = ga_res["best_traj"].get_path()
    # print("GA best lap_time:", ga_res["best_eval"]["lap_time"], "fitness:", ga_res["best_eval"]["fitness"])


    # 2) ES: refine throttle on the GA-best geometry
    # best_params, es_hist = es_optimize_throttle(best_path, track, vehicle, sim.simulate,
    #                                        pop_size=20, n_gens=80, Np=30, verbose=True)
    # print("ES best median multiplier (approx):", float(np.median(best_params)))


    best = joint_optimize(
        track,
        pop_size=40,
        n_gens=30,
        n_control=20,
        cxpb=0.7,
        mutpb=0.3,
        seed=0,
        verbose=True
    )

    print("Best fitness:", best.fitness.values[0])
    # print("Best genome:", best)


if __name__ == "__main__":
    main()