"""Main script to run lap simulation on a racing track using a vehicle model."""

import logging
from track.track_loader import Track
from track.track_visualizer import plot_track
from physics.vehicle_model import VehicleModel
from physics.lap_simulator import LapSimulator
from trajectory.trajectory_builder import TrajectoryBuilder
import numpy as np
from visualization.visualize_results import plot_lap_simulation_results

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading Silverstone track...")
    track = Track("data/silverstone.csv")

    logger.info("\nTrack width stats:")
    logger.info(f"  Min: {track.track_width.min():.2f} m")
    logger.info(f"  Max: {track.track_width.max():.2f} m")
    logger.info(f"  Mean: {track.track_width.mean():.2f} m")
    
    centerline = track.get_centerline()
    logger.info(f"\nRacing line points: {len(centerline)}")
    
    # Quick track preview
    logger.info("\nShowing track layout...")
    offsets = np.zeros(len(centerline))
    builder = TrajectoryBuilder(sample_count=2000)
    trajectory = builder.build_trajectory(centerline, offsets)

    plot_track(track, trajectory=trajectory)
    
    # Create vehicle with default parameters
    logger.info("\nCreating vehicle model...")
    vehicle_params = {}
    vehicle = VehicleModel(vehicle_params)
    
    logger.info("\n" + "="*60)
    logger.info("RUNNING LAP SIMULATION")
    logger.info("="*60)
    
    sim = LapSimulator(vehicle, trajectory)
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
    plot_lap_simulation_results(sim, track, trajectory)

if __name__ == "__main__":
    main()