from track.track_loader import Track
from track.track_visualizer import plot_track, plot_curvature
from physics.vehicle_model import VehicleModel
# from physics.lap_simulator import simulate_lap
import numpy as np

def main():
    track = Track("../data/silverstone.csv")

    print("Track width stats:")
    print("Min:", track.track_width.min())
    print("Max:", track.track_width.max())
    print("Mean:", track.track_width.mean())

    plot_track(track)
    plot_curvature(track)

    # Create vehicle and simulate lap
    vehicle_params = {}
    car = VehicleModel(vehicle_params)
    # trajectory = simulate_lap(car, track)

    # Plot simulated trajectory
    # plot_track(track, trajectory)

if __name__ == "__main__":
    main()