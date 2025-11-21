"""Module for visualizing the racing track and optimal trajectory."""

import matplotlib.pyplot as plt

def plot_track(track, trajectory=None):
    plt.figure(figsize=(12, 7))
    plt.plot(track.x_left, track.y_left, 'g-', label="Left Boundary")
    plt.plot(track.x_right, track.y_right, 'b-', label="Right Boundary")
    #plt.plot(track.x_center, track.y_center, 'k--', label="Centerline")


    if trajectory is not None:
        plt.plot(trajectory[:,0], trajectory[:,1], 'r', linewidth=2.5, label="Optimal Trajectory")

    plt.axis('equal')
    plt.legend()
    plt.title("Silverstone Circuit - Track Geometry")
    plt.show()
    