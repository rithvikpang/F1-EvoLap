import numpy as np

def simulate_lap(vehicle, track, dt=0.1):
    """
    Simulate a lap along the centerline using simple curvature-based speed control.
    """
    vehicle.reset(track.x_center[0], track.y_center[0], 0.0, 0.0)
    trajectory = []

    # Max speed depending on curvature
    k_max = np.max(np.abs(track.curvature))
    for i in range(len(track.x_center)-1):
        k = track.curvature[i]
        # simple speed planning: slower for higher curvature
        target_speed = vehicle.max_speed * np.exp(-5*np.abs(k))  
        accel = (target_speed - vehicle.v) / dt
        # compute steering as angle towards next centerline point
        dx = track.x_center[i+1] - vehicle.x
        dy = track.y_center[i+1] - vehicle.y
        target_yaw = np.arctan2(dy, dx)
        steer = target_yaw - vehicle.yaw
        # wrap steer angle
        steer = (steer + np.pi) % (2*np.pi) - np.pi

        vehicle.step(accel, steer, dt)
        trajectory.append((vehicle.x, vehicle.y))

    return np.array(trajectory)
