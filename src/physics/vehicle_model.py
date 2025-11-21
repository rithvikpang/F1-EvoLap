import numpy as np

class Vehicle:
    def __init__(self, max_speed=50.0, max_accel=5.0, max_steer=np.deg2rad(30)):
        self.max_speed = max_speed      # m/s
        self.max_accel = max_accel      # m/s^2
        self.max_steer = max_steer      # radians
        self.reset()

    def reset(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.trajectory = [(x, y)]

    def step(self, accel, steer, dt=0.1):
        # Clamp control inputs
        accel = np.clip(accel, -self.max_accel, self.max_accel)
        steer = np.clip(steer, -self.max_steer, self.max_steer)

        # Update velocity and pose
        self.v += accel * dt
        self.v = np.clip(self.v, 0.0, self.max_speed)

        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v * np.tan(steer) / 2.5 * dt  # wheelbase=2.5m

        self.trajectory.append((self.x, self.y))

    def get_trajectory(self):
        return np.array(self.trajectory)
