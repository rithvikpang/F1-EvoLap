import numpy as np
from scipy.interpolate import splprep, splev

class TrajectoryBuilder:

    def __init__ (self, sample_count = 2000):
        self.sample_count = sample_count

    def resample_uniform(self, x, y):        
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.insert(np.cumsum(ds), 0, 0)

        s_uniform = np.linspace(0, s[-1], self.sample_count)

        x_spline, _ = splprep([x], s=0, k=1)
        y_spline, _ = splprep([y], s=0, k=1)

        x_u = np.interp(s_uniform, s, x)
        y_u = np.interp(s_uniform, s, y)

        return np.vstack([x_u, y_u]).T
    
    def build_trajectory(self, centerline, offsets):
        x_center = centerline[:, 0]
        y_center = centerline[:, 1]

        dx = np.gradient(x_center)
        dy = np.gradient(y_center)
        mag = np.sqrt(dx**2 + dy**2) + 1e-9

        tx = dx / mag
        ty = dy / mag
        nx = -ty
        ny = tx 

        x_ctrl = x_center + offsets * nx
        y_ctrl = y_center + offsets * ny

        tck, _ = splprep([x_ctrl, y_ctrl], s=0, k=3)
        u_fine = np.linspace(0, 1, self.sample_count)

        x_spline, y_spline = splev(u_fine, tck)

        trajectory = self.resample_uniform(np.array(x_spline), np.array(y_spline))

        return trajectory