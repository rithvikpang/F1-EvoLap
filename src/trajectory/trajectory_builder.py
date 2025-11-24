# import numpy as np
# from scipy.interpolate import splprep, splev

# class TrajectoryBuilder:

#     def __init__ (self, sample_count = 2000):
#         self.sample_count = sample_count

#     def resample_uniform(self, x, y):        
#         dx = np.diff(x)
#         dy = np.diff(y)
#         ds = np.sqrt(dx**2 + dy**2)
#         s = np.insert(np.cumsum(ds), 0, 0)

#         s_uniform = np.linspace(0, s[-1], self.sample_count)

#         # x_spline, _ = splprep([x], s=0, k=1)
#         # y_spline, _ = splprep([y], s=0, k=1)

#         x_u = np.interp(s_uniform, s, x)
#         y_u = np.interp(s_uniform, s, y)

#         return np.vstack([x_u, y_u]).T
    
#     def build_trajectory(self, track, centerline, offsets): 
#         n_full = 1178

#         # Upsample offsets to match full resolution
#         offsets_full = np.interp(
#             np.linspace(0, len(offsets) - 1, n_full),
#             np.arange(len(offsets)),
#             offsets
#         )

#         # Clip to track boundarieS
#         half_width = track.get_track_width() / 2
#         offsets_full = np.clip(offsets_full, -half_width, half_width)

#         x_center = centerline[:, 0]
#         y_center = centerline[:, 1]

#         dx = np.gradient(x_center)
#         dy = np.gradient(y_center)
#         mag = np.sqrt(dx**2 + dy**2) + 1e-9

#         tx = dx / mag
#         ty = dy / mag
#         nx = -ty
#         ny = tx 

#         n_full = 11780

#         # Upsample offsets to match full resolution
#         offsets_full = np.interp(
#             np.linspace(0, len(offsets) - 1, n_full),
#             np.arange(len(offsets)),
#             offsets
#         )

#         x_ctrl = x_center + offsets_full * nx
#         y_ctrl = y_center + offsets_full * ny

#         tck, _ = splprep([x_ctrl, y_ctrl], s=0, k=3)
#         u_fine = np.linspace(0, 1, self.sample_count)

#         x_spline, y_spline = splev(u_fine, tck)

#         trajectory = self.resample_uniform(np.array(x_spline), np.array(y_spline))

#         return trajectory


import numpy as np
from scipy.interpolate import splprep, splev
from typing import Tuple

class TrajectoryBuilder:
    """
    Build a smooth, uniformly-sampled trajectory from a centerline and lateral offsets.

    - sample_count: number of points in the final returned trajectory (uniformly resampled)
    - build_trajectory(track, centerline, offsets):
        - track: Track object (optional methods used: get_track_width())
        - centerline: (N,2) array of centerline points
        - offsets: array-like of lateral offsets (length can be smaller than N; values will be interpolated)
    """

    def __init__(self, sample_count: int = 2000):
        self.sample_count = int(sample_count)

    def resample_uniform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Resample a parametric curve (x,y) uniformly by arc length to self.sample_count points.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        if x.size < 2:
            raise ValueError("x must contain at least two points")
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.insert(np.cumsum(ds), 0, 0.0)  # length == len(x)

        total = s[-1]
        if total <= 0:
            # degenerate case: return repeated point
            x_u = np.full(self.sample_count, x[0])
            y_u = np.full(self.sample_count, y[0])
            return np.vstack([x_u, y_u]).T

        s_uniform = np.linspace(0.0, total, self.sample_count)

        x_u = np.interp(s_uniform, s, x)
        y_u = np.interp(s_uniform, s, y)

        return np.vstack([x_u, y_u]).T

    def build_trajectory(self, track, centerline: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        """
        Create a smooth trajectory (Nx2) from centerline and offsets.

        - centerline: (M,2) np.ndarray
        - offsets: length K array (K may be < M). Offsets are lateral displacements along centerline normals.
        """
        centerline = np.asarray(centerline)
        offsets = np.asarray(offsets, dtype=float)

        M = centerline.shape[0]

        # Determine target full resolution for offsets (match centerline resolution)
        n_full = M

        # Interpolate offsets to full resolution
        if len(offsets) == n_full:
            offsets_full = offsets.copy()
        else:
            ctrl_idx = np.linspace(0, len(offsets)-1, len(offsets))
            target_idx = np.linspace(0, len(offsets)-1, n_full)
            offsets_full = np.interp(target_idx, ctrl_idx, offsets)

        # Clip to track boundaries if track provides widths
        half_width = track.get_track_width() / 2.0
        if np.isscalar(half_width) or (isinstance(half_width, np.ndarray) and half_width.ndim == 1 and len(half_width) == n_full):
            # vectorized clip (works for scalar or array half_width)
            offsets_full = np.minimum(offsets_full, half_width)
            offsets_full = np.maximum(offsets_full, -half_width)
        else:
            # other shapes: use scalar fallback
            offsets_full = np.clip(offsets_full, -6.0, 6.0)

        # Compute tangent & normals on the centerline
        x_center = centerline[:, 0]
        y_center = centerline[:, 1]

        dx = np.gradient(x_center)
        dy = np.gradient(y_center)
        mag = np.sqrt(dx**2 + dy**2) + 1e-9

        tx = dx / mag
        ty = dy / mag
        nx = -ty
        ny = tx

        # Apply lateral offsets to get control points of trajectory
        x_ctrl = x_center + offsets_full * nx
        y_ctrl = y_center + offsets_full * ny

        # Fit spline through the offset control points (closed loop assumed)
        # k=3 requires at least 4 points; if less, fall back to linear interp (k=1)
        k_try = 3 if len(x_ctrl) >= 4 else 1
        try:
            tck, _ = splprep([x_ctrl, y_ctrl], s=0, k=k_try, per=True)
            u_fine = np.linspace(0.0, 1.0, max(self.sample_count, n_full))
            x_spline, y_spline = splev(u_fine, tck)
        except Exception:
            # fallback: no spline (e.g., degenerate data), just use control points repeated/resampled
            x_spline = np.asarray(x_ctrl)
            y_spline = np.asarray(y_ctrl)

        # Finally, resample uniformly by arc length to requested sample_count
        trajectory = self.resample_uniform(np.array(x_spline), np.array(y_spline))

        return trajectory