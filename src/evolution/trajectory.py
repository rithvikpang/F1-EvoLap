"""
convert centerline + lateral offsets (control points) into full path coordinates (Nx2).
"""

import numpy as np

class Trajectory:
    def __init__(self, centerline: np.ndarray, n_control=30, offsets=None):
        """
        centerline: (M,2) ndarray of centerline points (smoothed)
        n_control: number of control points of offsets (coarse)
        offsets: array-like of length n_control (meters). If None, zeros.
        """
        self.centerline = np.asarray(centerline)
        self.M = len(self.centerline)
        self.n_control = n_control
        self.idx_control = np.round(np.linspace(0, self.M-1, n_control)).astype(int)
        if offsets is None:
            self.offsets = np.zeros(n_control, dtype=float)
        else:
            self.offsets = np.asarray(offsets, dtype=float)
        self._compute_normals()
        self._build_path()

    def _compute_normals(self):
        pts = self.centerline
        diffs = np.diff(pts, axis=0, append=[pts[0]])
        tang = diffs / (np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-9)
        # normals pointing to the left side
        self.normals = np.column_stack([-tang[:,1], tang[:,0]])

    def _build_path(self):
        # Interpolate offsets across entire centerline
        full_offsets = np.interp(np.arange(self.M), self.idx_control, self.offsets)
        self.path = self.centerline + (full_offsets[:,None] * self.normals)

    def update_offsets(self, offsets):
        self.offsets = np.asarray(offsets, dtype=float)
        self._build_path()

    def get_path(self):
        return self.path.copy()