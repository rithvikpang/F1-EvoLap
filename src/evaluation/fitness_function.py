import numpy as np
from copy import deepcopy
import math
from physics.lap_simulator import LapSimulator

def offtrack_penalty(path, track):
    center = track.get_centerline() if hasattr(track, 'get_centerline') else np.column_stack((track.x_center, track.y_center))
    widths = track.get_track_width()

    N_center = len(center)
    penalty = 0.0
    off_count = 0
    max_over = 0.0

    diffs = np.diff(center, axis=0, append=[center[0]])
    tang = diffs / (np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-9)
    normals = np.column_stack([-tang[:,1], tang[:,0]])

    for p in path:
        idx = int(np.argmin(np.linalg.norm(center - p, axis=1)))
        lateral = np.dot((p - center[idx]), normals[idx])
        if widths is None:
            try:
                left = np.array([track.x_left[idx], track.y_left[idx]])
                half_width = np.linalg.norm(left - center[idx])
            except Exception:
                half_width = 6.0
        else:
            if len(widths) == N_center:
                half_width = widths[idx] / 2.0
            else:
                half_width = float(np.mean(widths)) / 2.0

        if abs(lateral) > half_width:
            over = abs(lateral) - half_width
            penalty += over**2
            off_count += 1
            if over > max_over:
                max_over = over

    return {
        "penalty_value": float(penalty),
        "off_count": int(off_count),
        "max_over": float(max_over)
    }

def smoothness_penalty(path):
    p = np.array(path)
    if len(p) < 5:
        return 0.0
    v1 = p[1:] - p[:-1]
    angles = np.arctan2(v1[:,1], v1[:,0])
    ang_diff = np.diff(angles)
    ang_diff = (ang_diff + np.pi) % (2*np.pi) - np.pi
    return float(np.mean(ang_diff**2))

def evaluate_trajectory(path, track, vehicle, penalty_weights=None, debug=False):
    if penalty_weights is None:
        penalty_weights = {"offtrack": 100.0, "smoothness": 1.0}

    # protect vehicle copy
    try:
        vcopy = deepcopy(vehicle)
    except Exception:
        vcopy = vehicle

    res = LapSimulator(vcopy, path)
    time, _ = res.simulate()

    # compute penalties
    off = offtrack_penalty(path, track)
    smooth = smoothness_penalty(path)

    total_penalty = penalty_weights.get("offtrack", 0.2) * off["penalty_value"] + \
                    penalty_weights.get("smoothness", 0.3) * smooth

    fitness = time + total_penalty

    # Reject NaNs early
    if np.isnan(path).any():
        return {
            "lap_time": 1e6,
            "fitness": 1e6,
            "penalties": {"nan_path": 1},
            "speed_trace": None
        }

    return {
        "lap_time": float(time),
        "fitness": float(fitness),
        "penalties": {"offtrack": off, "smoothness": smooth}
    }