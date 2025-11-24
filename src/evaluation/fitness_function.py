import numpy as np
from copy import deepcopy
import math
from physics.lap_simulator import LapSimulator

def estimate_laptime_from_trajectory(traj, track, vehicle):
    traj = np.asarray(traj)
    if traj.ndim != 2 or traj.shape[1] != 2:
        raise ValueError("trajectory must be Nx2")

    # segment lengths
    ds = np.linalg.norm(np.diff(traj, axis=0, append=[traj[0]]), axis=1)
    total_len = ds.sum()

    # get track centerline & curvature
    if hasattr(track, 'get_centerline'):
        center = track.get_centerline()
    else:
        center = np.column_stack((track.x_center, track.y_center))
    if hasattr(track, 'get_curvature'):
        curvature = track.get_curvature()
    else:
        # approximate curvature from centerline if not provided
        dx = np.gradient(center[:,0])
        dy = np.gradient(center[:,1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = (dx*ddy - dy*ddx) / (dx**2 + dy**2 + 1e-12)**1.5

    # vehicle params
    g = getattr(vehicle, 'g', 9.81)
    mu = getattr(vehicle, 'mu', None)
    max_speed = getattr(vehicle, 'max_speed', None) or getattr(vehicle, 'P_max', None)
    # if P_max present but max_speed absent we'll clamp later

    # nearest centerline index for each traj point
    idxs = []
    for p in traj:
        idxs.append(int(np.argmin(np.linalg.norm(center - p, axis=1))))
    idxs = np.array(idxs)

    # compute v_target per segment using curvature at start point of segment
    v_targets = np.zeros(len(ds))
    for i in range(len(ds)):
        idx = idxs[i]
        k = curvature[idx] if idx < len(curvature) else curvature[-1]
        if k == 0 or mu is None:
            v_corner = max_speed if max_speed is not None else 20.0  # fallback
        else:
            R = abs(1.0 / k) if abs(k) > 1e-6 else 1e6
            if mu is None:
                mu_est = 1.0
            else:
                mu_est = mu
            v_corner = math.sqrt(max(0.0, mu_est * g * R))
        # safety factor and clamp
        if max_speed is not None:
            v_targets[i] = min(v_corner * 0.95, max_speed)
        else:
            v_targets[i] = v_corner * 0.95
        # avoid zeros
        if v_targets[i] < 0.5:
            v_targets[i] = 0.5

    # compute time
    time = float(np.sum(ds / v_targets))
    return time

def offtrack_penalty(path, track):
    center = track.get_centerline() if hasattr(track, 'get_centerline') else np.column_stack((track.x_center, track.y_center))
    widths = None
    if hasattr(track, "get_track_width"):
        try:
            widths = track.get_track_width()
        except Exception:
            widths = None

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
    # print("TIME: ", time)

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