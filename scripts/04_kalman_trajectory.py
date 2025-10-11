import numpy as np
import os
from typing import List, Tuple, Optional
from math import hypot
import numpy.linalg as la
import matplotlib.pyplot as plt
import time

# Kalman filter trajectory estimation for comet detections

"""
This script implements a simple linear Kalman Filter for smoothing and predicting
2D positions (image coordinates) of comet detections across frames. It is robust
to missing detections and includes utilities to scan a directory of `.npy` files
and build a time-ordered detection sequence.
"""

# ----------------------------
# 1) Utilities to find and load .npy detection files
# ----------------------------

def find_npy_files(root: str) -> List[str]:
    """Recursively find .npy files under root, sorted by path/name."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith('.npy'):
                files.append(os.path.join(dirpath, fn))
    files.sort()
    return files

def load_detections_np(path: str) -> Optional[np.ndarray]:
    """Load a .npy file and return an (M,2) array of (x,y) detections."""
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f'Failed to load {path}:', e)
        return None

    if isinstance(data, np.ndarray) and data.dtype == object:
        try:
            arr = np.vstack([np.asarray(x) for x in data if x is not None])
            if arr.ndim == 1 and arr.size == 2:
                return arr.reshape(1,2)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2].astype(float)
        except Exception:
            return None

    data = np.asarray(data)
    if data.size == 0:
        return None
    if data.ndim == 1 and data.size == 2:
        return data.reshape(1,2).astype(float)
    if data.ndim == 2 and data.shape[1] >= 2:
        return data[:, :2].astype(float)
    try:
        arr = np.vstack(data)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2].astype(float)
    except Exception:
        pass
    return None

# ----------------------------
# 2) Build a single-track sequence from detection files
# ----------------------------

def build_track_from_files(file_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a single track: choose the detection nearest previous estimate.
    Insert np.nan for missing detections.
    """
    times = np.arange(len(file_list))
    obs = np.full((len(file_list), 2), np.nan, dtype=float)

    last_pos = None
    for i, f in enumerate(file_list):
        dets = load_detections_np(f)
        if dets is None or dets.shape[0] == 0:
            continue
        if last_pos is None:
            chosen = dets[0]
            obs[i] = chosen
            last_pos = chosen
        else:
            dists = np.linalg.norm(dets - last_pos.reshape(1,2), axis=1)
            idx = np.argmin(dists)
            if dists[idx] > 50:  # gating threshold
                continue
            chosen = dets[idx]
            obs[i] = chosen
            last_pos = chosen
    return times, obs

# ----------------------------
# 3) Kalman Filter implementation
# ----------------------------

def kalman_filter(times: np.ndarray,
                  observations: np.ndarray,
                  dt_default: float = 1.0,
                  R_std: float = 2.0,
                  process_var: float = 1.0):
    """Run a constant-velocity linear Kalman Filter on 2D observations."""
    T = len(times)
    states = np.zeros((T, 4))
    covs = np.zeros((T, 4, 4))

    # Initialize state
    first_idx = next((i for i in range(T) if not np.isnan(observations[i,0])), None)
    if first_idx is None:
        raise ValueError('No valid observations to initialize filter')

    state = np.array([observations[first_idx,0],
                      observations[first_idx,1],
                      0.0, 0.0])
    P = np.diag([10.0, 10.0, 50.0, 50.0])
    H = np.array([[1,0,0,0],[0,1,0,0]])
    R = np.eye(2) * (R_std**2)

    prev_time = times[first_idx]
    for i in range(first_idx, T):
        t = times[i]
        dt = float(t - prev_time) if (t - prev_time) != 0 else dt_default
        prev_time = t

        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        q = process_var
        Q = q * np.array([[dt**3/3, 0, dt**2/2, 0],
                          [0, dt**3/3, 0, dt**2/2],
                          [dt**2/2, 0, dt, 0],
                          [0, dt**2/2, 0, dt]])

        # Predict
        state = F.dot(state)
        P = F.dot(P).dot(F.T) + Q

        z = observations[i]
        if not np.isnan(z[0]):
            y = z - H.dot(state)
            S = H.dot(P).dot(H.T) + R
            K = P.dot(H.T).dot(la.inv(S))
            state = state + K.dot(y)
            P = (np.eye(4) - K.dot(H)).dot(P)

        states[i] = state
        covs[i] = P

    for i in range(0, first_idx):
        states[i] = np.nan
        covs[i] = np.nan

    return states, covs

# ----------------------------
# 4) Visualization
# ----------------------------

def plot_track(times, observations, states, title='Kalman tracking'):
    fig, ax = plt.subplots(figsize=(8,6))
    obs_x = observations[:,0]
    obs_y = observations[:,1]
    pred_x = states[:,0]
    pred_y = states[:,1]

    ax.plot(pred_x, pred_y, '-o', label='Kalman estimate (states)')
    ax.plot(obs_x, obs_y, 'x', label='Observations')
    for i in range(len(times)):
        if not np.isnan(obs_x[i]):
            ax.plot([obs_x[i], pred_x[i]], [obs_y[i], pred_y[i]], ':', alpha=0.5)

    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    ax.set_title(title)
    ax.legend()
    ax.invert_yaxis()
    plt.show()

# ----------------------------
# 5) Main workflow
# ----------------------------

# Replace with your local folder path containing .npy files
DATA_PATH = "./npy_detections"  

if DATA_PATH and os.path.exists(DATA_PATH):
    print(f"📁 Starting Kalman tracking process in: {DATA_PATH}")

    file_list = find_npy_files(DATA_PATH)
    print(f"🔍 Found {len(file_list)} .npy files under {DATA_PATH}")

    if len(file_list) > 0:
        start_time = time.time()
        def build_track_with_progress(file_list):
            observations, times = [], []
            total = len(file_list)
            for i, f in enumerate(file_list):
                data = load_detections_np(f)
                if data is not None and len(data) > 0:
                    obs = data[0] if data.ndim > 1 else data
                    observations.append(obs)
                else:
                    observations.append([np.nan, np.nan])
                times.append(i)
                if i % 50 == 0:
                    print(f"Processed {i}/{total} files...")
            return np.array(times), np.array(observations)

        times, observations = build_track_with_progress(file_list)
        elapsed = time.time() - start_time
        print(f"⏱ Observation building completed in {elapsed:.2f} seconds.")

        if np.all(np.isnan(observations)):
            print("⚠️ No usable detections found.")
        else:
            print("🚀 Running Kalman filter on detections...")
            states, covs = kalman_filter(times, observations, R_std=3.0, process_var=1.0)
            print("✅ Kalman filter complete.")
            plot_track(times, observations, states)

            output_dir = "./kalman_output"
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, "kalman_track_output.csv")
            np.savetxt(out_path, np.hstack([times.reshape(-1,1), observations, states]),
                       delimiter=',', header='time,obs_x,obs_y,est_x,est_y,est_vx,est_vy', comments='')
            print(f"✅ Saved CSV to: {out_path}")
    else:
        print(f"⚠️ No .npy files found in {DATA_PATH}.")
else:
    print("❌ DATA_PATH not set or folder does not exist.")
