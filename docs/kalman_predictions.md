# Kalman Trajectory Script Overview

This script implements a **2D Kalman Filter** for estimating and smoothing the trajectory of comets from image detections. It works with `.npy` detection files containing (x, y) coordinates.

The main workflow is as follows:

---

## 1. Load Detection Files

### Utilities:

- **`find_npy_files(root)`**: Recursively finds all `.npy` files under `root`, sorted alphabetically.
- **`load_detections_np(path)`**: Attempts to load `.npy` files and return an array of `(x, y)` detections.

#### Supported formats:

- Single `(x, y)` pair
- `(N, 2)` array of multiple detections
- Object arrays of dictionaries or tuples

Returns `None` if the file is empty or cannot be parsed.

---

## 2. Build a Single Detection Track

**Function:** `build_track_from_files(file_list)`

- Goal: Build a **time ordered sequence of detections** for one comet.
- Method:  
  1. For each frame, select the detection closest to the previous position.
  2. If no detection is available or distance exceeds a threshold, insert `NaN`.
- Output:
  - `times`: array of frame indices
  - `observations`: `(T, 2)` array of `(x, y)` with `NaN` for missing frames

---

## 3. Kalman Filter Implementation

**Function:** `kalman_filter(times, observations, dt_default=1.0, R_std=2.0, process_var=1.0)`

- **State vector:** `[x, y, vx, vy]`  
- **Observation vector:** `[x, y]`  

### Steps:

1. **Initialization**
   - Use first valid observation for position.
   - Set initial velocity to zero.
   - Initial covariance: `P = diag([10, 10, 50, 50])`

2. **Prediction**
   - State transition for constant velocity model:

     ```
     F = [[1, 0, dt, 0],
          [0, 1, 0, dt],
          [0, 0, 1, 0],
          [0, 0, 0, 1]]
     ```

   - Process noise `Q`:

     ```
     Q = q * [[dt**3/3, 0, dt**2/2, 0],
              [0, dt**3/3, 0, dt**2/2],
              [dt**2/2, 0, dt, 0],
              [0, dt**2/2, 0, dt]]
     ```

   - Predicted state: `state = F @ state`
   - Predicted covariance: `P = F @ P @ F.T + Q`

3. **Update**
   - If observation exists:
     - Innovation: `y = z - H @ state`
     - Innovation covariance: `S = H @ P @ H.T + R`
     - Kalman gain: `K = P @ H.T @ inv(S)`
     - Update state: `state = state + K @ y`
     - Update covariance: `P = (I - K @ H) @ P`

- Returns:  
  - `states`: `(T,4)` array of `[x, y, vx, vy]`  
  - `covs`: `(T,4,4)` covariance matrices

---

## 4. Visualization

**Function:** `plot_track(times, observations, states)`

- Plots:
  - Observed detections (`x`)
  - Kalman filter predictions (`o`)
  - Lines connecting observations to predictions
- Axes:
  - `x` and `y` in pixels
  - `y-axis` inverted (image coordinates)

---

## 5. Running the Script

```python
# 1. Set path to your data folder
DATA_PATH = "/path/to/your/npy/folder"

# 2. Find .npy files
file_list = find_npy_files(DATA_PATH)
print(f"Found {len(file_list)} .npy files")

# 3. Build observation track
def build_track_from_files_with_progress(file_list):
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

times, observations = build_track_from_files_with_progress(file_list)

# 4. Run Kalman filter
states, covs = kalman_filter(times, observations, R_std=3.0, process_var=1.0)

# 5. Visualize results
plot_track(times, observations, states)

# 6. Save to CSV
output_dir = "kalman_data"
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, "kalman_track_output.csv")
np.savetxt(out_path, np.hstack([times.reshape(-1,1), observations, states]),
           delimiter=',', header='time,obs_x,obs_y,est_x,est_y,est_vx,est_vy', comments='')
print(f"Saved CSV of times, observations, and estimates to:\n{out_path}")
