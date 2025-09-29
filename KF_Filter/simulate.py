# kf_cv2d/simulate.py
from __future__ import annotations
import numpy as np

def simulate_cv2d(T: float = 20.0, dt: float = 0.1, meas_std: float = 0.5):
    """Simple CV target: starts at (0,0), v=(1.0, 0.5) m/s. Returns t, truth_x, meas_z."""
    steps = int(T / dt)
    t = np.arange(steps) * dt

    x_true = np.zeros((steps, 4))  # [px, py, vx, vy]
    z_meas = np.zeros((steps, 2))

    state = np.array([0.0, 0.0, 1.0, 0.5])
    R = meas_std * np.eye(2)

    for k in range(steps):
        # ground truth propagation
        state[0] += state[2] * dt
        state[1] += state[3] * dt
        x_true[k] = state.copy()

        # noisy position measurement
        z_meas[k] = state[:2] + R @ np.random.randn(2)

    return t, x_true, z_meas

