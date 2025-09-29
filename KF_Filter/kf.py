# kf_cv2d/kf.py
from __future__ import annotations
import numpy as np

class KalmanFilterCV2D:
    """
    Linear KF for constant-velocity in 2D.
    State:      x = [px, py, vx, vy]^T
    Measurement z = [px, py]^T
    """

    def __init__(self,
                 x0: np.ndarray = np.zeros(4),
                 P0: np.ndarray = None,
                 q: float = 0.6,     # process noise spectral density
                 r: float = 0.5):    # position meas. std dev
        self.x = x0.astype(float).copy()
        self.P = (P0 if P0 is not None else 10.0*np.eye(4)).astype(float).copy()
        self.q = float(q)
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]], dtype=float)
        self.R = (r**2) * np.eye(2)

    @staticmethod
    def _F(dt: float) -> np.ndarray:
        return np.array([[1,0,dt,0],
                         [0,1,0,dt],
                         [0,0, 1,0],
                         [0,0, 0,1]], dtype=float)

    def _Q(self, dt: float) -> np.ndarray:
        # Continuous white acceleration noise integrated for CV model
        dt2, dt3, dt4 = dt*dt, dt*dt*dt, dt*dt*dt*dt
        q = self.q
        return q * np.array([[dt4/4,     0,   dt3/2,     0   ],
                             [    0, dt4/4,      0 ,  dt3/2 ],
                             [dt3/2,     0,     dt2,     0   ],
                             [    0,  dt3/2,     0 ,    dt2 ]], dtype=float)

    def predict(self, dt: float):
        F = self._F(dt)
        Q = self._Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x.copy(), self.P.copy()

    def update(self, z: np.ndarray):
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        return self.x.copy(), self.P.copy()

