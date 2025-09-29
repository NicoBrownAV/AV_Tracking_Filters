# demo.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from kf import KalmanFilterCV2D
from simulate import simulate_cv2d

def main():
    t, x_true, z_meas = simulate_cv2d(T=20.0, dt=0.1, meas_std=0.5)
    dt = t[1] - t[0]

    kf = KalmanFilterCV2D(x0=np.array([0,0,0,0], float),
                          P0=10*np.eye(4),
                          q=0.2,    # tune process noise
                          r=0.5)    # tune measurement noise std

    est = []
    for k in range(len(t)):
        kf.predict(dt)
        kf.update(z_meas[k])
        est.append(kf.x.copy())
    est = np.array(est)

    # Quick position RMSE
    rmse = np.sqrt(np.mean((est[:,:2] - x_true[:,:2])**2))
    print(f"Position RMSE: {rmse:.3f} m")

    # Plot
    plt.figure()
    plt.plot(x_true[:,0], x_true[:,1], label="truth")
    plt.scatter(z_meas[:,0], z_meas[:,1], s=10, alpha=0.4, label="meas")
    plt.plot(est[:,0], est[:,1], label="KF estimate")
    plt.axis("equal"); plt.grid(True); plt.legend()
    plt.title("2D CV Tracking with Kalman Filter")
    plt.xlabel("x [m]"); plt.ylabel("y [m]")
    plt.show()

if __name__ == "__main__":
    main()

