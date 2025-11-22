import numpy as np
import pandas as pd


# src/pyfundlib/simulation/gbm.py
def gbm_simulator(S0, T, n_paths, rng, mu=0.10, sigma=0.18, **kwargs):
    dt = 1 / 252
    steps = T
    Z = rng.normal(0, 1, (steps, n_paths))
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_returns = np.cumsum(drift + diffusion, axis=0)
    paths = S0 * np.exp(np.vstack([np.zeros(n_paths), log_returns]))
    dates = pd.bdate_range(start="today", periods=steps + 1)
    return pd.DataFrame(paths, index=dates)
