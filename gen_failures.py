import numpy as np
import pandas as pd

def gen_failures(seed, wf, maintenance_categories):
    rng = np.random.default_rng(seed)
    
    p = np.array([m.failure_rate / 360 for m in maintenance_categories], dtype=float)
    if p.sum() > 1.0:
        raise ValueError("Sum of daily component failure probabilities > 1.")
    
    p0 = 1.0 - p.sum()  # no-failure probability
    p_all = np.append(p, p0)  # length K+1
    
    # Draw all days at once: shape (n_days, K+1)
    draws = rng.multinomial(int(wf.n_turbines), p_all, size=360)
    
    # Build long DataFrame for components only (exclude last column = no-failure)
    K = len(maintenance_categories)
    day_id = np.arange(1, 360 + 1)
    
    fdf = pd.DataFrame({
        "day_id": np.repeat(day_id, K),
        "component": np.tile([m.name for m in maintenance_categories], 360),
        "n_failures": draws[:, :K].reshape(-1)
    })

    return fdf