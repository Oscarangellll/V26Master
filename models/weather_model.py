import os
import pickle

import pandas as pd
import numpy as np

class WeatherModel:
    def __init__(self):
        self.rs = 0.8
        self.rh = 0.4 
        
        self.data = pd.read_csv(
            "data/weather/2015_2025.csv",
            index_col="time",
            parse_dates=True
        )
        
        self._models = {}
        
        filename = "weather.pkl"
        filepath = f"models/{filename}"

        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                self._models = pickle.load(f)
        else:
            self._fit()
            with open(filepath, "wb") as f:
                pickle.dump(self._models, f)
        

    def _fit(self):

        for wl_id, df_wl in self.data.groupby("weather_location_id"):
            y = df_wl[["speed", "height"]].to_numpy(copy=True)
            T, K = y.shape

            month_of_obs = df_wl.index.month.to_numpy() - 1
                        
            monthly_mean = np.zeros((12, K))
            monthly_std  = np.zeros((12, K))
            for m in range(12):
                idx = month_of_obs == m

                monthly_mean[m] = y[idx].mean(axis=0)
                monthly_std[m] = y[idx].std(axis=0)

                y[idx] = (y[idx] - monthly_mean[m]) / monthly_std[m]
    
            bin_width = np.array([self.rs, self.rh]) / monthly_std.mean(axis=0)
            
            bins = []
            for k in range(K):
                bins.append(np.arange(y[:, k].min(), y[:, k].max() + bin_width[k], bin_width[k]))
            
            idx = np.empty((T, K), dtype=int)
            for k in range(K):
                idx[:, k] = np.digitize(y[:, k], bins[k]) - 1
            
            states = np.zeros(T, dtype=int)
            multiplier = 1
            for k in reversed(range(K)):
                states += idx[:, k] * multiplier
                multiplier *= (len(bins[k]) - 1)

            init_states, counts = np.unique(states, return_counts=True)
            init_probs = counts / counts.sum()
            
            N = multiplier
            P = np.zeros((N, N))
            np.add.at(P, (states[:-1], states[1:]), 1)

            row_sums = P.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            P /= row_sums

            self._models[wl_id] = {
                "monthly_mean": monthly_mean,
                "monthly_std": monthly_std,
                "bins": bins,
                "init_states": init_states,
                "init_probs": init_probs,
                "N": N,
                "P": P,
            }

    def simulate(self, wl_id, seed, months=None, days_per_month=None):
        rng = np.random.default_rng(seed=seed)

        model = self._models[wl_id]

        if months is not None and days_per_month is not None:
            months = (pd.to_datetime(months, format="%b").month).to_numpy() - 1
            month_of_sim = np.repeat(months, 24 * days_per_month)
        else:
            df_wl = self.data[self.data["weather_location_id"] == wl_id]
            month_of_sim = df_wl.index.month.to_numpy() - 1

        T, K = len(month_of_sim), 2

        init_state = rng.choice(model["init_states"], p=model["init_probs"])
        
        sim_states = np.empty(T, dtype=int)
        sim_states[0] = init_state
        
        N = model["N"]
        P = model["P"]
        for t in range(1, T):
            sim_states[t] = rng.choice(N, p=P[sim_states[t - 1]])
       
        bins = model["bins"]

        sim_idx = np.zeros((T, K), dtype=int)
        tmp = sim_states.copy()
        for k in reversed(range(K)):
            sim_idx[:, k] = tmp % (len(bins[k]) - 1)
            tmp //= (len(bins[k]) - 1)

        y_sim = np.zeros((T, K))
        for k in range(K):
            i = sim_idx[:, k]
            y_sim[:, k] = rng.uniform(bins[k][i], bins[k][i + 1])

        for m in range(12):
            idx = month_of_sim == m

            y_sim[idx] = y_sim[idx] * model["monthly_std"][m] + model["monthly_mean"][m]

        return y_sim

