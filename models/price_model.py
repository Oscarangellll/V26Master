import os
from dataclasses import dataclass
import pickle

import numpy as np
import pandas as pd

@dataclass
class _LocationPriceModel:
    # OLS parameters per month
    B: np.ndarray # shape: (12, 2)
    sigma: np.ndarray # shape: (12,)

class PriceModel:
    def __init__(self):
        self._models = {}

        if os.path.exists("models/price.pkl"):
            with open("models/price.pkl", "rb") as f:
                self._models = pickle.load(f)
        else:
            self._fit()
            with open("models/price.pkl", "wb") as f:
                pickle.dump(self._models, f)

    def _fit(self):
        df_weather = pd.read_csv(
            "data/weather/2015_2025.csv", 
            index_col="time", 
            parse_dates=True
        )
        
        df_price = pd.read_csv(
            "data/electricity/processed.csv", 
            index_col="date", 
            parse_dates=True
        )
            
        for locID, df_price_loc in df_price.groupby("locationID"):
            df_weather_loc = df_weather[df_weather["locationID"] == locID]
            df_weather_loc = df_weather_loc[["speed", "height"]].resample("D").mean()

            df_loc = df_price_loc.join(df_weather_loc, how="inner")

            month_of_obs = df_loc.index.month.to_numpy()

            y = df_loc["price"].to_numpy()

            X = np.empty((len(df_loc), 2))
            X[:, 0] = np.ones(len(df_loc))
            X[:, 1] = df_loc["speed"].to_numpy()
            
            B = np.empty((12, 2))
            sigma = np.empty(12)
                
            for m in range(1, 13):
                idx = month_of_obs == m

                #B[m - 1] = np.linalg.inv(X[idx].T @ X[idx]) @ (X[idx].T @ y[idx])
                B[m - 1], sum_sq_res, *_ = np.linalg.lstsq(X[idx], y[idx])

                sigma[m - 1] = sum_sq_res[0] / (np.count_nonzero(idx) - 2)
            
            self._models[locID] = _LocationPriceModel(B, sigma)
    
    def simulate(self, speed, locID, seed):
        assert len(speed) == (24 * 30 * 12)
        rng = np.random.default_rng(seed)
        
        speed = speed.reshape(30 * 12, 24).mean(axis=1)

        months = np.repeat(np.arange(1, 13), 30)
        
        y_sim = np.empty(30 * 12)

        X = np.empty((30 * 12, 2))
        X[:, 0] = np.ones(30 * 12)
        X[:, 1] = speed

        B = self._models[locID].B
        sigma = self._models[locID].sigma

        for m in range(1, 13):
            idx = months == m
            
            eps = rng.normal(scale=np.sqrt(sigma[m - 1]), size=30)
            y_sim[idx] = X[idx] @ B[m - 1] + eps 
        
        return y_sim



