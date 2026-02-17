import os
import pickle

import numpy as np
import pandas as pd

from data.fixed_data import data
from data.hashing import *

class PriceModel:
    def __init__(self):
        
        self._models = {}
        
        iso_codes = {w.iso for w in data.wind_farms}
        self.filehash = hash_electricity_prices(
            iso_codes,
            data.electricity_price_from_year, 
            data.electricity_price_to_year
        )
                            
        model_filepath = f"models/{self.filehash}.pkl"

        if os.path.exists(model_filepath):
            print("Reading price model from file")
            with open(model_filepath, "rb") as f:
                self._models = pickle.load(f)
        else:
            print("Fitting price model")
            self._fit()
            with open(model_filepath, "wb") as f:
                pickle.dump(self._models, f)



    def _fit(self):
        data_weather_filehash = hash_all_weather_locations(
            data.weather_locations,
            data.weather_from_year,
            data.weather_to_year
        )
        data_weather_filepath = f"data/weather/{data_weather_filehash}.csv"
        df_weather = pd.read_csv(
            data_weather_filepath,
            index_col="time", 
            usecols=["time", "speed", "weather_location_id"],
            parse_dates=True
        )
        
        data_price_filepath = f"data/electricity/{self.filehash}.csv"
        df_price = pd.read_csv(
            data_price_filepath, 
            index_col="date", 
            parse_dates=True
        )
        
        for iso, df_price_iso in df_price.groupby("ISO3"):
            weather_location_ids = sorted({
                w.weather_location_id for w in data.wind_farms if w.iso == iso 
            })
            
            df_weather_iso = df_weather[
                df_weather["weather_location_id"].isin(weather_location_ids)
            ]
            
            df_weather_iso = (
                df_weather_iso.groupby("weather_location_id")[["speed"]]
                .resample("D")
                .mean()
            )

            df_weather_iso = df_weather_iso["speed"].unstack("weather_location_id")
            
            df = df_price_iso.join(df_weather_iso, how="inner")
            
            month_of_obs = df.index.month.to_numpy() - 1

            y = df["price"].to_numpy(copy=True)
            
            X = np.column_stack([
                np.ones(len(df)),
                df[weather_location_ids].to_numpy()
            ])
            
            B = np.empty((12, 1 + len(weather_location_ids)))
            sigma = np.empty(12)
                
            for m in range(12):
                idx = month_of_obs == m

                #B[m - 1] = np.linalg.inv(X[idx].T @ X[idx]) @ (X[idx].T @ y[idx])
                B[m], sum_sq_res, *_ = np.linalg.lstsq(X[idx], y[idx])

                sigma[m] = sum_sq_res[0] / (np.count_nonzero(idx) - (1 + len(weather_location_ids)))
            
            self._models[iso] = {"B": B, "sigma": sigma}
   
    
    def simulate(self, speed, iso, rng, months=None, days_per_month=None, month_of_obs=None):
        model = self._models[iso]

        if months is not None and days_per_month is not None:
            months = (pd.to_datetime(months, format="%b").month).to_numpy() - 1
            month_of_sim = np.repeat(months, days_per_month)
        elif month_of_obs is not None:
            month_of_sim = month_of_obs
        
        
        T = len(month_of_sim)
        y_sim = np.empty(T)
        
        X = np.column_stack([
            np.ones(T),
            speed
        ])

        B = model["B"]
        sigma = model["sigma"]

        for m in range(12):
            idx = month_of_sim == m
            
            eps = rng.normal(scale=np.sqrt(sigma[m]), size=np.count_nonzero(idx))
            y_sim[idx] = X[idx] @ B[m] + eps 
        
        return y_sim

