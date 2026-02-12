import numpy as np
import pandas as pd

from data import FixedData

class PriceModel:
    def __init__(self):
        self._models = {}
        
        self._fit()

    def _fit(self):
        data = FixedData()

        df_weather = pd.read_csv(
            "data/weather/2015_2025.csv", 
            index_col="time", 
            usecols=["time", "speed", "weather_location_id"],
            parse_dates=True
        )
        
        df_price = pd.read_csv(
            "data/electricity/processed.csv", 
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
   
    
    def simulate(self, speed, iso, seed, months, days_per_month):
        rng = np.random.default_rng(seed)

        model = self._models[iso]
        
        months = (pd.to_datetime(months, format="%b").month).to_numpy() - 1
        month_of_sim = np.repeat(months, days_per_month)
        
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

