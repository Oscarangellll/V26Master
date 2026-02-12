import numpy as np
import pandas as pd

class PriceModel:
    def __init__(self, case):
        self.case = case

        self._models = {}
        
        self._fit()



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

        for iso3, weather_location_ids in self.case.ISO_Codes.items():

            df_iso3 = df_price[df_price["ISO3"] == iso3]

            df_weather_iso = (
                df_weather[df_weather["weather_location_id"].isin(weather_location_ids)]
                .groupby("weather_location_id")[["speed"]]
                .resample("D")
                .mean()
                .groupby(level="time")
                .mean()
            )

            df = df_iso3.join(df_weather_iso, how="inner")
            
            month_of_obs = df.index.month.to_numpy() - 1

            y = df["price"].to_numpy(copy=True)

            X = np.empty((len(df), 2))
            X[:, 0] = np.ones(len(df))
            X[:, 1] = df["speed"].to_numpy(copy=True)
            
            B = np.empty((12, 2))
            sigma = np.empty(12)
                
            for m in range(12):
                idx = month_of_obs == m

                #B[m - 1] = np.linalg.inv(X[idx].T @ X[idx]) @ (X[idx].T @ y[idx])
                B[m], sum_sq_res, *_ = np.linalg.lstsq(X[idx], y[idx])

                sigma[m] = sum_sq_res[0] / (np.count_nonzero(idx) - 2)
            
            self._models[iso3] = {"B": B, "sigma": sigma} 
    
    def simulate(self, s, iso3, iso3_wind_speeds, months, days_per_month):
        rng = np.random.default_rng(s)

        model = self._models[iso3]
        
        months = (pd.to_datetime(months, format="%b").month).to_numpy() - 1
        month_of_sim = np.repeat(months, days_per_month)
        
        T = len(month_of_sim)
        y_sim = np.empty(T)
        n_locations = iso3_wind_speeds.shape[1]
        X = np.empty((T, n_locations + 1))
        X[:, 0] = np.ones(T)
        X[:, 1:] = iso3_wind_speeds

        B = model["B"]
        sigma = model["sigma"]

        for m in range(12):
            idx = month_of_sim == m
            
            eps = rng.normal(scale=np.sqrt(sigma[m]), size=np.count_nonzero(idx))
            y_sim[idx] = X[idx] @ B[m] + eps 
        
        return y_sim


model = PriceModel()