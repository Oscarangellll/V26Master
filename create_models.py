import pandas as pd
from scipy.stats import boxcox
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
import statsmodels.api as sm
import pickle

df_weather = pd.read_csv("data.csv", index_col="time", parse_dates=True)
df_price = pd.read_csv("electricity_price_data.csv", index_col="time", parse_dates=True)
df_price = df_price[df_price.index.year >= 2023]

models = {}

for loc, df_loc in df_weather.groupby("locationID"):
    df = df_loc[["speed"]].join(df_price, how="inner")
    
    models[loc] = {"electricity": {}}

    y = df.to_numpy(copy=True)
    
    month_idx = df.index.month.to_numpy()
    for m in range(1, 13):
        idx = month_idx == m
        
        X = sm.add_constant(y[idx, 0])
        models[loc]["electricity"][m] = sm.OLS(y[idx, 1], X).fit()
        
    y = df_loc[["speed", "height"]].to_numpy(copy=True)

    models[loc]["weather"] = {
        "boxcox": {},
        "monthly_mean": {},
        "monthly_std": {}
    }
    
    y[:, 0], models[loc]["weather"]["boxcox"]["Speed"] = boxcox(y[:, 0])
    y[:, 1], models[loc]["weather"]["boxcox"]["Height"] = boxcox(y[:, 1])
   
    month_idx = df_loc.index.month.to_numpy()
    for m in range(1, 13):
        idx = month_idx == m
        
        mu = y[idx].mean(axis=0)
        std = y[idx].std(axis=0)

        models[loc]["weather"]["monthly_mean"][m] = {
            "Speed": mu[0],
            "Height": mu[1]
        }
        models[loc]["weather"]["monthly_std"][m] = {
            "Speed": std[0],
            "Height": std[1]
        }

        y[idx] = (y[idx] - mu) / std 
    
    models[loc]["weather"]["model"] = VAR(y).fit(maxlags=20, ic='bic')

with open("models.pkl", "wb") as f:
    pickle.dump(models, f)
