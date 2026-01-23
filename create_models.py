import pandas as pd
from scipy.stats import boxcox
from statsmodels.tsa.vector_ar.var_model import VAR
import statsmodels.api as sm
import pickle

df_weather = pd.read_csv("data.csv", index_col="time", parse_dates=True)
df_price_daily = pd.read_csv("germany_electricity_price_daily_2023_2025.csv", index_col="Date", parse_dates=True)
df_price_daily = df_price_daily[df_price_daily.index.year >= 2023]
models = {}

for loc, df_loc in df_weather.groupby("locationID"):
    df_daily = df_loc.resample("D").mean()
    df = df_daily[["speed"]].join(df_price_daily, how="inner")
    
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
    
    y[:, 0], models[loc]["weather"]["boxcox"]["speed"] = boxcox(y[:, 0])
    y[:, 1], models[loc]["weather"]["boxcox"]["height"] = boxcox(y[:, 1])

    month_idx = df_loc.index.month.to_numpy()
    for m in range(1, 13):
        idx = month_idx == m
        
        mu = y[idx].mean(axis=0)
        std = y[idx].std(axis=0)

        models[loc]["weather"]["monthly_mean"][m] = mu 
        models[loc]["weather"]["monthly_std"][m] = std 

        y[idx] = (y[idx] - mu) / std 
    
    models[loc]["weather"]["model"] = VAR(y).fit(maxlags=20, ic='bic')

with open("models.pkl", "wb") as f:
    pickle.dump(models, f)
