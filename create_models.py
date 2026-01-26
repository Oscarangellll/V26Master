import pandas as pd
from scipy.stats import boxcox
from statsmodels.tsa.vector_ar.var_model import VAR
import statsmodels.api as sm
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df_weather = pd.read_csv("Data/Weather Data/weather_data_2015_2025.csv", index_col="time", parse_dates=True)
df_price = pd.read_csv("electricity_price_data.csv", index_col="time", parse_dates=True)

# Filter prices for 2023 onwards
df_price = df_price[df_price.index.year >= 2023]

# Resample to daily averages
daily_prices = df_price.resample('D').mean()
daily_wind = df_weather.resample('D').mean()

# Merge on dates (only keep dates present in both)
daily_data = pd.merge(daily_prices, daily_wind, left_index=True, right_index=True, how='inner')

# Linear regression: price = a + b * speed
x = daily_data["speed"].values
y = daily_data["price"].values

# Fit line using np.polyfit (degree 1)
b, a = np.polyfit(x, y, 1)  # np.polyfit returns [slope, intercept]

print(f"Linear fit: price = {a:.2f} + {b:.2f} * speed")

# Scatter plot with regression line
plt.figure(figsize=(12, 6))
plt.scatter(x, y, alpha=0.6, label='Data')
plt.plot(x, a + b * x, color='red', label=f'Fit: price = {a:.2f} + {b:.2f}*speed')
plt.title('Daily Average Power Prices vs Wind Speed (2023 onwards)')
plt.xlabel('Daily Wind Speed (m/s)')
plt.ylabel('Daily Average Power Price')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

exit()

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
