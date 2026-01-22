import numpy as np 
import pandas as pd

def gen_powerprices(seed, wdf, electricity_params):
    #electricity_params is a dict with 12 LR models: one per month
    #we must generate power prices based on wind speeds in wdf and the month column in wdf
    edf = pd.DataFrame({
        "month": wdf["month"].to_numpy(),
        "day": wdf["day"].to_numpy(),
        "hour": wdf["hour"].to_numpy(),
        "day_id": wdf["day_id"].to_numpy(),
        "power_price": np.nan
    })
    
    rng = np.random.default_rng(seed)
    for month in sorted(edf["month"].unique()):
        mask = (edf["month"] == month)
        wdf_month = wdf.loc[mask]

        LR_model = electricity_params[month]["model"]
        sigma = float(electricity_params[month]["monthly_sigma"])  # standard deviation of residuals
        
        X = wdf_month[["speed"]].to_numpy() # (n,1)
        mu = LR_model.predict(X).reshape(-1) # (n,)
        eps = rng.normal(0.0, sigma, size=mask.sum())
        
        edf.loc[mask, "power_price"] = mu + eps   
        
    return edf

# Det under er mer for å trene en modell, ikke del av funksjonen over
# #load historical wind speeds and power prices
# ws = pd.read_csv("real_ws.csv")
# pp = pd.read_csv("real_pp.csv")
# X = ws["Wind Speed (m/s)"].values.reshape(-1, 1)
# Y = pp["Price (EUR/MWhe)"].values

# #train a linear regression model
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(X, Y)

# #save the model
# import joblib
# joblib.dump(model, "power_price_model.pkl")

# #load the model and make predictions
# model = joblib.load("power_price_model.pkl")

# return model.predict(wind_speeds.reshape(-1, 1))