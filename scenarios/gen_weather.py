import numpy as np
import pandas as pd
from scipy.special import inv_boxcox

def gen_weather(seed, weather_params):
    H, D, M = 24, 30, 12  # hours, days, months
    
    #####################################################
    # Generate synthetic weather data using the VAR model
    #####################################################
    VAR_model = weather_params["model"]
    sim = VAR_model.simulate_var(steps=H*D*M, seed=seed) #(H*D*M)x2 array with boxcoxed and standardized Speed and Height
    
    # Create a MultiIndex for Month, Day, Hour (instead of time series)
    idx = pd.MultiIndex.from_product(
        [range(1, M+1), range(1, D+1), range(H)], 
            names=["month", "day", "hour"]
    )
    # Create a DataFrame to hold the synthetic weather data
    sd = pd.DataFrame(index=idx).reset_index()
    sd["speed_z"] = sim[:, 0]
    sd["height_z"] = sim[:, 1]
    sd["day_id"] = D * (sd["month"] - 1) + sd["day"]  #1-360
    
    ################
    # De-standardize
    ################
    m_mu = weather_params["monthly_mean"] #dict: m: (mean_speed, mean_height) for each month m=1..12
    m_std = weather_params["monthly_std"] #dict m: (std_speed, std_height) for each month m=1..12
    
    # Vi mapper month -> mean/std ved å slå opp i dict
    speed_mean  = sd["month"].map(lambda m: m_mu[m][0]).to_numpy()  #OBS: month er 1-indexed i sd. 
    height_mean = sd["month"].map(lambda m: m_mu[m][1]).to_numpy()  #Så vi må kanskje trekke 1 (gjøres nå)
    speed_std   = sd["month"].map(lambda m: m_std[m][0]).to_numpy()  #når vi slår opp i mm og ms. 
    height_std  = sd["month"].map(lambda m: m_std[m][1]).to_numpy()  #Verify!
    
    sd["speed_bc"] = sd["speed_z"].to_numpy() * speed_std + speed_mean
    sd["height_bc"] = sd["height_z"].to_numpy() * height_std + height_mean

    ###########
    # De-Boxcox
    ###########
    lam_speed = weather_params["boxcox"]["speed"]
    lam_height = weather_params["boxcox"]["height"]
    
    sd["speed"] = inv_boxcox(sd["speed_bc"].to_numpy(), lam_speed)
    sd["height"] = inv_boxcox(sd["height_bc"].to_numpy(), lam_height)
    
    # Transformation can result in negative values, fix these
    sd = sd.sort_values(["month", "day", "hour"])
    for col in ["speed", "height"]:
        sd.loc[sd[col] < 0, col] = np.nan  # set negative values to NaN
        sd[col] = sd[col].interpolate(method='linear', limit_direction='both').clip(lower=0)  # linear interpolation og lower clip at 0
        
    return sd.drop(columns=["speed_z","height_z","speed_bc","height_bc"])  # drop intermediate columns
