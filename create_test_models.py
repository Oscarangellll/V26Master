import pandas as pd
from scipy.stats import boxcox
import numpy as np

models = {
    "weather": {
        0: {  # locationID = 0
            "model": "VARResults_object_here",   # statsmodels VARResults
            "boxcox": {
                "Speed": 0.18, 
                "Height": 0.32
            },
            "monthly_mean": {
                m: np.array([8.5, 1.6]) for m in range(12)
            },
            "monthly_std": {
                m: np.array([2.1, 0.7]) for m in range(12)
            }
        },

        1: {  # locationID = 1
            "model": "VARResults_object_here",
            "boxcox": {
                "Speed": 0.25,
                "Height": 0.30
            },
            "monthly_mean": {
                m: np.array([9.1, 1.9]) for m in range(12)
            },
            "monthly_std": {
                m: np.array([2.4, 0.9]) for m in range(12)
            }
        }
    },

    "electricity": {
        0: {  # locationID = 0
            "model": "PriceModel_object_here",  # regression / AR / VAR
            "monthly_mean": {
                m: 55.0 + m for m in range(12)
            },
            "monthly_std": {
                m: 10.0 for m in range(12)
            }
        }
    }
}

import pickle

with open("models.pkl", "wb") as f:
    pickle.dump(models, f)

"""

df = pd.read_csv("weather_data.csv", index_col="time", parse_dates=True)

for loc, df_loc in df.groupby("locationID"):
    y = df_loc[["speed", "height"]].to_numpy()
    T, K = y.shape

    month_idx = df_loc.index.month.to_numpy() - 1

    y_bc = np.empty_like(y)
    lambdas = np.empty(K)
    for k in range(K):
        print(y[:, k])

        y_bc[:, k], lambdas[k] = boxcox(y[:, k])
  
    print(y_bc)
    print(lambdas)

"""


"""
    for m in range(12):
        idx = m == month_idx
        
        for k in range(K):
            y_bc[idx, k], lambdas[k] = boxcox(y[idx, k])
"""
    

"""
# YAML for storing VAR model parameters, Box-Cox lambdas, per location and per lag order
weather:
    location:
        boxcox:
        standardization:
            0: [mean0, std0]
            1: [mean1, std1]
        VAR:
            

electricty:
    location:
        boxcox:
        
        VAR:                 # VAR coefficients and covariances
        0:               # month 0 (January)
          B: [[...], [...]]       # K x (1 + K*p) matrix
          Sigma: [[...], [...]]   # K x K covariance matrix
        1:               # month 1 (February)
          B: [[...], [...]]
          Sigma: [[...], [...]]
    electricity
    params["locations"]|[1]["VAR"][2]
  2:
    boxcox:
      0: [0.22, 0.36]
      1: [0.19, 0.31]
    PAR:
      6:
        0:
          B: [[...], [...]]
          Sigma: [[...], [...]]
      10:
        0:
          B: [[...], [...]]
          Sigma: [[...], [...]]
  # ... more locations
"""
