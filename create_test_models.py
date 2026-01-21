import pandas as pd
from scipy.stats import boxcox
import numpy as np

models = {
    0: {  # locationID = 0
        "weather": {
            "model": "VARResults_object_here",   # statsmodels VARResults
            "boxcox": {
                "speed": 0.18, 
                "height": 0.32
            },
            "monthly_mean": {
                m: np.array([8.5, 1.6]) for m in range(12)
            },
            "monthly_std": {
                m: np.array([2.1, 0.7]) for m in range(12)
            }
        },
        "electricity": {
            1: { # month 1
                "model": "PriceModel_object_here",  # regression OLS
            },
            2: { # month 2
                "model": "PriceModel_object_here",  # regression OLS
            }
        }
    },
    1: {  # locationID = 0
        "weather": {
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
        "electricity": {
            1: { # month 1
                "model": "PriceModel_object_here",  # regression OLS
            },
            2: { # month 2
                "model": "PriceModel_object_here",  # regression OLS
            }    
        }
    }
}
   


