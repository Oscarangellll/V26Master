
import pandas as pd
import numpy as np

from data.fixed_data import data
from scenario_models.weather_model import WeatherModel
from scenario_models.price_model import PriceModel 

rng = np.random.default_rng(seed=422)

scenarios = [s for s in range(1, 301)]


# Weather
# | wl_id | d | hour | s | speed | height |

wm = WeatherModel()

rows = []
for wl in data.weather_locations:
    for s in scenarios:

        df = wm.simulate(wl.id, rng)
        df["s"] = s
        rows.append(df)

df = pd.concat(rows, ignore_index=True)
df = df[["wl_id", "d", "hour", "s", "speed", "height"]]

df.to_parquet("data/scenario_data/weather.parquet")
df.to_csv("data/scenario_data/weather.csv", index=False)


# Price
# | iso | d | s | price |

pm = PriceModel()

isos = {w.iso for w in data.wind_farms}

rows = []
for iso in isos:
    wl_ids = sorted(
        {w.weather_location_id for w in data.wind_farms if w.iso == iso}
    )
    for s in scenarios:
        df_iso = df[
            (df["wl_id"].isin(wl_ids)) & (df["s"] == s)
        ]
        
        df_iso = (
            df_iso.groupby(["wl_id", "d"])["speed"]
            .mean()
            .reset_index()
        )

        df_iso = df_iso.pivot(index="d", columns="wl_id", values="speed")
        df_iso = df_iso[wl_ids]
        
        speed = df_iso.to_numpy()
        
        df_p = pm.simulate(speed, iso, rng)
        df_p["s"] = s
        rows.append(df_p)

df = pd.concat(rows, ignore_index=True)
df = df[["iso", "d", "s", "price"]]

df.to_parquet("data/scenario_data/price.parquet")
df.to_csv("data/scenario_data/price.csv", index=False)


# Failures 
# | w | m | d | s | failures |

p = [m.failure_rate / 365 for m in data.maintenance_categories]
p.append(1 - sum(p))

rows = []
for w in data.wind_farms:
    draws = rng.multinomial(    
        w.n_turbines,
        p,
        size=(len(scenarios), len(data.days))
    )
    for s_idx, s in enumerate(scenarios):
        for d_idx, d in enumerate(data.days):
            for m_idx, m in enumerate(data.maintenance_categories):
                rows.append({
                    "w": w.name,
                    "m": m.name,
                    "d": d,
                    "s": s,
                    "failures": draws[s_idx, d_idx, m_idx]
                })

df = pd.DataFrame(rows)
df.to_parquet("data/scenario_data/failures.parquet")
df.to_csv("data/scenario_data/failures.csv", index=False)

