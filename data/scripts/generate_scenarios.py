import pandas as pd
import numpy as np

from data.fixed_data import data
from scenario_models.weather_model import WeatherModel
from scenario_models.price_model import PriceModel 

def _generate_weather(rng, scenarios):
    
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

def _generate_prices(rng, scenarios):
    
    # Price
    # | iso | d | s | price |

    pm = PriceModel()
    
    df_weather_sim = pd.read_parquet("data/scenario_data/weather.parquet")

    isos = {w.iso for w in data.wind_farms}

    rows = []
    for iso in isos:
        wl_ids = sorted(
            {w.weather_location_id for w in data.wind_farms if w.iso == iso}
        )
        for s in scenarios:
            df_iso = df_weather_sim[
                (df_weather_sim["wl_id"].isin(wl_ids)) & (df_weather_sim["s"] == s)
            ]
            
            df_iso = (
                df_iso.groupby(["wl_id", "d"])["speed"]
                .mean()
                .reset_index()
            )

            df_iso = df_iso.pivot(index="d", columns="wl_id", values="speed")
            df_iso = df_iso[wl_ids]
            
            speed = df_iso.to_numpy()
            
            df = pm.simulate(speed, iso, rng)
            df["s"] = s
            rows.append(df)

    df = pd.concat(rows, ignore_index=True)
    df = df[["iso", "d", "s", "price"]]

    df.to_parquet("data/scenario_data/price.parquet")
    df.to_csv("data/scenario_data/price.csv", index=False)

def _generate_failures(rng, scenarios):

    # Failures 
    # | w | m | d | s | failures |

    p = [m.failure_rate / 365 for m in data.maintenance_categories]
    p.append(1 - sum(p))

    rows = []
    for w in data.wind_farms:
        draws = rng.multinomial(    
            w.n_turbines,
            p,
            size=(len(scenarios), data.days_per_period * len(data.periods))
        )
        for s_idx, s in enumerate(scenarios):
            for d in range(data.days_per_period * len(data.periods)):
                for m_idx, m in enumerate(data.maintenance_categories):
                    rows.append({
                        "w": w.name,
                        "m": m.name,
                        "d": d + 1,
                        "s": s,
                        "failures": draws[s_idx, d, m_idx]
                    })

    df = pd.DataFrame(rows)

    df.to_parquet("data/scenario_data/failures.parquet")
    df.to_csv("data/scenario_data/failures.csv", index=False)



def generate_scenarios():
    
    rng = np.random.default_rng(seed=data.generate_scenarios_seed)
    scenarios = [s for s in range(1, data.n_scenarios_to_generate + 1)]


    _generate_weather(rng, scenarios)

    _generate_prices(rng, scenarios)

    _generate_failures(rng, scenarios)




