import os 
from pathlib import Path

import pandas as pd

from data.fixed_data import data
from scenario_models import PriceModel, WeatherModel

scenario_data_dir = Path(os.environ.get("SCENARIO_DATA_DIR", "data/scenario_data"))

def _generate_weather(rng, scenarios):
    
    # Weather
    # | wl_id | d | hour | s | speed | height |

    wm = WeatherModel()

    for s in scenarios:
        df = wm.simulate(rng)

        df["s"] = s
        
        df.to_parquet(
            scenario_data_dir / "weather",
            partition_cols=["s"],
            basename_template="part-{i}",
            existing_data_behavior="overwrite_or_ignore"
        )


def _generate_prices(rng, scenarios):
    
    # Price
    # | iso | d | s | price |

    pm = PriceModel()
    
    isos = {w.iso for w in data.wind_farms}

    for s in scenarios:
        df_weather_sim = pd.read_parquet(
            scenario_data_dir / "weather",
            filters=[("s", "==", s)]
        )
        
        rows = []
        for iso in isos:
            wl_ids = sorted(
                {w.weather_location_id for w in data.wind_farms if w.iso == iso}
            )
            
            df_iso = df_weather_sim[df_weather_sim["wl_id"].isin(wl_ids)]
            
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
        df.to_parquet(
            scenario_data_dir / "price",
            partition_cols=["s"],
            basename_template="part-{i}",
            existing_data_behavior="overwrite_or_ignore"
        )
            

def _generate_failures(rng, scenarios):

    # Failures 
    # | w | m | d | s | failures |

    p = [m.failure_rate / 365 for m in data.maintenance_categories]
    p.append(1 - sum(p))

    for s in scenarios:

        rows = []
        for w in data.wind_farms:
            draws = rng.multinomial(    
                w.n_turbines,
                p,
                size=(data.days_per_period * len(data.periods))
            )
            for d in range(data.days_per_period * len(data.periods)):
                for m_idx, m in enumerate(data.maintenance_categories):
                    rows.append({
                        "w": w.name,
                        "m": m.name,
                        "d": d + 1,
                        "s": s,
                        "failures": m.scale * draws[d, m_idx]
                    })

        df = pd.DataFrame(rows)
        df.to_parquet(
            scenario_data_dir / "failures",
            partition_cols=["s"],
            basename_template="part-{i}",
            existing_data_behavior="overwrite_or_ignore"
        )


def generate_scenarios(rng, scenarios):
    
    print("Generating weather")
    _generate_weather(rng, scenarios)

    print("Generating prices")
    _generate_prices(rng, scenarios)

    print("Generating failures")
    _generate_failures(rng, scenarios)




