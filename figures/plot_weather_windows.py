
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.fixed_data import data

def _max_consecutive(group, max_height):
    feasible = (
        group["height"] <= max_height
    ).to_numpy().astype(int)
    
    max_len = 0
    current_len = 0
    for val in feasible:
        if val:
            current_len += 1
            max_len = max(max_len, current_len)
        else:
            current_len = 0

    return max_len

def plot_weather_windows():
    rng = np.random.default_rng()
    
    wl_id = rng.choice([wl.id for wl in data.weather_locations])

    df_w = pd.read_parquet(
        "data/weather/weather.parquet",
        filters=[
            ("weather_location_id", "==", wl_id)
        ]
    )
    
    working_hours = list(range(data.work_day_start, data.work_day_end))
    df_w = df_w[df_w.index.hour.isin(working_hours)]
    
    for vessel_type in data.vessel_types:
        df_ww = (
            df_w.groupby(pd.Grouper(freq="d"))
                .apply(lambda group: _max_consecutive(group, vessel_type.max_wave))
        )
        df_ww = df_ww.to_frame(name="ww")
        df_ww.index.name = "date"


        df_ww_syn = pd.read_parquet(
            "data/scenario_data/weather_windows.parquet",
            filters=[
                ("wl_id", "==", 3),
                ("h", "==", vessel_type.name)
            ]
        )
        
        plt.hist(df_ww["ww"], alpha=0.5, label="Real", density=True)
        plt.hist(df_ww_syn["ww"], alpha=0.5, label="Synthetic", density=True)
        plt.xlabel("Weather window (hours)")
        plt.ylabel("Density")
        plt.title(f"Vessel: {vessel_type.name}, loc: {wl_id}")
        plt.legend()
        plt.show()
