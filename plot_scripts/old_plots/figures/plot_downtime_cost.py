import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.fixed_data import data

def plot_downtime_cost():
    rng = np.random.default_rng()
    
    wind_farm = rng.choice(data.wind_farms)
    w = wind_farm.name
    iso = wind_farm.iso
    wl_id = wind_farm.weather_location_id

    s = rng.choice(range(1, data.n_scenarios_to_generate + 1))
   
    year = rng.integers(data.price_from_year, data.price_to_year + 1)
    
    df_p = pd.read_parquet(
        "data/price/price.parquet",
        filters=[
            ("ISO3", "==", iso)
        ]
    )
    df_p = df_p[df_p.index.year == year]

    df_w = pd.read_parquet(
        "data/weather/weather.parquet",
        filters=[
            ("weather_location_id", "==", wl_id)
        ]
    )
    df_w = df_w[df_w.index.year == year]
    df_w["power"] = data.power_curve(df_w["speed"])
    df_w = df_w[["power"]].resample("d").mean()
    
    df = df_p.join(df_w, how="inner")
    
    df["downtime_cost"] = df["power"] * 24 * df["price"]

    df_C_D = pd.read_parquet(
        "data/scenario_data/downtime_cost",
        filters=[   
            ("s", "==", s),
            ("w", "==", w)
        ]
    )
    
    fig, axs = plt.subplots(ncols=2, figsize=(15/2.54,2), sharey=True)

    axs[0].plot(df.index, df["downtime_cost"], color="blue")
    axs[0].set_title(f"Real Downtime Cost\nWind Farm: {w}, ISO: {iso}, Year: {year}")
    axs[0].set_xlabel("Date")
    axs[0].set_ylabel("Downtime Cost")
    axs[0].grid(True)

    axs[1].plot(df_C_D["d"], df_C_D["downtime_cost"], color="red")
    axs[1].set_title(f"Synthetic Downtime Cost\nWind Farm: {w}, Scenario: {s}")
    axs[1].set_xlabel("Day")
    axs[1].grid(True)

    fig.savefig("figures/plots/downtime_cost")
