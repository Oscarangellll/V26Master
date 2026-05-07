
import matplotlib.pyplot as plt
import pandas as pd

from plot_scripts.config import PLOT_DIR, colors, FIGWIDTH

def plot_real_weather_seasonality():
    df = pd.read_parquet("data/weather/weather.parquet")
    
    df["month"] = df.index.month

    # Monthly average by location
    monthly = (
        df.groupby(["weather_location_id", "month"])[["speed", "height"]]
        .mean()
        .reset_index()
    )

    fig, axs = plt.subplots(1, 2, figsize=(FIGWIDTH / 2.54, 6 / 2.54), sharex=True)

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for loc_id, group in monthly.groupby("weather_location_id"):
        axs[0].plot(
            group["month"],
            group["speed"],
        )

        axs[1].plot(
            group["month"],
            group["height"],
        )

    axs[0].set_ylabel("Wind speed [m/s]")
    axs[1].set_ylabel("Wave height [m]")
    
    for ax in axs:
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_labels, rotation=45)
    
    fig.savefig(PLOT_DIR + "real_weather_seasonality")
    plt.show()
