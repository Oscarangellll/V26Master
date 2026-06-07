
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from plot_scripts.config import PLOT_DIR, colors, FIGWIDTH

def plot_real_weather_seasonality():
    df = pd.read_parquet("data/weather/weather.parquet")
    weather_location_ids = [2, 3, 4, 5]
    location_labels = {2: 1, 3: 2, 4: 3, 5: 4}
    df = df[df["weather_location_id"].isin(weather_location_ids)].copy()
    
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
            label=location_labels[loc_id],
        )

        axs[1].plot(
            group["month"],
            group["height"],
            label=location_labels[loc_id],
        )

    axs[0].set_ylabel("Wind speed [m/s]")
    axs[1].set_ylabel("Wave height [m]")
    
    for ax in axs:
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_labels, rotation=45)
    
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Location",
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=8,
        title_fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    
    output_dir = Path(PLOT_DIR) / "weather_validation_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "real_weather_seasonality.svg")
    plt.close(fig)
