import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from plot_scripts.config import PLOT_DIR, FIGWIDTH


def plot_real_weather_correlation():

    df = pd.read_parquet("data/weather/weather.parquet")
    weather_location_ids = [2, 3, 4, 5]
    labels = [1, 2, 3, 4]
    df = df[df["weather_location_id"].isin(weather_location_ids)]

    # ---------------------------------
    # Create wide tables
    # ---------------------------------

    speed_table = df.pivot(
        columns="weather_location_id",
        values="speed"
    )

    height_table = df.pivot(
        columns="weather_location_id",
        values="height"
    )

    # ---------------------------------
    # Correlation matrices
    # ---------------------------------

    speed_corr = speed_table.corr()

    height_corr = height_table.corr()

    # ---------------------------------
    # Plot
    # ---------------------------------

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(FIGWIDTH / 2.54, 6 / 2.54)
    )

    im0 = axs[0].imshow(
        speed_corr,
        vmin=0,
        vmax=1
    )

    im1 = axs[1].imshow(
        height_corr,
        vmin=0,
        vmax=1
    )

    # ---------------------------------
    # Labels
    # ---------------------------------

    for ax in axs:

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))

        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

    axs[0].set_title("Wind speed")
    axs[1].set_title("Wave height")

    # ---------------------------------
    # Shared colorbar
    # ---------------------------------

    cbar = fig.colorbar(
        im1,
        ax=axs,
        shrink=0.8
    )

    cbar.set_label("Correlation")

    output_dir = Path(PLOT_DIR) / "weather_validation_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "real_weather_correlation.svg")
    plt.close(fig)
