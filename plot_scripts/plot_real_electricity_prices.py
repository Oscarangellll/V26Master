import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from plot_scripts.config import FIGWIDTH, PLOT_DIR


def plot_real_electricity_prices():
    isos = ["DEU", "GBR", "NLD"]

    df = pd.read_csv("data/price/original.csv")
    df = df[df["ISO3 Code"].isin(isos)].copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df = df[df.index.year >= 2019]

    fig, ax = plt.subplots(figsize=(FIGWIDTH / 2.54, 2.5))

    for iso3 in isos:
        group = df[df["ISO3 Code"] == iso3]
        ax.plot(group.index, group["Price (EUR/MWhe)"], label=iso3, linewidth=0.8)

    ax.set_ylabel("Price [EUR/MWh]")
    ax.legend(loc="upper left")

    fig.tight_layout()
    output_dir = Path(PLOT_DIR) / "weather_validation_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "real_electricity_prices.svg")
    plt.close(fig)


if __name__ == "__main__":
    plot_real_electricity_prices()
