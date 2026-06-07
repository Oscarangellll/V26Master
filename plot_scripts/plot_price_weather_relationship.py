import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_scripts.config import PLOT_DIR, FIGWIDTH, colors


COUNTRIES = {
    "DEU": [2, 3],
    "NLD": [4],
    "GBR": [5],
}
MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


def _daily_weather():
    weather = pd.read_parquet("data/weather/weather.parquet")
    daily = (
        weather.groupby("weather_location_id")[["speed"]]
        .resample("D")
        .mean()["speed"]
        .unstack("weather_location_id")
    )
    return daily


def _price_weather_data(iso, wl_ids, daily_weather, price):
    df = price[price["ISO3"] == iso].join(daily_weather[wl_ids], how="inner")
    df = df.dropna(subset=["price", *wl_ids]).copy()
    df["month"] = df.index.month - 1
    return df


def _fit_price_models(price, daily_weather):
    models = {}
    for iso, wl_ids in COUNTRIES.items():
        df = _price_weather_data(iso, wl_ids, daily_weather, price)
        y = df["price"].to_numpy(copy=True)
        X = np.column_stack([np.ones(len(df)), df[wl_ids].to_numpy()])
        months = df["month"].to_numpy()

        B = np.empty((12, 1 + len(wl_ids)))
        for month in range(12):
            idx = months == month
            B[month], *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        models[iso] = {"B": B}
    return models


def _plot_2d(ax, df, iso, wl_id, price_models, month_idx):
    x = df[wl_id].to_numpy()
    y = df["price"].to_numpy()
    coeff = price_models[iso]["B"][month_idx]

    ax.scatter(x, y, color=colors.blue, alpha=0.22, s=8, linewidths=0)

    x_line = np.linspace(np.nanpercentile(x, 2), np.nanpercentile(x, 98), 100)
    y_line = coeff[0] + coeff[1] * x_line
    ax.plot(x_line, y_line, color=colors.red, linewidth=1.4)
    ax.set_title(f"{iso}, weather zone {wl_id}", pad=4)
    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Price [EUR/MWh]")
    ax.grid(color="0.90", linewidth=0.6)
    ax.tick_params(axis="both", labelsize=7, pad=1)


def _plot_3d(ax, df, iso, wl_ids, price_models, month_idx):
    x = df[wl_ids[0]].to_numpy()
    y = df[wl_ids[1]].to_numpy()
    z = df["price"].to_numpy()

    ax.scatter(x, y, z, color=colors.blue, alpha=0.20, s=7, depthshade=False)

    coeff = price_models[iso]["B"][month_idx]
    x_grid = np.linspace(np.nanpercentile(x, 2), np.nanpercentile(x, 98), 18)
    y_grid = np.linspace(np.nanpercentile(y, 2), np.nanpercentile(y, 98), 18)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = coeff[0] + coeff[1] * xx + coeff[2] * yy
    ax.plot_surface(xx, yy, zz, color=colors.red, alpha=0.30, linewidth=0, antialiased=True)

    ax.set_title(f"{iso}, weather zones {wl_ids[0]} and {wl_ids[1]}", pad=0)
    ax.set_xlabel(f"Zone {wl_ids[0]} wind [m/s]", labelpad=2)
    ax.set_ylabel(f"Zone {wl_ids[1]} wind [m/s]", labelpad=2)
    ax.set_zlabel("Price [EUR/MWh]", labelpad=2)
    ax.tick_params(axis="both", labelsize=7, pad=1)
    ax.zaxis.set_tick_params(labelsize=7, pad=1)
    ax.view_init(elev=22, azim=-48)
    ax.set_box_aspect((1.45, 1.05, 0.70))


def plot_price_weather_relationship(month=1):
    month_idx = month - 1
    daily_weather = _daily_weather()
    price = pd.read_parquet("data/price/price.parquet")
    price_models = _fit_price_models(price, daily_weather)

    fig = plt.figure(figsize=(FIGWIDTH / 2.54, 8.3 / 2.54), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.45],
        height_ratios=[1.0, 1.0],
        left=0.13,
        right=0.97,
        bottom=0.16,
        top=0.82,
        wspace=0.22,
        hspace=0.52,
    )
    ax_nld = fig.add_subplot(gs[0, 0])
    ax_gbr = fig.add_subplot(gs[1, 0])
    ax_deu = fig.add_subplot(gs[:, 1], projection="3d")

    deu = _price_weather_data("DEU", COUNTRIES["DEU"], daily_weather, price)
    nld = _price_weather_data("NLD", COUNTRIES["NLD"], daily_weather, price)
    gbr = _price_weather_data("GBR", COUNTRIES["GBR"], daily_weather, price)
    deu = deu[deu["month"] == month_idx]
    nld = nld[nld["month"] == month_idx]
    gbr = gbr[gbr["month"] == month_idx]

    _plot_2d(ax_nld, nld, "NLD", COUNTRIES["NLD"][0], price_models, month_idx)
    _plot_2d(ax_gbr, gbr, "GBR", COUNTRIES["GBR"][0], price_models, month_idx)
    _plot_3d(ax_deu, deu, "DEU", COUNTRIES["DEU"], price_models, month_idx)
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=colors.blue, alpha=0.45, markersize=4, label="Historical observations"),
        Line2D([0], [0], color=colors.red, linewidth=1.4, label="Fitted monthly regression"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        frameon=False,
        ncol=2,
        fontsize=7,
    )

    fig.suptitle(f"Electricity price and wind speed relationship, {MONTH_NAMES[month]}", y=0.99, fontsize=9)
    output_dir = Path(PLOT_DIR) / "weather_validation_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "price_weather_relationship.svg", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote {output_dir / 'price_weather_relationship.svg'}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot historical wind-speed and electricity-price relationships."
    )
    parser.add_argument(
        "--month",
        type=int,
        choices=range(1, 13),
        default=1,
        help="Representative month to plot, as an integer from 1 to 12.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    plot_price_weather_relationship(month=args.month)
