from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.fixed_data import data
from plot_scripts.config import FIGWIDTH, PLOT_DIR, colors


OUTPUT_DIR = Path(PLOT_DIR) / "weather_validation_plots"
TARGET_WIND_FARM = "E"
HISTORICAL_YEAR = 2024
SCENARIO_ID = 1
MAX_SCENARIOS = 120
ACF_MAX_LAG = 90


def _target_wind_farm():
    return next(wind_farm for wind_farm in data.wind_farms if wind_farm.name == TARGET_WIND_FARM)


def _target_iso():
    return _target_wind_farm().iso


def _scenario_ids(dataset_name, limit):
    path = Path("data/scenario_data") / dataset_name
    ids = []
    for child in path.iterdir():
        if child.is_dir() and child.name.startswith("s="):
            ids.append(int(child.name.split("=", 1)[1]))
    return sorted(ids)[:limit]


def _acf(values, max_lag):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= max_lag + 1:
        return np.full(max_lag + 1, np.nan)
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 0:
        out = np.zeros(max_lag + 1)
        out[0] = 1
        return out
    n = len(x)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=nfft)
    raw = np.fft.irfft(f * np.conjugate(f), n=nfft)[:n]
    return raw[: max_lag + 1] / denom


def _load_historical_price(iso=None, year=None):
    iso = iso or _target_iso()
    df = pd.read_parquet("data/price/price.parquet", filters=[("ISO3", "==", iso)])
    if year is not None:
        df = df[df.index.year == year]
    df = df.sort_index().reset_index()
    df["d"] = df["date"].dt.dayofyear
    return df[df["d"] <= 360]


def _load_synthetic_price(iso=None, scenario_id=None):
    iso = iso or _target_iso()
    scenario_id = SCENARIO_ID if scenario_id is None else scenario_id
    df = pd.read_parquet(Path("data/scenario_data/price") / f"s={scenario_id}" / "part-0")
    df = df[df["iso"] == iso].copy()
    df["s"] = scenario_id
    df = df.sort_values("d").copy()
    df["date"] = pd.Timestamp(HISTORICAL_YEAR, 1, 1) + pd.to_timedelta(df["d"] - 1, unit="D")
    return df


def _load_synthetic_price_sample(iso=None, limit=MAX_SCENARIOS):
    iso = iso or _target_iso()
    frames = []
    for scenario_id in _scenario_ids("price", limit):
        df = pd.read_parquet(Path("data/scenario_data/price") / f"s={scenario_id}" / "part-0")
        df = df[df["iso"] == iso].copy()
        df["s"] = scenario_id
        frames.append(df[["s", "iso", "d", "price"]])
    return pd.concat(frames, ignore_index=True)


def _load_historical_downtime_cost():
    wind_farm = _target_wind_farm()
    price = _load_historical_price(wind_farm.iso, HISTORICAL_YEAR).set_index("date")

    weather = pd.read_parquet(
        "data/weather/weather.parquet",
        filters=[("weather_location_id", "==", wind_farm.weather_location_id)],
    )
    weather = weather[weather.index.year == HISTORICAL_YEAR].copy()
    weather["power"] = data.power_curve(weather["speed"])
    power = weather[["power"]].resample("D").mean()

    df = price.join(power, how="inner")
    df["downtime_cost"] = df["power"] * 24 * df["price"]
    df = df.reset_index().rename(columns={"index": "date"})
    df["d"] = df["date"].dt.dayofyear
    return df[df["d"] <= 360]


def _load_synthetic_downtime_cost():
    df = pd.read_parquet(
        "data/scenario_data/downtime_cost",
        filters=[("s", "==", SCENARIO_ID), ("w", "==", TARGET_WIND_FARM)],
        columns=["s", "w", "d", "downtime_cost"],
    )
    df = df.sort_values("d").copy()
    df["date"] = pd.Timestamp(HISTORICAL_YEAR, 1, 1) + pd.to_timedelta(df["d"] - 1, unit="D")
    return df


def _plot_two_panel_time_series(real, sim, value_col, ylabel, left_title, right_title, filename):
    with plt.rc_context({"figure.constrained_layout.use": False}):
        fig, axs = plt.subplots(
            1,
            2,
            figsize=(FIGWIDTH / 2.54, 3.8 / 2.54),
            sharey=True,
            gridspec_kw={"wspace": 0.16},
        )

    axs[0].plot(real["date"], real[value_col], color=colors.blue, linewidth=0.65)
    axs[1].plot(sim["date"], sim[value_col], color=colors.red, linewidth=0.65)

    axs[0].set_title(left_title)
    axs[1].set_title(right_title)
    axs[0].set_ylabel(ylabel)

    for ax in axs:
        ax.grid(axis="y", color="0.90", linewidth=0.5)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.set_xlim(pd.Timestamp(HISTORICAL_YEAR, 1, 1), pd.Timestamp(HISTORICAL_YEAR, 12, 25))

    axs[0].set_xlabel("Date")
    axs[1].set_xlabel("Planning horizon")
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.25, top=0.82, wspace=0.18)
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Wrote {OUTPUT_DIR / filename}", flush=True)


def _plot_price_example():
    iso = _target_iso()
    real = _load_historical_price(iso, HISTORICAL_YEAR)
    sim = _load_synthetic_price(iso, SCENARIO_ID)
    _plot_two_panel_time_series(
        real,
        sim,
        "price",
        "Price\n[EUR/MWh]",
        f"Historical {HISTORICAL_YEAR}, {iso}",
        f"Synthetic scenario {SCENARIO_ID}, {iso}",
        "electricity_price_example_year.svg",
    )


def _plot_downtime_cost_example():
    real = _load_historical_downtime_cost()
    sim = _load_synthetic_downtime_cost()
    _plot_two_panel_time_series(
        real,
        sim,
        "downtime_cost",
        "Downtime cost\n[EUR/day]",
        f"Historical {HISTORICAL_YEAR}, wind farm {TARGET_WIND_FARM}",
        f"Synthetic scenario {SCENARIO_ID}, wind farm {TARGET_WIND_FARM}",
        "downtime_cost_example_year.svg",
    )


def _plot_price_distribution():
    iso = _target_iso()
    real = _load_historical_price(iso)["price"].dropna().to_numpy()
    sim = _load_synthetic_price_sample(iso)["price"].dropna().to_numpy()
    lower = min(np.nanpercentile(real, 0.5), np.nanpercentile(sim, 0.5))
    upper = max(np.nanpercentile(real, 99.5), np.nanpercentile(sim, 99.5))
    bins = np.linspace(lower, upper, 70)

    fig, ax = plt.subplots(figsize=(FIGWIDTH / 2.54, 4.4 / 2.54), constrained_layout=True)
    ax.hist(real, bins=bins, density=True, histtype="step", linewidth=1.5, color=colors.blue, label="Historical")
    ax.hist(sim, bins=bins, density=True, histtype="step", linewidth=1.5, color=colors.red, label="Synthetic")
    ax.set_xlabel("Price [EUR/MWh]")
    ax.set_ylabel("Density")
    ax.grid(axis="y", color="0.90", linewidth=0.5)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT_DIR / "electricity_price_distribution.svg")
    plt.close(fig)
    print(f"Wrote {OUTPUT_DIR / 'electricity_price_distribution.svg'}", flush=True)


def _plot_price_acf():
    iso = _target_iso()
    real = _load_historical_price(iso)
    sim = _load_synthetic_price_sample(iso)
    real_acf = np.nanmean(
        np.vstack([_acf(group["price"].to_numpy(), ACF_MAX_LAG) for _, group in real.groupby(real["date"].dt.year)]),
        axis=0,
    )
    sim_acf = np.nanmean(
        np.vstack([_acf(group["price"].to_numpy(), ACF_MAX_LAG) for _, group in sim.groupby("s")]),
        axis=0,
    )
    lags = np.arange(ACF_MAX_LAG + 1)

    fig, ax = plt.subplots(figsize=(FIGWIDTH / 2.54, 4.4 / 2.54), constrained_layout=True)
    ax.plot(lags, real_acf, color=colors.blue, linewidth=1.2, label="Historical")
    ax.plot(lags, sim_acf, color=colors.red, linewidth=1.2, label="Synthetic")
    ax.set_xlabel("Lag [days]")
    ax.set_ylabel("Autocorrelation")
    ax.set_ylim(bottom=-0.25, top=1.02)
    ax.grid(color="0.90", linewidth=0.5)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT_DIR / "electricity_price_acf.svg")
    plt.close(fig)
    print(f"Wrote {OUTPUT_DIR / 'electricity_price_acf.svg'}", flush=True)


def plot_electricity_validation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_price_example()
    _plot_downtime_cost_example()
    _plot_price_distribution()
    _plot_price_acf()


if __name__ == "__main__":
    plot_electricity_validation()
