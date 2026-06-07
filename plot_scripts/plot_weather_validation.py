import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from data.fixed_data import data
from plot_scripts.config import FIGWIDTH, PLOT_DIR, colors


OUTPUT_DIR = Path(PLOT_DIR) / "weather_validation_plots"
SCENARIO_DATA_DIR = Path(os.environ.get("SCENARIO_DATA_DIR", "data/scenario_data"))

LOCATION_IDS = [2, 3, 4, 5]
LOCATION_LABELS = {2: "1", 3: "2", 4: "3", 5: "4"}
TARGET_LOCATION = 2
OLD_ACF_LOCATION_IDS = [3, 4]
TARGET_VESSEL = "CTV"
MAX_SCENARIOS_RAW = 80
MAX_SCENARIOS_WINDOWS = 1500
KIND_MONTHS = {1, 2, 3, 10, 11, 12}


def _scenario_ids(dataset_name, limit):
    path = SCENARIO_DATA_DIR / dataset_name
    ids = []
    for child in path.iterdir():
        if child.is_dir() and child.name.startswith("s="):
            ids.append(int(child.name.split("=", 1)[1]))
    return sorted(ids)[:limit]


def _day_to_month(day):
    return int((pd.Timestamp("2011-01-01") + pd.Timedelta(days=int(day) - 1)).month)


def _month_labels():
    return [pd.Timestamp(2011, m, 1).strftime("%b") for m in range(1, 13)]


def _longest_consecutive_true(values):
    best = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _cv(series):
    mean = series.mean()
    if pd.isna(mean) or mean == 0:
        return np.nan
    return series.std() / mean


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


def _mean_group_acf(df, group_cols, value_col, max_lag):
    acfs = []
    for _, group in df.groupby(group_cols):
        acfs.append(_acf(group[value_col].to_numpy(), max_lag))
    return np.nanmean(np.vstack(acfs), axis=0)


def _load_real_weather():
    df = pd.read_parquet("data/weather/weather.parquet").reset_index()
    df = df[df["weather_location_id"].isin(LOCATION_IDS)].copy()
    df = df.rename(columns={"weather_location_id": "wl_id"})
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day_of_year"] = df["time"].dt.dayofyear
    df["hour"] = df["time"].dt.hour
    return df


def _load_scenario_weather(limit=MAX_SCENARIOS_RAW):
    scenario_ids = _scenario_ids("weather", limit)
    frames = []
    for scenario_id in scenario_ids:
        df = pd.read_parquet(
            SCENARIO_DATA_DIR / "weather",
            filters=[("s", "==", scenario_id)],
            columns=["s", "wl_id", "d", "hour", "speed", "height"],
        )
        frames.append(df[df["wl_id"].isin(LOCATION_IDS)])
    df = pd.concat(frames, ignore_index=True)
    df["s"] = df["s"].astype(int)
    df["wl_id"] = df["wl_id"].astype(int)
    df["d"] = df["d"].astype(int)
    df["hour"] = df["hour"].astype(int)
    df["month"] = df["d"].map(_day_to_month)
    return df


def _load_scenario_windows(limit=MAX_SCENARIOS_WINDOWS):
    scenario_ids = _scenario_ids("weather_windows", limit)
    all_ids = _scenario_ids("weather_windows", None)
    if len(scenario_ids) == len(all_ids):
        df = pd.read_parquet(
            SCENARIO_DATA_DIR / "weather_windows",
            columns=["h", "wl_id", "s", "d", "ww"],
        )
    else:
        frames = []
        for scenario_id in scenario_ids:
            df = pd.read_parquet(
                SCENARIO_DATA_DIR / "weather_windows",
                filters=[("s", "==", scenario_id)],
                columns=["h", "wl_id", "s", "d", "ww"],
            )
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
    df = df[
        (df["h"] == TARGET_VESSEL)
        & (df["wl_id"].astype(int).isin(LOCATION_IDS))
    ].copy()
    df["s"] = df["s"].astype(int)
    df["wl_id"] = df["wl_id"].astype(int)
    df["d"] = df["d"].astype(int)
    df["ww"] = df["ww"].astype(float)
    df["month"] = df["d"].map(_day_to_month)
    return df


def _real_weather_windows(real_weather):
    max_wave = next(v.max_wave for v in data.vessel_types if v.name == TARGET_VESSEL)
    working_hours = list(range(data.work_day_start, data.work_day_end))
    df = real_weather[real_weather["hour"].isin(working_hours)].copy()
    rows = (
        df.groupby(["wl_id", "year", "day_of_year", "month"])["height"]
        .apply(lambda x: _longest_consecutive_true((x <= max_wave).to_numpy()))
        .reset_index(name="ww")
    )
    rows["d"] = rows["day_of_year"]
    return rows


def _plot_example_year(real_weather, scenario_weather):
    real = real_weather[
        (real_weather["year"] == 2024) & (real_weather["wl_id"] == TARGET_LOCATION)
    ].copy()
    scenario_id = int(scenario_weather["s"].min())
    sim = scenario_weather[
        (scenario_weather["s"] == scenario_id) & (scenario_weather["wl_id"] == TARGET_LOCATION)
    ].copy()
    sim["t"] = (sim["d"] - 1) * 24 + sim["hour"]

    with plt.rc_context({"figure.constrained_layout.use": False}):
        fig, axs = plt.subplots(
            2,
            2,
            figsize=(FIGWIDTH / 2.54, 7.3 / 2.54),
            sharex="col",
            gridspec_kw={"hspace": 0.12, "wspace": 0.18},
        )
    for ax, var, ylabel in [
        (axs[0, 0], "speed", "Wind speed\n[m/s]"),
        (axs[1, 0], "height", "Wave height\n[m]"),
    ]:
        ax.plot(real["time"], real[var], color=colors.blue, linewidth=0.45)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="0.90", linewidth=0.5)

    for ax, var, ylabel in [
        (axs[0, 1], "speed", "Wind speed\n[m/s]"),
        (axs[1, 1], "height", "Wave height\n[m]"),
    ]:
        ax.plot(sim["t"], sim[var], color=colors.red, linewidth=0.45)
        ax.grid(axis="y", color="0.90", linewidth=0.5)

    axs[0, 0].set_title(f"Historical 2024, location {LOCATION_LABELS[TARGET_LOCATION]}")
    axs[0, 1].set_title(f"Synthetic scenario {scenario_id}, location {LOCATION_LABELS[TARGET_LOCATION]}")
    for ax in axs[:, 0]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.set_xlim(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31 23:00:00"))
    axs[1, 1].set_xticks([0, 2160, 4344, 6552])
    axs[1, 1].set_xticklabels(["Jan", "Apr", "Jul", "Oct"])
    axs[1, 0].set_xlabel("Date")
    axs[1, 1].set_xlabel("Planning horizon")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.88, wspace=0.20, hspace=0.18)
    fig.savefig(OUTPUT_DIR / "weather_example_year.svg")
    plt.close(fig)


def _plot_empirical_distributions(real_weather, scenario_weather):
    fig, axs = plt.subplots(1, 2, figsize=(FIGWIDTH / 2.54, 5 / 2.54), constrained_layout=True)
    for ax, var, xlabel in [
        (axs[0], "speed", "Wind speed [m/s]"),
        (axs[1], "height", "Wave height [m]"),
    ]:
        real_values = real_weather[var].dropna().to_numpy()
        sim_values = scenario_weather[var].dropna().to_numpy()
        bins = np.linspace(min(real_values.min(), sim_values.min()), max(real_values.max(), sim_values.max()), 70)
        ax.hist(
            real_values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            color=colors.blue,
            label="Historical",
        )
        ax.hist(
            sim_values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            color=colors.red,
            label="Synthetic",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.grid(axis="y", color="0.90", linewidth=0.5)
    axs[0].legend(frameon=False)
    fig.savefig(OUTPUT_DIR / "weather_empirical_distributions.svg")
    plt.close(fig)


def _plot_acf_with_variation(real_weather, scenario_weather):
    max_lag = 168
    real = real_weather[real_weather["wl_id"] == TARGET_LOCATION].sort_values("time")
    sim = scenario_weather[scenario_weather["wl_id"] == TARGET_LOCATION].sort_values(["s", "d", "hour"])
    lags = np.arange(max_lag + 1)

    fig, axs = plt.subplots(1, 2, figsize=(FIGWIDTH / 2.54, 5 / 2.54), constrained_layout=True)
    for ax, var, title in [
        (axs[0], "speed", "Wind speed"),
        (axs[1], "height", "Wave height"),
    ]:
        for _, group in real.groupby("year"):
            ax.plot(
                lags,
                _acf(group[var].to_numpy(), max_lag),
                color=colors.blue,
                alpha=0.18,
                linewidth=0.45,
            )
        for _, group in list(sim.groupby("s"))[:30]:
            ax.plot(
                lags,
                _acf(group[var].to_numpy(), max_lag),
                color=colors.red,
                alpha=0.13,
                linewidth=0.45,
            )
        ax.plot(
            lags,
            _mean_group_acf(real, ["year"], var, max_lag),
            color=colors.blue,
            linewidth=1.35,
            label="Historical mean",
        )
        ax.plot(
            lags,
            _mean_group_acf(sim, ["s"], var, max_lag),
            color=colors.red,
            linewidth=1.35,
            label="Synthetic mean",
        )
        ax.set_title(title)
        ax.set_xlabel("Lag [hours]")
        ax.set_ylabel("ACF")
        ax.set_ylim(-0.1, 1.05)
        ax.grid(color="0.90", linewidth=0.5)
    axs[0].legend(frameon=False)
    fig.savefig(OUTPUT_DIR / "weather_acf_with_variation.svg")
    plt.close(fig)


def _plot_acf_old_style(real_weather, scenario_weather):
    max_lag = 168
    real = real_weather[real_weather["wl_id"].isin(OLD_ACF_LOCATION_IDS)].sort_values("time")
    sim = scenario_weather[scenario_weather["wl_id"].isin(OLD_ACF_LOCATION_IDS)].sort_values(["s", "d", "hour"])
    lags = np.arange(max_lag + 1)

    fig, axs = plt.subplots(1, 2, figsize=(FIGWIDTH / 2.54, 5 / 2.54), constrained_layout=True)
    for ax, var, title in [
        (axs[0], "speed", "Wind speed"),
        (axs[1], "height", "Wave height"),
    ]:
        ax.plot(
            lags,
            _acf(sim[var].to_numpy(), max_lag),
            color=colors.red,
            linewidth=0.75,
            alpha=0.85,
            label="Synthetic",
        )
        ax.plot(
            lags,
            _acf(real[var].to_numpy(), max_lag),
            color=colors.blue,
            linewidth=0.75,
            alpha=0.95,
            label="Historical",
        )
        ax.set_title(title)
        ax.set_xlabel("Lag [hours]")
        ax.set_ylabel("ACF")
        ax.set_ylim(-0.1, 1.05)
        ax.grid(color="0.90", linewidth=0.5)
    axs[0].legend(frameon=False)
    fig.savefig(OUTPUT_DIR / "weather_acf_old_style.svg")
    plt.close(fig)


def _plot_correlation_matrices(real_weather, scenario_weather):
    fig, axs = plt.subplots(2, 2, figsize=(FIGWIDTH / 2.54, 7 / 2.54), constrained_layout=True)
    matrices = []
    for source_df, source, index_cols in [
        (real_weather, "Historical", ["time"]),
        (scenario_weather, "Synthetic", ["s", "d", "hour"]),
    ]:
        for var, title_var in [("speed", "Wind speed"), ("height", "Wave height")]:
            table = source_df.pivot_table(index=index_cols, columns="wl_id", values=var)
            corr = table[LOCATION_IDS].corr()
            matrices.append((source, title_var, corr))

    for ax, (source, title_var, corr) in zip(axs.flat, matrices):
        im = ax.imshow(corr, vmin=0, vmax=1)
        ax.set_title(f"{source}: {title_var}")
        ax.set_xticks(range(len(LOCATION_IDS)))
        ax.set_yticks(range(len(LOCATION_IDS)))
        ax.set_xticklabels([LOCATION_LABELS[i] for i in LOCATION_IDS])
        ax.set_yticklabels([LOCATION_LABELS[i] for i in LOCATION_IDS])
    fig.colorbar(im, ax=axs, shrink=0.8, label="Correlation")
    fig.savefig(OUTPUT_DIR / "weather_correlation_matrices.svg")
    plt.close(fig)


def _plot_window_duration_distribution(real_windows, scenario_windows):
    fig, ax = plt.subplots(figsize=(FIGWIDTH / 2.54, 5 / 2.54))
    fig.set_layout_engine(None)
    bins = np.arange(0, max(real_windows["ww"].max(), scenario_windows["ww"].max()) + 2)
    ax.hist(
        real_windows["ww"],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.5,
        color=colors.blue,
        label="Historical",
    )
    ax.hist(
        scenario_windows["ww"],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.5,
        color=colors.red,
        label="Synthetic",
    )
    ax.set_xlabel("Weather window duration [hours]")
    ax.set_ylabel("Density")
    ax.grid(axis="y", color="0.90", linewidth=0.5)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT_DIR / "weather_window_duration_distribution.svg")
    plt.close(fig)


def _plot_window_cv(real_windows, scenario_windows):
    real_sum = real_windows.groupby(["year", "d"], as_index=False)["ww"].sum()
    sim_sum = scenario_windows.groupby(["s", "d"], as_index=False)["ww"].sum()
    real_cv = real_sum.groupby("d")["ww"].agg(_cv)
    sim_cv = sim_sum.groupby("d")["ww"].agg(_cv)
    days = sorted(set(real_cv.index).intersection(sim_cv.index))

    fig, ax = plt.subplots(figsize=(FIGWIDTH / 2.54, 5 / 2.54), constrained_layout=True)
    ax.plot(days, real_cv.reindex(days), color=colors.blue, label="Historical")
    ax.plot(days, sim_cv.reindex(days), color=colors.red, label="Synthetic")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("CV of summed weather windows")
    ax.grid(color="0.90", linewidth=0.5)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT_DIR / "weather_window_cv.svg")
    plt.close(fig)


def _metric_scores(windows, entity_col, high_threshold=8, low_threshold=4):
    winter = windows[windows["month"].isin(KIND_MONTHS)].copy()
    all_entities = pd.Index(sorted(winter[entity_col].unique()), name=entity_col)

    over = (
        winter[winter["ww"] >= high_threshold]
        .groupby([entity_col, "wl_id"])
        .size()
        .groupby(entity_col)
        .sum()
        .reindex(all_entities, fill_value=0)
    )
    total = winter.groupby(entity_col)["ww"].sum().reindex(all_entities, fill_value=0)
    under = (
        winter[winter["ww"] <= low_threshold]
        .groupby([entity_col, "wl_id"])
        .size()
        .groupby(entity_col)
        .sum()
        .reindex(all_entities, fill_value=0)
    )

    streak = {}
    day_col = "d"
    for entity, group in winter.groupby(entity_col):
        value = 0
        for _, loc_group in group.groupby("wl_id"):
            loc_group = loc_group.sort_values(day_col)
            value += _longest_consecutive_true((loc_group["ww"] <= low_threshold).to_numpy())
        streak[entity] = value
    streak = pd.Series(streak).reindex(all_entities, fill_value=0)

    return {
        "count_ge_8": over,
        "total_hours": total,
        "neg_count_le_4": -under,
        "neg_max_bad_streak": -streak,
    }


def _plot_metric_distributions(real_windows, scenario_windows):
    sim_scores = _metric_scores(scenario_windows, "s")
    real_scores = _metric_scores(real_windows, "year")
    configs = [
        ("count_ge_8", "Location-days with\nWW >= 8"),
        ("total_hours", "Total winter\nweather-window hours"),
        ("neg_count_le_4", "Negative location-days\nwith WW <= 4"),
        ("neg_max_bad_streak", "Negative maximum\nbad-weather streak"),
    ]
    with plt.rc_context({"figure.constrained_layout.use": False}):
        fig, axs = plt.subplots(2, 2, figsize=(FIGWIDTH / 2.54, 9.4 / 2.54))
    for ax, (key, title) in zip(axs.flat, configs):
        scores = sim_scores[key].sort_values(ascending=False)
        ranks = np.arange(1, len(scores) + 1)
        ax.plot(ranks, scores.values, color=colors.blue, linewidth=1.2, label="Synthetic scenarios")

        scenario_values = scores.values
        real_values = real_scores[key].values
        real_ranks = np.searchsorted(-scenario_values, -real_values, side="right") + 1
        real_ranks = np.clip(real_ranks, 1, len(scores))
        ax.scatter(real_ranks, real_values, color=colors.red, marker="x", s=20, label="Historical years")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(color="0.90", linewidth=0.5)
    fig.text(0.045, 0.71, "Score", rotation=90, ha="center", va="center")
    fig.text(0.045, 0.29, "Score", rotation=90, ha="center", va="center")
    fig.add_artist(Line2D([0.885, 0.885], [0.56, 0.66], transform=fig.transFigure, color=colors.blue, linewidth=1.2))
    fig.text(0.925, 0.61, "Synthetic\nscenarios", rotation=270, ha="center", va="center", linespacing=0.9)
    fig.add_artist(Line2D([0.885], [0.34], transform=fig.transFigure, color=colors.red, marker="x", linestyle="None", markersize=5))
    fig.text(0.925, 0.34, "Historical\nyears", rotation=270, ha="center", va="center", linespacing=0.9)
    fig.subplots_adjust(left=0.13, right=0.86, bottom=0.13, top=0.90, wspace=0.24, hspace=0.62)
    fig.canvas.draw()
    for ax in axs[1, :]:
        bbox = ax.get_position()
        fig.text((bbox.x0 + bbox.x1) / 2, 0.045, "Rank", ha="center", va="center")
    fig.savefig(OUTPUT_DIR / "weather_metric_rank_distributions.svg")
    plt.close(fig)


def _plot_monthly_longest_storm(real_windows, scenario_windows, threshold=4):
    def monthly(windows, entity_col):
        rows = []
        for (entity, month), group in windows.groupby([entity_col, "month"]):
            daily = (
                group.groupby("d")
                .agg(n_locs=("wl_id", "nunique"), n_bad=("ww", lambda x: int((x <= threshold).sum())))
                .reset_index()
            )
            daily = daily[daily["n_locs"] == len(LOCATION_IDS)].copy()
            daily["all_bad"] = daily["n_bad"] == len(LOCATION_IDS)
            rows.append({
                entity_col: entity,
                "month": month,
                "longest_storm": _longest_consecutive_true(daily.sort_values("d")["all_bad"].to_numpy()),
            })
        return pd.DataFrame(rows).groupby("month")["longest_storm"].mean().reindex(range(1, 13), fill_value=0)

    real = monthly(real_windows, "year")
    sim = monthly(scenario_windows, "s")
    x = np.arange(1, 13)

    fig, ax = plt.subplots(figsize=(FIGWIDTH / 2.54, 5 / 2.54), constrained_layout=True)
    ax.plot(x, real, marker="o", color=colors.blue, label="Historical")
    ax.plot(x, sim, marker="s", color=colors.red, label="Synthetic")
    ax.set_xticks(x)
    ax.set_xticklabels(_month_labels(), rotation=45)
    ax.set_ylabel("Longest all-location\nbad-weather streak\n[days]")
    ax.grid(color="0.90", linewidth=0.5)
    ax.legend(frameon=False)
    fig.subplots_adjust(left=0.17, bottom=0.22, right=0.98, top=0.96)
    fig.savefig(OUTPUT_DIR / "weather_monthly_longest_storm.svg")
    plt.close(fig)


def _plot_threshold_boxplots(real_windows, scenario_windows):
    def counts(windows, entity_col, op):
        if op == "good":
            selected = windows[windows["ww"] >= 8]
        else:
            selected = windows[windows["ww"] <= 4]
        return (
            selected.groupby([entity_col, "wl_id"]).size().groupby(entity_col).sum()
            .reindex(sorted(windows[entity_col].unique()), fill_value=0)
        )

    data_groups = [
        counts(real_windows, "year", "good"),
        counts(scenario_windows, "s", "good"),
        counts(real_windows, "year", "bad"),
        counts(scenario_windows, "s", "bad"),
    ]

    fig, axs = plt.subplots(1, 2, figsize=(FIGWIDTH / 2.54, 5 / 2.54), constrained_layout=True)
    for ax, groups, title in [
        (axs[0], data_groups[:2], "Location-days with WW >= 8"),
        (axs[1], data_groups[2:], "Location-days with WW <= 4"),
    ]:
        ax.boxplot(groups, labels=["Historical", "Synthetic"], showfliers=False)
        ax.scatter([1] * len(groups[0]), groups[0], color=colors.blue, s=10, alpha=0.6)
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.grid(axis="y", color="0.90", linewidth=0.5)
    fig.savefig(OUTPUT_DIR / "weather_window_threshold_boxplots.svg")
    plt.close(fig)


def plot_weather_validation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    real_weather = _load_real_weather()
    scenario_weather = _load_scenario_weather()
    scenario_windows = _load_scenario_windows()
    real_windows = _real_weather_windows(real_weather)

    _plot_example_year(real_weather, scenario_weather)
    _plot_empirical_distributions(real_weather, scenario_weather)
    _plot_acf_with_variation(real_weather, scenario_weather)
    _plot_acf_old_style(real_weather, scenario_weather)
    _plot_correlation_matrices(real_weather, scenario_weather)
    _plot_window_duration_distribution(real_windows, scenario_windows)
    _plot_window_cv(real_windows, scenario_windows)
    _plot_threshold_boxplots(real_windows, scenario_windows)
    _plot_monthly_longest_storm(real_windows, scenario_windows)
    _plot_metric_distributions(real_windows, scenario_windows)


if __name__ == "__main__":
    plot_weather_validation()
