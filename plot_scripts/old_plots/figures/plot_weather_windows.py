
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data.fixed_data import data

Z_95 = 1.96

def _add_month(df: pd.DataFrame) -> pd.DataFrame:
    days_per_period = data.days_per_period
    periods = data.periods

    df = df.copy()
    period_idx = ((df["d"] - 1) // days_per_period).astype(int)
    df["month"] = period_idx.map(dict(enumerate(periods)))
    return df


def plot_weather_windows(
    dataset_path: str | Path = "data/scenario_data/weather_windows",
    output_plot_path: str | Path = "figures/weather_window_monthly_category_means.png",
    output_sorted_plot_path: str | Path = "figures/weather_window_monthly_sorted_by_long_windows.png",
    output_csv_path: str | Path = "results/weather_window_monthly_category_means.csv",
    show: bool = True,
):
    df = pd.read_parquet(dataset_path, columns=["s", "d", "ww"])
    df = df.dropna(subset=["s", "d", "ww"]).copy()
    df["s"] = df["s"].astype(int)
    df["d"] = df["d"].astype(int)
    df["ww"] = df["ww"].astype(float)
    df = _add_month(df)

    if df.empty:
        raise ValueError("No weather-window rows found in scenario data.")

    df["cat_0_4"] = ((df["ww"] >= 0) & (df["ww"] < 4)).astype(int)
    df["cat_4_8"] = ((df["ww"] >= 4) & (df["ww"] <= 8)).astype(int)
    df["cat_gt_8"] = (df["ww"] > 8).astype(int)

    by_scenario_month = (
        df.groupby(["s", "month"], as_index=False)[["cat_0_4", "cat_4_8", "cat_gt_8"]]
        .sum()
    )

    month_means = (
        by_scenario_month.groupby("month", as_index=False)[["cat_0_4", "cat_4_8", "cat_gt_8"]]
        .mean()
    )
    month_order = {m: i for i, m in enumerate(data.periods)}
    month_means = month_means.sort_values("month", key=lambda s: s.map(month_order)).reset_index(drop=True)

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    month_means.to_csv(output_csv_path, index=False)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar(month_means["month"], month_means["cat_0_4"], label="0-4h")
    ax.bar(
        month_means["month"],
        month_means["cat_4_8"],
        bottom=month_means["cat_0_4"],
        label="4-8h",
    )
    ax.bar(
        month_means["month"],
        month_means["cat_gt_8"],
        bottom=month_means["cat_0_4"] + month_means["cat_4_8"],
        label=">8h",
    )
    ax.set_title("Average monthly number of weather windows by length category")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average count across scenarios")
    ax.legend()

    output_plot_path = Path(output_plot_path)
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot_path, dpi=200)

    sorted_month_means = month_means.sort_values("cat_gt_8", ascending=False).reset_index(drop=True)
    fig_sorted, ax_sorted = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax_sorted.bar(sorted_month_means["month"], sorted_month_means["cat_0_4"], label="0-4h")
    ax_sorted.bar(
        sorted_month_means["month"],
        sorted_month_means["cat_4_8"],
        bottom=sorted_month_means["cat_0_4"],
        label="4-8h",
    )
    ax_sorted.bar(
        sorted_month_means["month"],
        sorted_month_means["cat_gt_8"],
        bottom=sorted_month_means["cat_0_4"] + sorted_month_means["cat_4_8"],
        label=">8h",
    )
    ax_sorted.set_title("Average monthly weather windows sorted by long windows (>8h)")
    ax_sorted.set_xlabel("Month (sorted)")
    ax_sorted.set_ylabel("Average count across scenarios")
    ax_sorted.legend()

    output_sorted_plot_path = Path(output_sorted_plot_path)
    output_sorted_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig_sorted.savefig(output_sorted_plot_path, dpi=200)

    print(f"Saved figure to: {output_plot_path}")
    print(f"Saved sorted figure to: {output_sorted_plot_path}")
    print(f"Saved table to: {output_csv_path}")
    print("\nMonthly means across scenarios:")
    print(month_means.round(2).to_string(index=False))

    if show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig_sorted)


def plot_daily_weather_window_mean_with_ci(
    dataset_path: str | Path = "data/scenario_data/weather_windows",
    output_plot_path: str | Path = "figures/weather_window_daily_mean_ci.png",
    output_csv_path: str | Path = "results/weather_window_daily_mean_ci.csv",
    show: bool = True,
):
    df = pd.read_parquet(dataset_path, columns=["s", "d", "ww"])
    df = df.dropna(subset=["s", "d", "ww"]).copy()
    df["s"] = df["s"].astype(int)
    df["d"] = df["d"].astype(int)
    df["ww"] = df["ww"].astype(float)

    if df.empty:
        raise ValueError("No weather-window rows found in scenario data.")

    # First collapse by (scenario, day), then estimate uncertainty across scenarios.
    by_scenario_day = df.groupby(["s", "d"], as_index=False)["ww"].mean()

    daily_stats = (
        by_scenario_day.groupby("d")["ww"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
        .sort_values("d")
        .reset_index(drop=True)
    )
    daily_stats["se"] = daily_stats["std"] / daily_stats["n"].pow(0.5)
    daily_stats["ci_half_width_95"] = Z_95 * daily_stats["se"]
    daily_stats["ci_low_95"] = daily_stats["mean"] - daily_stats["ci_half_width_95"]
    daily_stats["ci_high_95"] = daily_stats["mean"] + daily_stats["ci_half_width_95"]

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    daily_stats.to_csv(output_csv_path, index=False)

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.plot(daily_stats["d"], daily_stats["mean"], color="tab:blue", linewidth=1.8, label="Mean")
    ax.fill_between(
        daily_stats["d"],
        daily_stats["ci_low_95"],
        daily_stats["ci_high_95"],
        color="tab:blue",
        alpha=0.2,
        label="95% CI",
    )
    ax.set_title("Daily average weather-window length with 95% confidence interval")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Weather-window length [hours]")
    ax.set_xlim(daily_stats["d"].min(), daily_stats["d"].max())
    ax.grid(True, alpha=0.2)
    ax.legend()

    output_plot_path = Path(output_plot_path)
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot_path, dpi=200)

    print(f"Saved figure to: {output_plot_path}")
    print(f"Saved table to: {output_csv_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    plot_weather_windows(show=False)
    plot_daily_weather_window_mean_with_ci(show=False)
