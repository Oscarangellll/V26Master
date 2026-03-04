import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.fixed_data import data
from scenario_models import WeatherModel, PriceModel
from scenarios.gen_windows import _find_window 

rng = np.random.default_rng(seed=5)

FIG_WIDTH = 15/2.54

data_w_hash = data.weather_data_hash()
data_w_path = f"data/weather/{data_w_hash}.csv"
df_weather = pd.read_csv(
    data_w_path,
    index_col="time",
    parse_dates=True
)
df_weather = df_weather[df_weather["weather_location_id"] == 1]
y = df_weather[["speed", "height"]].to_numpy()
month_of_obs_w = df_weather.index.month.to_numpy() - 1

weather_model = WeatherModel(0.4, 0.1)
ys = weather_model.simulate(1, rng, month_of_obs=month_of_obs_w)

# Distributions
if False:
    fig, axs = plt.subplots(1, 2, figsize=(FIG_WIDTH, 4))

    axs[0].hist((y[:,0], ys[:,0]), bins=30, density=True, label=["Observed", "Simulated"])
    axs[0].set_xlabel("Wind speed [m/s]")
    axs[0].legend()

    axs[1].hist((y[:,1], ys[:,1]), bins=30, density=True, label=["Observed", "Simulated"])
    axs[1].set_xlabel("Wave heigth [m]")
    axs[1].legend()
    
    fig.savefig("distributions.png", format="png")

    plt.show()

# Daily weather_windows
if False:
    for m in range(12):
        idx = month_of_obs_w == m
        ym = y[idx]
        y_days = ym.reshape(-1, 24, 2)

        daily_windows = [
            _find_window(day[:,0], day[:,1], data.vessel_types[1])
            for day in y_days
        ]
        ysm = ys[idx]
        ys_days = ysm.reshape(-1, 24, 2)

        daily_windows_s = [
            _find_window(day[:,0], day[:,1], data.vessel_types[1])
            for day in ys_days
        ]
        plt.hist((daily_windows, daily_windows_s), bins=12)
        plt.show()

# Time series
if False:
    fig, axs = plt.subplots(2, 1, figsize=(FIG_WIDTH, 4))
    axs[0].plot(y[:,1])

    axs[1].plot(ys[:,1])
    plt.show()


# Weather window persistance
if False:
    # --- real WW ---
    cond = ((y[:,0] <= 30) & (y[:,1] <= 2)).astype(int)
    cond = np.pad(cond, 1)
    diffs = np.diff(cond)
    starts = (diffs == 1).nonzero()[0]
    ends = (diffs == -1).nonzero()[0]
    ww_real = ends - starts

    # --- simulated WW ---
    cond_sim = ((ys[:, 0] <= 30) & (ys[:, 1] <= 2)).astype(int)
    cond_sim = np.pad(cond_sim, 1)
    diffs_sim = np.diff(cond_sim)
    starts_sim = (diffs_sim == 1).nonzero()[0]
    ends_sim = (diffs_sim == -1).nonzero()[0]
    ww_sim = ends_sim - starts_sim

    # --- bins ---
    bin_size = 24 
    bins = np.arange(0, 400 + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Function to compute normalized histogram
    def compute_bin_durations(ww, bins, total_length):
        bin_durations = np.zeros(len(bins)-1)
        for i in range(len(bins)-1):
            mask = (ww >= bins[i]) & (ww < bins[i+1])
            bin_durations[i] = ww[mask].sum()
        return bin_durations / total_length

    normalized_real = compute_bin_durations(ww_real, bins, len(y))
    normalized_sim  = compute_bin_durations(ww_sim, bins, len(ys))

    # --- Plot both side by side ---
    fig, axs = plt.subplots(1, 2, figsize=(13,5))

    # Left: persistence histogram (time fraction)
    width = bin_size / 3
    axs[0].bar(bin_centers - width/2, normalized_real, width=width, label='obs', edgecolor='k')
    axs[0].bar(bin_centers + width/2, normalized_sim, width=width, label='sim', edgecolor='k')
    axs[0].set_xlabel("Weather window duration (hours)")
    axs[0].set_ylabel("Fraction of total time")
    axs[0].set_title("Persistence histogram")
    axs[0].legend()

    # Right: normal histogram of window durations
    axs[1].hist((ww_real, ww_sim), bins=bins, edgecolor='k', label=["obs", "sim"])
    axs[1].set_xlabel("Weather window duration (hours)")
    axs[1].set_ylabel("Number of windows")
    axs[1].set_title("Histogram of window durations")
    axs[1].legend()

    plt.show()

# Create summary csv of speed and height
if False:
    monthly_mean_obs = np.zeros((12, 2))
    monthly_std_obs  = np.zeros((12, 2))

    monthly_mean_syn = np.zeros((12, 2))
    monthly_std_syn  = np.zeros((12, 2))
    
    for m in range(12):
        idx = month_of_obs == m

        monthly_mean_obs[m] = y[idx].mean(axis=0)
        monthly_std_obs[m]  = y[idx].std(axis=0)

        monthly_mean_syn[m] = ys[idx].mean(axis=0)
        monthly_std_syn[m]  = ys[idx].std(axis=0)

    speed_summary = pd.DataFrame({
        ("Mean", "Observed"): monthly_mean_obs[:,0], 
        ("Mean", "Synthetic"): monthly_mean_syn[:,0], 
        ("Std", "Observed"): monthly_std_obs[:,0], 
        ("Std", "Synthetic"): monthly_std_syn[:,0]}, 
        index=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    )
    speed_summary.to_csv("wind_speed_summary.csv", float_format="%.2f")
    
    height_summary = pd.DataFrame({
        ("Mean", "Observed"): monthly_mean_obs[:,1], 
        ("Mean", "Synthetic"): monthly_mean_syn[:,1], 
        ("Std", "Observed"): monthly_std_obs[:,1], 
        ("Std", "Synthetic"): monthly_std_syn[:,1]}, 
        index=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    )
    height_summary.to_csv("wave_height_summary.csv", float_format="%.2f")

#### Prices
price_model = PriceModel()
if True:
    price_filepath = f"data/electricity/{data.price_data_hash}.csv"

    df_price = pd.read_csv(
        price_filepath,
        index_col="date",
        parse_dates=True
    )
    df_price = df_price[df_price["ISO3"] == "DEU"]
    p = df_price["price"].to_numpy()
    month_of_obs_p = df_price.index.month.to_numpy() - 1
    
    wl_ids = sorted({
        w.weather_location_id for w in data.wind_farms if w.iso == "DEU"
    })

    speeds = np.column_stack([weather_model.simulate(wl_id, rng, month_of_obs=np.repeat(month_of_obs_p, 24))[:,0].reshape(-1,24).mean(axis=1) for wl_id in wl_ids])
    
    ps = price_model.simulate(speeds, "DEU", rng, month_of_obs=month_of_obs_p)
    
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4))

    ax.hist((p, ps), bins=40, density=True, label=["Observed", "Simulated"])
    ax.set_xlabel("Electricity price [EUR/MWh]")
    ax.legend()
    
    fig.savefig("price_distribution.png", format="png")

    plt.show()

# --- Average year weather window time series (observed vs synthetic), 07–19 only ---
if False:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    # 1. Filter to hours between 07:00 and 19:00
    hour_mask = (df_weather.index.hour >= 7) & (df_weather.index.hour < 19)

    y_day = y[hour_mask]
    ys_day = ys[hour_mask]
    doy = df_weather.index[hour_mask].dayofyear.to_numpy()
    years = df_weather.index[hour_mask].year.to_numpy()

    unique_years = np.unique(years)
    n_years = len(unique_years)

    # 2. Weather window condition
    ww_obs_hourly = ((y_day[:,0] <= 30) & (y_day[:,1] <= 2)).astype(int)
    ww_syn_hourly = ((ys_day[:,0] <= 30) & (ys_day[:,1] <= 2)).astype(int)

    # 3. Prepare arrays
    max_doy = doy.max()
    daily_obs = np.zeros(max_doy)
    daily_syn = np.zeros(max_doy)

    # 4. Aggregate per day-of-year, then average across years
    for d in range(1, max_doy+1):
        mask = (doy == d)
        if mask.sum() > 0:
            total_obs = ww_obs_hourly[mask].sum()
            total_syn = ww_syn_hourly[mask].sum()

            # Convert to hours per day (divide by number of years)
            daily_obs[d-1] = total_obs / n_years
            daily_syn[d-1] = total_syn / n_years
        else:
            daily_obs[d-1] = np.nan
            daily_syn[d-1] = np.nan

    # 5. LOWESS smoothing
    x = np.arange(max_doy)
    frac = 0.1  # smoothing parameter

    trend_obs = lowess(daily_obs, x, frac=frac, return_sorted=False)
    trend_syn = lowess(daily_syn, x, frac=frac, return_sorted=False)

    # 6. Plot: two subplots
    fig, axs = plt.subplots(2, 1, figsize=(FIG_WIDTH, 8), sharex=True)

    # Observed
    axs[0].plot(x, daily_obs, label="Observed (avg year)", lw=1.0)
    axs[0].plot(x, trend_obs, label="Trend (LOWESS)", lw=2, color="red")
    axs[0].set_ylabel("Weather window hours/day (07–19)")
    axs[0].set_title("Observed average year (07–19 only)")
    axs[0].set_ylim(0, 13)
    axs[0].legend()

    # Synthetic
    axs[1].plot(x, daily_syn, label="Synthetic (avg year)", lw=1.0)
    axs[1].plot(x, trend_syn, label="Trend (LOWESS)", lw=2, color="red")
    axs[1].set_ylabel("Weather window hours/day (07–19)")
    axs[1].set_title("Synthetic average year (07–19 only)")
    axs[1].set_ylim(0, 13)
    axs[1].legend()

    axs[1].set_xlabel("Day of year")
    plt.tight_layout()
    plt.show()

# --- Average year weather window time series (observed vs synthetic), 07–19 only ---
if False:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    # 1. Filter to hours between 07:00 and 19:00
    hour_mask = (df_weather.index.hour >= 7) & (df_weather.index.hour < 19)

    y_day = y[hour_mask]
    ys_day = ys[hour_mask]
    doy = df_weather.index[hour_mask].dayofyear.to_numpy()
    years = df_weather.index[hour_mask].year.to_numpy()

    unique_years = np.unique(years)
    n_years = len(unique_years)

    # 2. Weather window condition
    ww_obs_hourly = ((y_day[:,0] <= 30) & (y_day[:,1] <= 2)).astype(int)
    ww_syn_hourly = ((ys_day[:,0] <= 30) & (ys_day[:,1] <= 2)).astype(int)

    # 3. Prepare arrays
    max_doy = doy.max()
    daily_obs = np.zeros(max_doy)
    daily_syn = np.zeros(max_doy)

    # 4. Aggregate per day-of-year, then average across years
    for d in range(1, max_doy+1):
        mask = (doy == d)
        if mask.sum() > 0:
            total_obs = ww_obs_hourly[mask].sum()
            total_syn = ww_syn_hourly[mask].sum()

            # Convert to hours per day (divide by number of years)
            daily_obs[d-1] = total_obs / n_years
            daily_syn[d-1] = total_syn / n_years
        else:
            daily_obs[d-1] = np.nan
            daily_syn[d-1] = np.nan

    # 5. LOWESS smoothing for observed average year
    x = np.arange(max_doy)
    frac = 0.1  # smoothing parameter

    trend_obs = lowess(daily_obs, x, frac=frac, return_sorted=False)

    # --- One plot per synthetic year: observed avg (top) + synthetic year (bottom) ---
    syn_years = unique_years

    for yr in syn_years:
        # Mask for this synthetic year
        mask_yr = (years == yr)

        # Extract synthetic WW hours for this year
        ww_syn_year = ww_syn_hourly[mask_yr]
        doy_year = doy[mask_yr]

        # Prepare daily array for this year
        daily_syn_year = np.zeros(max_doy)
        daily_syn_year[:] = np.nan

        for d in range(1, max_doy+1):
            m = (doy_year == d)
            if m.sum() > 0:
                daily_syn_year[d-1] = ww_syn_year[m].sum()

        # LOWESS trend for this synthetic year
        trend_syn_year = lowess(daily_syn_year, x, frac=0.1, return_sorted=False)

        # --- Plot ---
        fig, axs = plt.subplots(2, 1, figsize=(FIG_WIDTH, 8), sharex=True)

        # Top: observed average year
        axs[0].plot(x, daily_obs, label="Observed (avg year)", lw=1.0)
        axs[0].plot(x, trend_obs, label="Trend (LOWESS)", lw=2, color="red")
        axs[0].set_ylabel("Weather window hours/day (07–19)")
        axs[0].set_title("Observed average year (07–19 only)")
        axs[0].set_ylim(0, 13)
        axs[0].legend()

        # Bottom: synthetic year
        axs[1].plot(x, daily_syn_year, label=f"Synthetic year {yr}", lw=1.0)
        axs[1].plot(x, trend_syn_year, label="Trend (LOWESS)", lw=2, color="red")
        axs[1].set_ylabel("Weather window hours/day (07–19)")
        axs[1].set_title(f"Synthetic year {yr} (07–19 only)")
        axs[1].set_ylim(0, 13)
        axs[1].legend()

        axs[1].set_xlabel("Day of year")
        plt.tight_layout()
        plt.show()
