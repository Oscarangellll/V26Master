import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.fixed_data import data

def table_downtime_cost():

    rows = []

    wl_ids = {wf.weather_location_id for wf in data.wind_farms}

    for wl_id in wl_ids:
        print(wl_id)
        if wl_id != 2:
            continue
        # ---------------------------------------------------
        # Identify ISO and wind farm
        # ---------------------------------------------------
        iso = {
            wf.iso for wf in data.wind_farms
            if wf.weather_location_id == wl_id
        }.pop()

        w = {
            wf.name for wf in data.wind_farms
            if wf.weather_location_id == wl_id and wf.iso == iso
        }.pop()

        # ---------------------------------------------------
        # REAL DATA
        # ---------------------------------------------------
        df_p = pd.read_parquet(
            "data/price/price.parquet",
            filters=[("ISO3", "==", iso)]
        )

        df_w = pd.read_parquet(
            "data/weather/weather.parquet",
            filters=[("weather_location_id", "==", wl_id)]
        )

        df_w["power"] = data.power_curve(df_w["speed"])
        df_w = df_w[["power"]].resample("D").mean()

        df_real = df_p.join(df_w, how="inner")
        df_real["downtime_cost"] = df_real["power"] * 24 * df_real["price"]

        # monthly aggregation (REAL)
        real_monthly_mean = df_real.groupby(df_real.index.month)["downtime_cost"].mean()
        real_monthly_std  = df_real.groupby(df_real.index.month)["downtime_cost"].std()

        # ---------------------------------------------------
        # SYNTHETIC DATA
        # ---------------------------------------------------
        df_syn = pd.read_parquet(
            "data/scenario_data/downtime_cost",
            filters=[("w", "==", w)]
        )

        # map day index → month
        df_syn["month"] = ((df_syn["d"] - 1) // 30) + 1

        syn_monthly_mean = df_syn.groupby("month")["downtime_cost"].mean()
        syn_monthly_std  = df_syn.groupby("month")["downtime_cost"].std()

        # ---------------------------------------------------
        # ALIGN
        # ---------------------------------------------------
        df_cmp = pd.DataFrame({
            "real_mean": real_monthly_mean / 1000,
            "syn_mean": syn_monthly_mean / 1000,
            "real_std": real_monthly_std / 1000,
            "syn_std": syn_monthly_std / 1000
        }).dropna()

        df_cmp["mean_diff_%"] = 100 * (df_cmp["syn_mean"] - df_cmp["real_mean"]) / df_cmp["real_mean"]
        df_cmp["std_diff_%"]  = 100 * (df_cmp["syn_std"] - df_cmp["real_std"]) / df_cmp["real_std"]
        # ---------------------------------------------------
        # STORE ROWS
        # ---------------------------------------------------
        for month, row in df_cmp.iterrows():
            rows.append({
                "wl_id": wl_id,
                "month": pd.to_datetime(int(month), format="%m").strftime("%b"),

                "real_mean": row["real_mean"],
                "syn_mean": row["syn_mean"],
                "mean_diff_%": row["mean_diff_%"],

                "real_std": row["real_std"],
                "syn_std": row["syn_std"],
                "mean_std_%": row["std_diff_%"]
            })

    table = pd.DataFrame(rows)

    print(table.round(2))

    table.to_csv(
        "figures/tables/downtime_cost_summary.csv",
        index=False,
        float_format="%.2f"
    )

    return table
