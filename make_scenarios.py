
import pandas as pd
import numpy as np

from data.fixed_data import data

# Downtime cost
# | w | d | s | downtime_cost |

df_w = pd.read_parquet("data/scenario_data/weather.parquet")
df_w["power"] = data.power_curve(df_w["speed"])

rows = []

for w in data.wind_farms:
    df = df_w[df_w["wl_id"] == w.weather_location_id].copy()
    
    df = df.groupby(["d", "s"])["power"].mean().reset_index()
    df["w"] = w.name
    df["iso"] = w.iso
    
    rows.append(df)

df_w = pd.concat(rows, ignore_index=True)

df_p = pd.read_parquet("data/scenario_data/price.parquet")

df = df_w.merge(df_p, on=["iso", "d", "s"])
df["downtime_cost"] = df["power"] * 24 * df["price"]
df = df[["w", "d", "s", "downtime_cost"]]

df.to_parquet("data/scenario_data/downtime_cost.parquet")
df.to_csv("data/scenario_data/downtime_cost.csv", index=False)
exit()

# Weather window
# | wl_id | d | s | weather_window |


def make_weather_windows():
    df_weather = pd.read_csv("data/scenario_data/weather.csv")

    working_hours = list(range(data.work_day_start, data.work_day_end))
    df_working = df_weather[df_weather["hour"].isin(working_hours)]

    rows = []

    for h in data.vessel_types:
        max_speed = h.max_wind
        max_height = h.max_wave

        for (s, wl_id, d), group in df_working.groupby(["s", "wl_id", "d"]):
            feasible = (
                (group["speed"] <= max_speed) &
                (group["height"] <= max_height)
            ).to_numpy().astype(int)

            max_len = 0
            current_len = 0
            for val in feasible:
                if val:
                    current_len += 1
                    max_len = max(max_len, current_len)
                else:
                    current_len = 0

            rows.append({
                "h": h.name,
                "wl_id": wl_id,
                "d": d,
                "ww": max_len,
                "s": s
            })

    df_ww = pd.DataFrame(rows)
    df_ww.to_csv("data/scenario_data/weather_windows.csv", index=False)

# make_weather_windows()
########
    df.to_parquet("data/scenario_data/failures.parquet")
    df.to_csv("data/scenario_data/failures.csv", index=False)

#generate_failures()

def make_weather_windows():
    df_weather = pd.read_csv("data/scenario_data/weather.csv")

    working_hours = list(range(data.work_day_start, data.work_day_end))
    df_working = df_weather[df_weather["hour"].isin(working_hours)]

    rows = []

    for h in data.vessel_types:
        max_speed = h.max_wind
        max_height = h.max_wave

        for (s, wl_id, d), group in df_working.groupby(["s", "wl_id", "d"]):
            feasible = (
                (group["speed"] <= max_speed) & 
                (group["height"] <= max_height)
            ).to_numpy().astype(int)

            max_len = 0
            current_len = 0
            for val in feasible:
                if val:
                    current_len += 1
                    max_len = max(max_len, current_len)
                else:
                    current_len = 0

            rows.append({
                "h": h.name,
                "wl_id": wl_id,
                "d": d,
                "ww": max_len,
                "s": s
            })
    
    df_ww = pd.DataFrame(rows)
    df_ww.to_csv("data/scenario_data/weather_windows.csv", index=False)

# make_weather_windows()
















