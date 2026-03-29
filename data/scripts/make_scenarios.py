import itertools
import os
from pathlib import Path

from haversine import haversine, Unit
import numpy as np
import pandas as pd

from data.fixed_data import data

scenario_data_dir = Path(os.environ.get("SCENARIO_DATA_DIR", "data/scenario_data"))

def _make_downtime_cost():

    # Downtime cost
    # | w | d | s | downtime_cost |

    df_w = pd.read_parquet(scenario_data_dir / "weather.parquet")
    df_w["power"] = data.power_curve(df_w["speed"])

    rows = []

    for w in data.wind_farms:
        df = df_w[df_w["wl_id"] == w.weather_location_id].copy()
    
        df = df.groupby(["d", "s"])["power"].mean().reset_index()
        df["w"] = w.name
        df["iso"] = w.iso
    
        rows.append(df)

    df_w = pd.concat(rows, ignore_index=True)

    df_p = pd.read_parquet(scenario_data_dir / "price.parquet")

    df = df_w.merge(df_p, on=["iso", "d", "s"])
    df["downtime_cost"] = df["power"] * 24 * df["price"]
    df = df[["w", "d", "s", "downtime_cost"]]

    df.to_parquet(scenario_data_dir / "downtime_cost.parquet")
    #df.to_csv("data/scenario_data/downtime_cost.csv", index=False)


def _make_weather_windows():
    
    # Weather window
    # | h | wl_id | d | s | ww |
    
    df = pd.read_parquet(scenario_data_dir / "weather.parquet")

    working_hours = list(range(data.work_day_start, data.work_day_end))
    df = df[df["hour"].isin(working_hours)]

    rows = []
    for h in data.vessel_types:
        max_height = h.max_wave

        for (s, wl_id, d), group in df.groupby(["s", "wl_id", "d"]):
            feasible = (
                group["height"] <= max_height
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
                "s": s,
                "ww": max_len,
            })

    df = pd.DataFrame(rows)
    df.to_parquet(scenario_data_dir / "weather_windows.parquet")
    #df.to_csv("data/scenario_data/weather_windows.csv", index=False)


def _make_patterns():
    
    # Pattern
    # | m | k | task_count | duration_of_pattern |
    
    mcats = data.maintenance_categories

    ub_ww = data.work_day_end - data.work_day_start
    
    max_count = {m.name: int(ub_ww // m.duration) for m in mcats}
    
    ranges = [range(max_count[m.name] + 1) for m in mcats]
    
    rows = []
     
    k = 1
    for pattern in itertools.product(*ranges):
        if all(c == 0 for c in pattern):
            continue
        
        duration = sum(c * m.duration for c, m in zip(pattern, mcats))
        if duration > ub_ww:
            continue

        for c, m in zip(pattern, mcats):
            rows.append({
                "m": m.name,
                "k": k,
                "task_count": c,
                "duration_of_pattern": duration
            })

        k += 1
    
    df = pd.DataFrame(rows)
    df.to_parquet(scenario_data_dir / "patterns.parquet")
    #df.to_csv("data/scenario_data/patterns.csv", index=False)

def _make_pattern_sets():
    
    def remove_dominated(pattern_idxs, vectors):
        kept = []

        for k1 in pattern_idxs:
            v1 = vectors[k1]
            
            dominated = False
            for k2 in pattern_idxs:
                if k1 == k2:
                    continue

                v2 = vectors[k2]
                if all(x <= y for x, y in zip(v1, v2)) and any(x < y for x, y in zip(v1, v2)):
                    dominated = True
                    break
            
            if not dominated:
                kept.append(k1)

        return kept

    # K_S
    # | h | b | w | d | s | list(k) |
    #
    # K_M
    # | h | w | d | s | list(k) |

    df_pattern = pd.read_parquet(scenario_data_dir / "patterns.parquet")
    
    pattern_vectors = (
        df_pattern
        .pivot(index="k", columns="m", values="task_count")
        .apply(list, axis=1)
        .to_dict()
    )

    allowed_vessel_types = {m.name: m.vessel_types for m in data.maintenance_categories}
    K = {h.name: [] for h in data.vessel_types}
    
    for k, group in df_pattern.groupby("k"):
        active_categories = group.loc[group["task_count"] > 0, "m"].tolist()

        for h in data.vessel_types:
            if all(h.name in allowed_vessel_types[m] for m in active_categories):
                K[h.name].append(k)
    
    
    L = (
        df_pattern
        .drop_duplicates("k")
        .set_index("k")["duration_of_pattern"]
        .to_dict()
    )
    
    L_RT = {(h.name, b.name, w.name):
        2 * haversine((b.lat, b.lon), (w.lat, w.lon), unit=Unit.KILOMETERS) / h.travel_speed
        for h in data.vessel_types if not h.multiday
        for b in data.bases
        for w in data.wind_farms
    }
    
    df_ww = pd.read_parquet(scenario_data_dir / "weather_windows.parquet")
    days = df_ww["d"].unique()
    scenarios = df_ww["s"].unique()
    
    weather_window = (
        df_ww
        .set_index(["h", "wl_id", "d", "s"])["ww"]
        .to_dict()
    )
    
    rows_K_S = []
    rows_K_M = []
    
    for h in data.vessel_types:
        for w in data.wind_farms:
            wl_id = w.weather_location_id

            for d in days:
                for s in scenarios:
                    ww = weather_window[(h.name, wl_id, d, s)]
                    frik = 1 + data.work_friction
                    
                    if h.multiday:
                        feasible_patterns = [
                            k for k in K[h.name] if
                            frik * L[k] <= ww
                        ]
                        feasible_patterns = remove_dominated(   
                            feasible_patterns,
                            pattern_vectors
                        )
                        rows_K_M.append({
                            "h": h.name,
                            "w": w.name,
                            "d": d,
                            "s": s,
                            "patterns": feasible_patterns
                        })

                    else:
                        for b in data.bases:
                            rt = L_RT[(h.name, b.name, w.name)]
                            
                            feasible_patterns = [
                                k for k in K[h.name] if
                                frik * L[k] + rt <= ww
                            ]
                            feasible_patterns = remove_dominated(   
                                feasible_patterns,
                                pattern_vectors
                            )
                            
                            rows_K_S.append({
                                "h": h.name,
                                "b": b.name,
                                "w": w.name,
                                "d": d,
                                "s": s,
                                "patterns": feasible_patterns
                            })
    
    df_K_S = pd.DataFrame(rows_K_S)
    df_K_S.to_parquet(scenario_data_dir / "singleday_pattern_set.parquet")
    #df_K_S.to_csv("data/scenario_data/singleday_pattern_set.csv", index=False)
    
    df_K_M = pd.DataFrame(rows_K_M)
    df_K_M.to_parquet(scenario_data_dir / "multiday_pattern_set.parquet")
    #df_K_M.to_csv("data/scenario_data/multiday_pattern_set.csv", index=False)

def make_scenarios():
    print("Making costs") 
    _make_downtime_cost()
    print("Making WW")
    _make_weather_windows()
    print("Makgin patterns")
    _make_patterns()
    print("Making pattern sets")
    _make_pattern_sets()







