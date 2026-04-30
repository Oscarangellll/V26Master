import os
from pathlib import Path

import numpy as np
import pandas as pd

from data.fixed_data import data


class WeatherModel:
    def __init__(self):
        self.rs = data.wind_speed_resolution
        self.rh = data.wave_height_resolution

        # Bootstrap parameters.
        self.bootstrap_block_days = int(os.environ.get("WEATHER_BOOTSTRAP_BLOCK_DAYS", "7"))
        self.bootstrap_overlap_hours = int(os.environ.get("WEATHER_BOOTSTRAP_OVERLAP_HOURS", "24"))
        self.bootstrap_transition_hours = int(os.environ.get("WEATHER_BOOTSTRAP_TRANSITION_HOURS", "24"))
        self.bootstrap_candidate_top_k = int(os.environ.get("WEATHER_BOOTSTRAP_CANDIDATE_TOP_K", "25"))
        self.bootstrap_candidate_pool = int(os.environ.get("WEATHER_BOOTSTRAP_CANDIDATE_POOL", "300"))
        self.bootstrap_transition_max_z = float(os.environ.get("WEATHER_BOOTSTRAP_TRANSITION_MAX_Z", "2.5"))
        self.bootstrap_same_month = (
            os.environ.get("WEATHER_BOOTSTRAP_SAME_MONTH", "1").strip().lower() not in {"0", "false", "no"}
        )

        self._weather_df = None
        self._fit()

    def _fit(self):
        data_dir = Path(os.environ.get("DATA_DIR", "data"))
        df_weather = pd.read_parquet(
            data_dir / "weather/weather.parquet"
        )
        if not isinstance(df_weather.index, pd.DatetimeIndex):
            if "time" in df_weather.columns:
                df_weather["time"] = pd.to_datetime(df_weather["time"])
                df_weather = df_weather.set_index("time")
            else:
                df_weather.index = pd.to_datetime(df_weather.index)
        df_weather = df_weather.sort_index()
        self._weather_df = df_weather

    def _month_of_sim(self):
        months = (pd.to_datetime(data.periods, format="%b").month).to_numpy() - 1
        return np.repeat(months, 24 * data.days_per_period)

    def _build_joint_history(self, wl_ids):
        df = self._weather_df
        df = df[df["weather_location_id"].isin(wl_ids)][["weather_location_id", "speed", "height"]].copy()

        long_df = df.reset_index().rename(columns={"index": "time"})
        speed_wide = (
            long_df.pivot(index="time", columns="weather_location_id", values="speed")
            .sort_index()
            .reindex(columns=wl_ids)
        )
        height_wide = (
            long_df.pivot(index="time", columns="weather_location_id", values="height")
            .sort_index()
            .reindex(columns=wl_ids)
        )

        valid = speed_wide.notna().all(axis=1) & height_wide.notna().all(axis=1)
        speed_wide = speed_wide.loc[valid]
        height_wide = height_wide.loc[valid]

        y_hist = np.stack([speed_wide.to_numpy(), height_wide.to_numpy()], axis=2)
        month_hist = speed_wide.index.month.to_numpy() - 1

        return y_hist, month_hist

    def _candidate_starts(self, month_hist, block_len, target_month):
        max_start = len(month_hist) - block_len
        if max_start < 0:
            return np.array([], dtype=int)

        if self.bootstrap_same_month:
            starts = np.where(month_hist[: max_start + 1] == target_month)[0]
        else:
            starts = np.arange(max_start + 1)

        if starts.size == 0:
            starts = np.arange(max_start + 1)

        return starts

    def _select_block_start(self, y_hist, month_hist, y_out, t, block_len, month_of_sim, rng):
        target_month = month_of_sim[t]
        starts = self._candidate_starts(month_hist, block_len, target_month)
        if starts.size == 0:
            raise ValueError("Not enough historical data to sample block bootstrap weather.")

        if t == 0:
            return int(rng.choice(starts))

        trans = min(self.bootstrap_transition_hours, t, block_len)
        if trans <= 0:
            return int(rng.choice(starts))

        prev_tail = y_out[t - trans : t]

        # Evaluate transition cost on a random pool for speed.
        pool_size = max(1, min(self.bootstrap_candidate_pool, starts.size))
        if pool_size < starts.size:
            starts = rng.choice(starts, size=pool_size, replace=False)

        scale = np.std(y_hist, axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)

        offsets = np.arange(trans)
        cand_heads = y_hist[starts[:, None] + offsets]
        costs = np.mean(np.abs(prev_tail[None, ...] - cand_heads) / scale[None, None, ...], axis=(1, 2, 3))

        order = np.argsort(costs)
        starts_sorted = starts[order]
        costs_sorted = costs[order]

        if self.bootstrap_transition_max_z > 0:
            valid = starts_sorted[costs_sorted <= self.bootstrap_transition_max_z]
            if valid.size > 0:
                starts_sorted = valid

        k = max(1, min(self.bootstrap_candidate_top_k, starts_sorted.size))
        return int(rng.choice(starts_sorted[:k]))

    def simulate_joint(self, wl_ids, rng):
        wl_ids = list(wl_ids)
        y_hist, month_hist = self._build_joint_history(wl_ids)

        month_of_sim = self._month_of_sim()
        T_sim = len(month_of_sim)
        n_loc = len(wl_ids)

        block_hours = max(24, int(self.bootstrap_block_days * 24))
        overlap_hours = max(0, int(self.bootstrap_overlap_hours))

        y_out = np.zeros((T_sim, n_loc, 2), dtype=float)

        t = 0
        while t < T_sim:
            overlap = 0 if t == 0 else min(overlap_hours, t)
            remaining_with_overlap = T_sim - t + overlap
            block_len = min(block_hours, remaining_with_overlap)

            start = self._select_block_start(
                y_hist=y_hist,
                month_hist=month_hist,
                y_out=y_out,
                t=t,
                block_len=block_len,
                month_of_sim=month_of_sim,
                rng=rng,
            )
            block = y_hist[start : start + block_len]

            if t == 0:
                write_len = min(block_len, T_sim)
                y_out[:write_len] = block[:write_len]
                t = write_len
                continue

            if overlap > 0:
                existing = y_out[t - overlap : t]
                incoming = block[:overlap]
                alpha = np.linspace(0.0, 1.0, overlap + 2)[1:-1].reshape(-1, 1, 1)
                y_out[t - overlap : t] = (1.0 - alpha) * existing + alpha * incoming

            new_part = block[overlap:]
            new_len = min(len(new_part), T_sim - t)
            if new_len > 0:
                y_out[t : t + new_len] = new_part[:new_len]
                t += new_len
            else:
                break

        y_out = np.maximum(y_out, 0)
        hours = np.arange(T_sim)

        out = {}
        for i, wl_id in enumerate(wl_ids):
            out[wl_id] = pd.DataFrame({
                "d": hours // 24 + 1,
                "hour": hours % 24,
                "speed": y_out[:, i, 0],
                "height": y_out[:, i, 1],
                "wl_id": wl_id,
            })

        return out

    def simulate(self, wl_id, rng):
        return self.simulate_joint([wl_id], rng)[wl_id]
