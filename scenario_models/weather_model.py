import os
from pathlib import Path

import numpy as np
import pandas as pd

from data.fixed_data import data


class WeatherModel:
    def __init__(self):

        # Bootstrap parameters.
        self.bootstrap_block_days = 7
        self.bootstrap_overlap_days = 1
        self.bootstrap_perturbation_weight = 0.10

        self._weather_df = None
        self._month_segments = None
        self._month_scales = None
        self._wl_ids_cached = None

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
        
        wl_ids = [wl.id for wl in data.weather_locations]
        
        self.precompute(wl_ids)

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

        wide = pd.concat(
            [
                speed_wide.add_suffix("_speed"),
                height_wide.add_suffix("_height"),
            ],
            axis=1,
        ).sort_index()
        return wide

    def _build_month_segments(self, wl_ids):
        wide = self._build_joint_history(wl_ids)
        month_segments = {month: [] for month in range(12)}

        for (year, month), group in wide.groupby([wide.index.year, wide.index.month], sort=True):
            group = group.sort_index()
            if len(group) < self.bootstrap_block_days * 24:
                continue

            if group.index.to_series().diff().dropna().ne(pd.Timedelta(hours=1)).any():
                continue

            n_hours = (len(group) // 24) * 24
            if n_hours < self.bootstrap_block_days * 24:
                continue

            month_segments[month - 1].append(group.iloc[:n_hours].to_numpy(dtype=float))

        return month_segments

    def _build_month_scales(self, wl_ids):
        wide = self._build_joint_history(wl_ids)
        month_scales = {}

        for month_idx in range(12):
            month_wide = wide[wide.index.month == (month_idx + 1)]
            if month_wide.empty:
                month_scales[month_idx] = (1.0, 1.0)
                continue

            speed_values = month_wide.filter(like="_speed").to_numpy(dtype=float).ravel()
            height_values = month_wide.filter(like="_height").to_numpy(dtype=float).ravel()

            speed_scale = float(np.nanstd(speed_values)) if speed_values.size else 1.0
            height_scale = float(np.nanstd(height_values)) if height_values.size else 1.0
            month_scales[month_idx] = (
                speed_scale if speed_scale > 1e-8 else 1.0,
                height_scale if height_scale > 1e-8 else 1.0,
            )

        return month_scales

    def precompute(self, wl_ids):
        wl_ids = list(wl_ids)
        self._wl_ids_cached = wl_ids
        self._month_segments = self._build_month_segments(wl_ids)
        self._month_scales = self._build_month_scales(wl_ids)

        missing_months = [month for month, segments in self._month_segments.items() if len(segments) == 0]
        if missing_months:
            raise ValueError(
                "Missing historical bootstrap segments for months: "
                + ", ".join(str(month + 1) for month in missing_months)
            )

    def _sample_bootstrap_block(self, month_idx, rng):
        segments = self._month_segments.get(month_idx, [])
        if not segments:
            raise ValueError(f"No historical bootstrap segments available for month {month_idx + 1}.")

        segment = segments[int(rng.integers(0, len(segments)))]
        block_hours = self.bootstrap_block_days * 24
        segment_days = segment.shape[0] // 24
        max_start_day = segment_days - self.bootstrap_block_days
        if max_start_day < 0:
            raise ValueError(f"Historical month {month_idx + 1} does not contain enough days for a bootstrap block.")

        start_day = int(rng.integers(0, max_start_day + 1))
        start = start_day * 24
        return segment[start : start + block_hours].copy()

    def _blend_blocks(self, left_block, right_block):
        overlap_hours = self.bootstrap_overlap_days * 24
        if overlap_hours <= 0:
            return np.vstack([left_block, right_block])

        left_tail = left_block[-overlap_hours:]
        right_head = right_block[:overlap_hours]
        weights = np.linspace(0.0, 1.0, overlap_hours, endpoint=True)[:, None]
        blended = left_tail * (1.0 - weights) + right_head * weights
        return np.vstack([left_block[:-overlap_hours], blended, right_block[overlap_hours:]])

    def _simulate_month(self, month_idx, rng):
        target_hours = data.days_per_period * 24
        block_hours = self.bootstrap_block_days * 24
        overlap_hours = self.bootstrap_overlap_days * 24
        if block_hours <= overlap_hours:
            raise ValueError("Bootstrap block must be longer than the overlap window.")

        step_hours = block_hours - overlap_hours
        if target_hours <= block_hours:
            n_blocks = 1
        else:
            n_blocks = int(np.ceil((target_hours - block_hours) / step_hours)) + 1

        assembled = self._sample_bootstrap_block(month_idx, rng)
        for _ in range(1, n_blocks):
            next_block = self._sample_bootstrap_block(month_idx, rng)
            assembled = self._blend_blocks(assembled, next_block)

        while assembled.shape[0] < target_hours:
            next_block = self._sample_bootstrap_block(month_idx, rng)
            assembled = self._blend_blocks(assembled, next_block)

        assembled = assembled[:target_hours]

        perturbation_weight = np.clip(self.bootstrap_perturbation_weight, 0.0, 1.0)
        if perturbation_weight > 0.0:
            speed_scale, height_scale = self._month_scales.get(month_idx, (1.0, 1.0))
            latent = rng.normal(size=target_hours)
            n_loc = assembled.shape[1] // 2
            assembled[:, :n_loc] = assembled[:, :n_loc] + perturbation_weight * speed_scale * latent[:, None]
            assembled[:, n_loc:] = assembled[:, n_loc:] + perturbation_weight * height_scale * latent[:, None]

        return assembled

    def simulate(self, rng):
        if self._month_segments is None or self._wl_ids_cached is None:
            raise ValueError("WeatherModel must be precomputed before simulation.")

        month_order = (pd.to_datetime(data.periods, format="%b").month).to_numpy() - 1
        month_arrays = [self._simulate_month(month_idx, rng) for month_idx in month_order]
        y_out = np.vstack(month_arrays)
        y_out = np.maximum(y_out, 0.0)

        n_loc = len(self._wl_ids_cached)
        hours = np.arange(y_out.shape[0])
        frames = []
        for i, wl_id in enumerate(self._wl_ids_cached):
            frames.append(
                pd.DataFrame(
                    {
                        "d": hours // 24 + 1,
                        "hour": hours % 24,
                        "speed": y_out[:, i],
                        "height": y_out[:, n_loc + i],
                        "wl_id": wl_id,
                    }
                )
            )

        return pd.concat(frames, ignore_index=True)
