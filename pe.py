import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path

from data.fixed_data import data


TARGET_VESSEL = "CTV"
TARGET_WL_IDS = [3, 4]
MAX_SCENARIOS = 3000
WW_KIND_THRESHOLD = 8
KIND_MONTHS = {10, 11, 12, 1, 2, 3}
SCENARIO_DATA_DIR = Path(os.environ.get("SCENARIO_DATA_DIR", "data/scenario_data"))


def locations_label():
	return ",".join(str(wl_id) for wl_id in TARGET_WL_IDS)


def coefficient_of_variation(series):
	mean_val = series.mean()
	if pd.isna(mean_val) or mean_val == 0:
		return 0.0
	return float(series.std() / mean_val)


def keep_first_scenarios(df):
	first_scenarios = sorted(df["s"].unique())[:MAX_SCENARIOS]
	return df[df["s"].isin(first_scenarios)].copy()


def longest_consecutive_ones(values):
	max_len = 0
	current = 0
	for value in values:
		if value:
			current += 1
			if current > max_len:
				max_len = current
		else:
			current = 0
	return max_len


def longest_consecutive_bad_days(days, is_bad):
	max_len = 0
	current = 0
	prev_day = None

	for day, bad in zip(days, is_bad):
		if bad:
			if prev_day is not None and day == prev_day + 1:
				current += 1
			else:
				current = 1
			if current > max_len:
				max_len = current
		else:
			current = 0
		prev_day = day

	return max_len


def build_scenario_kindness_scores(metric, ww_threshold=8.0):
	df = pd.read_parquet(
		SCENARIO_DATA_DIR / "weather_windows",
		columns=["h", "wl_id", "s", "d", "ww"],
	)
	df = df.dropna(subset=["h", "wl_id", "s", "d", "ww"]).copy()
	df["h"] = df["h"].astype(str)
	df["wl_id"] = df["wl_id"].astype(int)
	df["s"] = df["s"].astype(int)
	df["d"] = df["d"].astype(int)
	df["ww"] = df["ww"].astype(float)
	df = keep_first_scenarios(df)

	df = df[(df["h"] == TARGET_VESSEL) & (df["wl_id"].isin(TARGET_WL_IDS))].copy()
	df["month"] = df["d"].map(day_to_month)
	df = df[df["month"].isin(KIND_MONTHS)].copy()

	all_scenarios = pd.Index(sorted(df["s"].unique()), name="s")

	if metric == "count_location_days_over_threshold":
		scores = (
			df[df["ww"] >= ww_threshold]
			.groupby(["s", "wl_id"])
			.size()
			.groupby("s")
			.sum()
			.reindex(all_scenarios, fill_value=0)
		)
		return scores.rename("score")

	if metric == "total_window_hours":
		scores = (
			df.groupby("s")["ww"]
			.sum()
			.reindex(all_scenarios, fill_value=0)
		)
		return scores.rename("score")

	if metric == "count_location_days_under_threshold":
		bad_counts = (
			df[df["ww"] < ww_threshold]
			.groupby(["s", "wl_id"])
			.size()
			.groupby("s")
			.sum()
			.reindex(all_scenarios, fill_value=0)
		)
		# Keep "higher is kinder" orientation.
		return (-bad_counts).rename("score")

	if metric == "max_bad_streak_under_threshold":
		streak_scores = {}
		for s, scenario_group in df.groupby("s"):
			total_max_streak = 0
			for _, loc_group in scenario_group.groupby("wl_id"):
				loc_group = loc_group.sort_values("d")
				max_streak = longest_consecutive_bad_days(
					loc_group["d"].to_numpy(),
					(loc_group["ww"] < ww_threshold).to_numpy(),
				)
				total_max_streak += int(max_streak)
			streak_scores[int(s)] = -float(total_max_streak)

		scores = pd.Series(streak_scores).reindex(all_scenarios, fill_value=0.0)
		scores.index.name = "s"
		return scores.rename("score")

	raise ValueError(
		"Unsupported metric. Use count_location_days_over_threshold, total_window_hours, "
		"count_location_days_under_threshold, or max_bad_streak_under_threshold."
	)


def build_real_kindness_scores(metric, ww_threshold=8.0):
	max_wave = next(v.max_wave for v in data.vessel_types if v.name == TARGET_VESSEL)

	df = pd.read_csv("data/weather/weather.csv")
	df = df[df["weather_location_id"].isin(TARGET_WL_IDS)].copy()
	df["time"] = pd.to_datetime(df["time"])
	df["hour"] = df["time"].dt.hour
	df["year"] = df["time"].dt.year
	df["day_of_year"] = df["time"].dt.dayofyear
	df["month"] = df["time"].dt.month

	working_hours = list(range(data.work_day_start, data.work_day_end))
	df = df[df["hour"].isin(working_hours)].copy()

	daily_ww = (
		df.groupby(["weather_location_id", "year", "day_of_year", "month"])["height"]
		.apply(lambda x: longest_consecutive_ones((x <= max_wave).to_numpy()))
		.reset_index(name="ww")
	)
	daily_ww = daily_ww[daily_ww["month"].isin(KIND_MONTHS)].copy()

	all_years = pd.Index(sorted(daily_ww["year"].unique()), name="year")

	if metric == "count_location_days_over_threshold":
		scores = (
			daily_ww[daily_ww["ww"] >= ww_threshold]
			.groupby(["year", "weather_location_id"])
			.size()
			.groupby("year")
			.sum()
			.reindex(all_years, fill_value=0)
		)
		return scores.rename("score")

	if metric == "total_window_hours":
		scores = (
			daily_ww.groupby("year")["ww"]
			.sum()
			.reindex(all_years, fill_value=0)
		)
		return scores.rename("score")

	if metric == "count_location_days_under_threshold":
		bad_counts = (
			daily_ww[daily_ww["ww"] < ww_threshold]
			.groupby(["year", "weather_location_id"])
			.size()
			.groupby("year")
			.sum()
			.reindex(all_years, fill_value=0)
		)
		return (-bad_counts).rename("score")

	if metric == "max_bad_streak_under_threshold":
		streak_scores = {}
		for year, year_group in daily_ww.groupby("year"):
			total_max_streak = 0
			for _, loc_group in year_group.groupby("weather_location_id"):
				loc_group = loc_group.sort_values("day_of_year")
				max_streak = longest_consecutive_bad_days(
					loc_group["day_of_year"].to_numpy(),
					(loc_group["ww"] < ww_threshold).to_numpy(),
				)
				total_max_streak += int(max_streak)
			streak_scores[int(year)] = -float(total_max_streak)

		scores = pd.Series(streak_scores).reindex(all_years, fill_value=0.0)
		scores.index.name = "year"
		return scores.rename("score")

	raise ValueError(
		"Unsupported metric. Use count_location_days_over_threshold, total_window_hours, "
		"count_location_days_under_threshold, or max_bad_streak_under_threshold."
	)


def build_scenario_joint_bad_day_share(ww_threshold=4.0):
	df = pd.read_parquet(
		SCENARIO_DATA_DIR / "weather_windows",
		columns=["h", "wl_id", "s", "d", "ww"],
	)
	df = df.dropna(subset=["h", "wl_id", "s", "d", "ww"]).copy()
	df["h"] = df["h"].astype(str)
	df["wl_id"] = df["wl_id"].astype(int)
	df["s"] = df["s"].astype(int)
	df["d"] = df["d"].astype(int)
	df["ww"] = df["ww"].astype(float)
	df = keep_first_scenarios(df)

	df = df[(df["h"] == TARGET_VESSEL) & (df["wl_id"].isin(TARGET_WL_IDS))].copy()
	df["month"] = df["d"].map(day_to_month)
	df = df[df["month"].isin(KIND_MONTHS)].copy()

	all_scenarios = pd.Index(sorted(df["s"].unique()), name="s")

	daily = (
		df.groupby(["s", "d"], as_index=False)
		.agg(
			n_locs=("wl_id", "nunique"),
			n_bad=("ww", lambda x: int((x < ww_threshold).sum())),
		)
	)
	daily = daily[daily["n_locs"] == len(TARGET_WL_IDS)].copy()
	daily["all_bad"] = daily["n_bad"] == len(TARGET_WL_IDS)

	shares = (
		daily.groupby("s")["all_bad"]
		.mean()
		.reindex(all_scenarios, fill_value=0.0)
	)
	return shares.rename("joint_bad_share")


def build_real_joint_bad_day_share(ww_threshold=4.0):
	max_wave = next(v.max_wave for v in data.vessel_types if v.name == TARGET_VESSEL)

	df = pd.read_csv("data/weather/weather.csv")
	df = df[df["weather_location_id"].isin(TARGET_WL_IDS)].copy()
	df["time"] = pd.to_datetime(df["time"])
	df["hour"] = df["time"].dt.hour
	df["year"] = df["time"].dt.year
	df["day_of_year"] = df["time"].dt.dayofyear
	df["month"] = df["time"].dt.month

	working_hours = list(range(data.work_day_start, data.work_day_end))
	df = df[df["hour"].isin(working_hours)].copy()

	daily_ww = (
		df.groupby(["weather_location_id", "year", "day_of_year", "month"])["height"]
		.apply(lambda x: longest_consecutive_ones((x <= max_wave).to_numpy()))
		.reset_index(name="ww")
	)
	daily_ww = daily_ww[daily_ww["month"].isin(KIND_MONTHS)].copy()

	all_years = pd.Index(sorted(daily_ww["year"].unique()), name="year")

	daily = (
		daily_ww.groupby(["year", "day_of_year"], as_index=False)
		.agg(
			n_locs=("weather_location_id", "nunique"),
			n_bad=("ww", lambda x: int((x < ww_threshold).sum())),
		)
	)
	daily = daily[daily["n_locs"] == len(TARGET_WL_IDS)].copy()
	daily["all_bad"] = daily["n_bad"] == len(TARGET_WL_IDS)

	shares = (
		daily.groupby("year")["all_bad"]
		.mean()
		.reindex(all_years, fill_value=0.0)
	)
	return shares.rename("joint_bad_share")


def build_scenario_per_location_max_bad_streak(ww_threshold=4.0):
	df = pd.read_parquet(
		SCENARIO_DATA_DIR / "weather_windows",
		columns=["h", "wl_id", "s", "d", "ww"],
	)
	df = df.dropna(subset=["h", "wl_id", "s", "d", "ww"]).copy()
	df["h"] = df["h"].astype(str)
	df["wl_id"] = df["wl_id"].astype(int)
	df["s"] = df["s"].astype(int)
	df["d"] = df["d"].astype(int)
	df["ww"] = df["ww"].astype(float)
	df = keep_first_scenarios(df)

	df = df[(df["h"] == TARGET_VESSEL) & (df["wl_id"].isin(TARGET_WL_IDS))].copy()
	df["month"] = df["d"].map(day_to_month)
	df = df[df["month"].isin(KIND_MONTHS)].copy()

	all_scenarios = sorted(df["s"].unique())
	streak_map = {}
	for (s, wl_id), group in df.groupby(["s", "wl_id"]):
		group = group.sort_values("d")
		streak_map[(int(s), int(wl_id))] = int(
			longest_consecutive_bad_days(
				group["d"].to_numpy(),
				(group["ww"] < ww_threshold).to_numpy(),
			)
		)

	idx = pd.MultiIndex.from_product(
		[all_scenarios, TARGET_WL_IDS],
		names=["s", "wl_id"],
	)
	streaks = pd.Series(streak_map).reindex(idx, fill_value=0).rename("max_bad_streak")
	return streaks.reset_index()


def build_real_per_location_max_bad_streak(ww_threshold=4.0):
	max_wave = next(v.max_wave for v in data.vessel_types if v.name == TARGET_VESSEL)

	df = pd.read_csv("data/weather/weather.csv")
	df = df[df["weather_location_id"].isin(TARGET_WL_IDS)].copy()
	df["time"] = pd.to_datetime(df["time"])
	df["hour"] = df["time"].dt.hour
	df["year"] = df["time"].dt.year
	df["day_of_year"] = df["time"].dt.dayofyear
	df["month"] = df["time"].dt.month

	working_hours = list(range(data.work_day_start, data.work_day_end))
	df = df[df["hour"].isin(working_hours)].copy()

	daily_ww = (
		df.groupby(["weather_location_id", "year", "day_of_year", "month"])["height"]
		.apply(lambda x: longest_consecutive_ones((x <= max_wave).to_numpy()))
		.reset_index(name="ww")
	)
	daily_ww = daily_ww[daily_ww["month"].isin(KIND_MONTHS)].copy()

	all_years = sorted(daily_ww["year"].unique())
	streak_map = {}
	for (year, wl_id), group in daily_ww.groupby(["year", "weather_location_id"]):
		group = group.sort_values("day_of_year")
		streak_map[(int(year), int(wl_id))] = int(
			longest_consecutive_bad_days(
				group["day_of_year"].to_numpy(),
				(group["ww"] < ww_threshold).to_numpy(),
			)
		)

	idx = pd.MultiIndex.from_product(
		[all_years, TARGET_WL_IDS],
		names=["year", "wl_id"],
	)
	streaks = pd.Series(streak_map).reindex(idx, fill_value=0).rename("max_bad_streak")
	return streaks.reset_index()


def plot_persistence_diagnostics(ww_threshold=4.0):
	joint_scenario = build_scenario_joint_bad_day_share(ww_threshold=ww_threshold).sort_values()
	joint_real = build_real_joint_bad_day_share(ww_threshold=ww_threshold)

	per_loc_scenario = build_scenario_per_location_max_bad_streak(ww_threshold=ww_threshold)
	per_loc_real = build_real_per_location_max_bad_streak(ww_threshold=ww_threshold)

	n_panels = 1 + len(TARGET_WL_IDS)
	fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
	if n_panels == 1:
		axes = [axes]

	# Panel 1: share of winter days where all locations are bad simultaneously.
	ax = axes[0]
	n = len(joint_scenario)
	ranks = np.arange(1, n + 1)
	ax.plot(
		ranks,
		joint_scenario.values,
		color="tab:blue",
		linewidth=1.8,
		label="Alle scenarioer",
	)
	real_values = joint_real.values
	real_ranks = np.searchsorted(joint_scenario.values, real_values, side="right") + 1
	real_ranks = np.clip(real_ranks, 1, n + 1)
	ax.scatter(
		real_ranks,
		real_values,
		s=36,
		marker="x",
		color="tab:orange",
		label="Ekte aar",
	)
	ax.set_xlim(1, n + 1)
	ax.set_title(f"Samtidig bad-day andel (WW < {ww_threshold})")
	ax.set_xlabel("Rank (lavere risiko til venstre)")
	ax.set_ylabel("Andel vinterdager")
	ax.grid(True, alpha=0.25)

	# Panels 2..N: max bad streak per location.
	for i, wl_id in enumerate(TARGET_WL_IDS, start=1):
		ax = axes[i]
		scen_loc = (
			per_loc_scenario[per_loc_scenario["wl_id"] == wl_id]
			.set_index("s")["max_bad_streak"]
			.sort_values()
		)
		real_loc = (
			per_loc_real[per_loc_real["wl_id"] == wl_id]
			.set_index("year")["max_bad_streak"]
		)

		n_loc = len(scen_loc)
		ranks_loc = np.arange(1, n_loc + 1)
		ax.plot(
			ranks_loc,
			scen_loc.values,
			color="tab:blue",
			linewidth=1.8,
			label="Alle scenarioer",
		)
		real_values_loc = real_loc.values
		real_ranks_loc = np.searchsorted(scen_loc.values, real_values_loc, side="right") + 1
		real_ranks_loc = np.clip(real_ranks_loc, 1, n_loc + 1)
		ax.scatter(
			real_ranks_loc,
			real_values_loc,
			s=36,
			marker="x",
			color="tab:orange",
			label="Ekte aar",
		)
		ax.set_xlim(1, n_loc + 1)
		ax.set_title(f"Maks bad streak per lokasjon (wl_id={wl_id})")
		ax.set_xlabel("Rank (kortere streak til venstre)")
		ax.set_ylabel("Dager")
		ax.grid(True, alpha=0.25)

	handles, labels = axes[0].get_legend_handles_labels()
	if handles:
		fig.legend(handles, labels, loc="upper center", ncol=2)

	fig.suptitle(
		f"Persistence diagnostics ({TARGET_VESSEL}, wl_ids={locations_label()}, first {MAX_SCENARIOS} scenarios)",
		y=0.98,
	)
	fig.tight_layout(rect=[0, 0, 1, 0.93])
	plt.savefig("figures/persistence_diagnostics.svg", dpi=100, bbox_inches="tight")
	plt.close()


def plot_kindness_metric_distributions():
	metric_configs = [
		("count_location_days_over_threshold", "Count days WW >= 8", 8.0, "Kindness score"),
		("total_window_hours", "Total WW hours", None, "Kindness score"),
		("count_location_days_under_threshold", "Count days WW < 4", 4.0, "Kindness score"),
		("joint_bad_day_share", "Samtidig dårligvær-andel (alle lokasjoner)", 4.0, "Neg. andel samtidig bad-day"),
	]

	fig, axes = plt.subplots(2, 2, figsize=(15, 10))
	axes = axes.flatten()

	for ax, (metric, title, threshold, y_label) in zip(axes, metric_configs):
		if metric == "joint_bad_day_share":
			# Use negative share so "higher is kinder" orientation is consistent.
			scores = (-build_scenario_joint_bad_day_share(ww_threshold=threshold)).rename("score")
			real_scores = (-build_real_joint_bad_day_share(ww_threshold=threshold)).rename("score")
		elif threshold is None:
			scores = build_scenario_kindness_scores(metric)
			real_scores = build_real_kindness_scores(metric)
		else:
			scores = build_scenario_kindness_scores(metric, ww_threshold=threshold)
			real_scores = build_real_kindness_scores(metric, ww_threshold=threshold)

		sorted_scores_with_id = scores.sort_values(ascending=False)
		n = len(sorted_scores_with_id)
		ranks = np.arange(1, n + 1)

		ax.plot(
			ranks,
			sorted_scores_with_id.values,
			color="tab:blue",
			linewidth=1.8,
			label="Alle scenarioer",
		)

		# Place each real year on the full scenario axis by score position.
		scenario_values_desc = sorted_scores_with_id.values
		real_values = real_scores.values
		real_ranks = np.searchsorted(-scenario_values_desc, -real_values, side="right") + 1
		real_ranks = np.clip(real_ranks, 1, n + 1)
		ax.scatter(
			real_ranks,
			real_values,
			s=28,
			alpha=0.9,
			marker="x",
			color="tab:orange",
			label="Ekte aar",
		)
		ax.set_xlim(1, n + 1)

		ax.set_title(title)
		ax.set_xlabel("Rank (snillest til venstre)")
		ax.set_ylabel(y_label)
		ax.grid(True, alpha=0.25)

	handles, labels = axes[0].get_legend_handles_labels()
	if handles:
		fig.legend(handles, labels, loc="upper center", ncol=2)

	fig.suptitle(
		f"Sorted metric distributions ({TARGET_VESSEL}, wl_ids={locations_label()}, first {MAX_SCENARIOS} scenarios)",
		y=0.98,
	)
	fig.tight_layout(rect=[0, 0, 1, 0.95])
	plt.savefig("figures/kindness_metric_distributions.svg", dpi=100, bbox_inches="tight")
	plt.close()


def build_scenario_cv():
	df = pd.read_parquet(
		SCENARIO_DATA_DIR / "weather_windows",
		columns=["h", "wl_id", "s", "d", "ww"],
	)
	df = df.dropna(subset=["h", "wl_id", "s", "d", "ww"]).copy()
	df["h"] = df["h"].astype(str)
	df["wl_id"] = df["wl_id"].astype(int)
	df["s"] = df["s"].astype(int)
	df["d"] = df["d"].astype(int)
	df["ww"] = df["ww"].astype(float)
	df = keep_first_scenarios(df)

	df = df[(df["h"] == TARGET_VESSEL) & (df["wl_id"].isin(TARGET_WL_IDS))].copy()

	# Sum weather windows across selected locations for each scenario-day before CV.
	df_sum = df.groupby(["s", "d"], as_index=False)["ww"].sum()
	return df_sum.groupby("d")["ww"].agg(coefficient_of_variation)


def day_to_month(day_index):
	# Map model day index to month using a non-leap reference year.
	reference = pd.Timestamp("2011-01-01")
	return int((reference + pd.Timedelta(days=int(day_index) - 1)).month)


def build_scenario_winter_kindness():
	df = pd.read_parquet(
		SCENARIO_DATA_DIR / "weather_windows",
		columns=["h", "wl_id", "s", "d", "ww"],
	)
	df = df.dropna(subset=["h", "wl_id", "s", "d", "ww"]).copy()
	df["h"] = df["h"].astype(str)
	df["wl_id"] = df["wl_id"].astype(int)
	df["s"] = df["s"].astype(int)
	df["d"] = df["d"].astype(int)
	df["ww"] = df["ww"].astype(float)
	df = keep_first_scenarios(df)

	df = df[(df["h"] == TARGET_VESSEL) & (df["wl_id"].isin(TARGET_WL_IDS))].copy()
	df["month"] = df["d"].map(day_to_month)
	df = df[df["month"].isin(KIND_MONTHS)]

	all_scenarios = pd.Index(sorted(df["s"].unique()), name="s")
	counts = (
		df[df["ww"] >= WW_KIND_THRESHOLD]
		.groupby(["s", "wl_id"])
		.size()
		.groupby("s")
		.sum()
		.reindex(all_scenarios, fill_value=0)
		.rename("n_good_winter_days")
	)

	counts = counts.sort_values(ascending=False).reset_index()
	counts["rank"] = range(1, len(counts) + 1)
	return counts


def build_real_cv():
	max_wave = next(v.max_wave for v in data.vessel_types if v.name == TARGET_VESSEL)

	df = pd.read_csv("data/weather/weather.csv")
	df = df[df["weather_location_id"].isin(TARGET_WL_IDS)].copy()
	df["time"] = pd.to_datetime(df["time"])
	df["hour"] = df["time"].dt.hour
	df["year"] = df["time"].dt.year
	df["day_of_year"] = df["time"].dt.dayofyear

	working_hours = list(range(data.work_day_start, data.work_day_end))
	df = df[df["hour"].isin(working_hours)].copy()
	df = df.sort_values(["year", "day_of_year", "hour"])

	daily_ww = (
		df.groupby(["weather_location_id", "year", "day_of_year"])["height"]
		.apply(lambda x: longest_consecutive_ones((x <= max_wave).to_numpy()))
		.reset_index(name="ww")
	)

	daily_ww = daily_ww.groupby(["year", "day_of_year"], as_index=False)["ww"].sum()

	return daily_ww.groupby("day_of_year")["ww"].agg(coefficient_of_variation)


def build_real_winter_kindness():
	max_wave = next(v.max_wave for v in data.vessel_types if v.name == TARGET_VESSEL)

	df = pd.read_csv("data/weather/weather.csv")
	df = df[df["weather_location_id"].isin(TARGET_WL_IDS)].copy()
	df["time"] = pd.to_datetime(df["time"])
	df["hour"] = df["time"].dt.hour
	df["year"] = df["time"].dt.year
	df["month"] = df["time"].dt.month
	df["date"] = df["time"].dt.date

	working_hours = list(range(data.work_day_start, data.work_day_end))
	df = df[df["hour"].isin(working_hours)].copy()
	df = df.sort_values(["weather_location_id", "year", "date", "hour"])

	daily_ww = (
		df.groupby(["weather_location_id", "year", "month", "date"])["height"]
		.apply(lambda x: longest_consecutive_ones((x <= max_wave).to_numpy()))
		.reset_index(name="ww")
	)

	daily_ww = daily_ww[daily_ww["month"].isin(KIND_MONTHS)].copy()

	counts = (
		daily_ww[daily_ww["ww"] >= WW_KIND_THRESHOLD]
		.groupby(["year", "weather_location_id"])
		.size()
		.groupby("year")
		.sum()
		.rename("n_good_winter_days")
	)

	all_years = pd.Index(sorted(daily_ww["year"].unique()), name="year")
	counts = counts.reindex(all_years, fill_value=0).reset_index()
	counts = counts.sort_values("n_good_winter_days", ascending=False).reset_index(drop=True)
	counts["rank"] = range(1, len(counts) + 1)
	return counts


def main():
	scenario_cv = build_scenario_cv().rename("scenario_cv")
	real_cv = build_real_cv().rename("real_cv")
	winter_kindness = build_scenario_winter_kindness()
	real_winter_kindness = build_real_winter_kindness()
	scenario_cv.index.name = "d"
	real_cv.index.name = "d"

	compare = pd.concat([scenario_cv, real_cv], axis=1, join="inner").reset_index()

	plt.figure(figsize=(12, 5))
	plt.plot(compare["d"], compare["scenario_cv"], label="Scenario weather windows")
	plt.plot(compare["d"], compare["real_cv"], label="Real weather windows")
	plt.xlabel("Day")
	plt.ylabel("Coefficient of Variation (CV)")
	plt.title(
		f"CV of Weather Windows by Day ({TARGET_VESSEL}, wl_ids={locations_label()})"
	)
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig("figures/cv_weather_windows.svg", dpi=100, bbox_inches="tight")
	plt.close()

	plot_kindness_metric_distributions()
	plot_persistence_diagnostics(ww_threshold=4.0)

	plt.figure(figsize=(12, 5))
	bins_min = int(min(
		winter_kindness["n_good_winter_days"].min(),
		real_winter_kindness["n_good_winter_days"].min(),
	))
	bins_max = int(max(
		winter_kindness["n_good_winter_days"].max(),
		real_winter_kindness["n_good_winter_days"].max(),
	))
	bins = range(
		bins_min,
		bins_max + 2,
	)
	scenario_weights = [1.0 / len(winter_kindness)] * len(winter_kindness)
	real_weights = [1.0 / len(real_winter_kindness)] * len(real_winter_kindness)
	plt.hist(
		winter_kindness["n_good_winter_days"],
		bins=bins,
		weights=scenario_weights,
		edgecolor="black",
		alpha=0.6,
		label="Scenarioer",
	)
	plt.hist(
		real_winter_kindness["n_good_winter_days"],
		bins=bins,
		weights=real_weights,
		edgecolor="black",
		alpha=0.6,
		label="Ekte weather (år)",
	)
	plt.xlabel(f"Dager med weather window > {WW_KIND_THRESHOLD} (Oct-Mar)")
	plt.ylabel("Andel av egen gruppe")
	plt.title(
		f"Histogram av snill/slem-metrikken ({TARGET_VESSEL}, wl_ids={locations_label()})"
	)
	plt.grid(True, axis="y", alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig("figures/histogram_kindness_weather.svg", dpi=100, bbox_inches="tight")
	plt.close()

	combined_sorted = pd.concat(
		[
			winter_kindness[["n_good_winter_days"]].assign(source="Scenario"),
			real_winter_kindness[["year", "n_good_winter_days"]].assign(source="Ekte aar"),
		],
		ignore_index=True,
		sort=False,
	)
	combined_sorted = combined_sorted.sort_values(
		"n_good_winter_days", ascending=False
	).reset_index(drop=True)
	combined_sorted["rank"] = range(1, len(combined_sorted) + 1)

	plt.figure(figsize=(12, 5))
	plt.plot(
		combined_sorted["rank"],
		combined_sorted["n_good_winter_days"],
		color="lightgray",
		linewidth=1.2,
		label="Total sortert fordeling",
	)
	scen_mask = combined_sorted["source"] == "Scenario"
	real_mask = combined_sorted["source"] == "Ekte aar"
	plt.scatter(
		combined_sorted.loc[scen_mask, "rank"],
		combined_sorted.loc[scen_mask, "n_good_winter_days"],
		s=18,
		alpha=0.75,
		color="tab:blue",
		label="Scenarioer",
	)
	plt.scatter(
		combined_sorted.loc[real_mask, "rank"],
		combined_sorted.loc[real_mask, "n_good_winter_days"],
		s=38,
		alpha=0.95,
		color="tab:orange",
		label="Ekte weather (aar)",
	)
	plt.xlabel("Felles rank (snillest til venstre)")
	plt.ylabel(f"Dager med weather window > {WW_KIND_THRESHOLD} (Oct-Mar)")
	plt.title(
		f"Felles fordeling: scenarioer + ekte aar ({TARGET_VESSEL}, wl_ids={locations_label()})"
	)
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig("figures/combined_distribution_weather.svg", dpi=100, bbox_inches="tight")
	plt.close()


if __name__ == "__main__":
	main()

