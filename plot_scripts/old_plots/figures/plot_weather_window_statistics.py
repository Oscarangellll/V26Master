import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path

from data.fixed_data import data


TARGET_VESSEL = "CTV"
TARGET_WL_IDS = [3, 4]
MAX_SCENARIOS = 3000
RAW_WEATHER_MAX_SCENARIOS = 300
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


def build_scenario_monthly_longest_storm_mean(ww_threshold=4.0):
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

	daily = (
		df.groupby(["s", "d", "month"], as_index=False)
		.agg(
			n_locs=("wl_id", "nunique"),
			n_bad=("ww", lambda x: int((x < ww_threshold).sum())),
		)
	)
	daily = daily[daily["n_locs"] == len(TARGET_WL_IDS)].copy()
	daily["all_bad"] = daily["n_bad"] == len(TARGET_WL_IDS)

	rows = []
	for (s, month), group in daily.groupby(["s", "month"]):
		group = group.sort_values("d")
		longest = longest_consecutive_bad_days(
			group["d"].to_numpy(),
			group["all_bad"].to_numpy(),
		)
		rows.append({"s": int(s), "month": int(month), "longest_storm": float(longest)})

	if not rows:
		return pd.Series(dtype=float, name="scenario_longest_storm")

	monthly = pd.DataFrame(rows).groupby("month")["longest_storm"].mean()
	monthly = monthly.reindex(range(1, 13), fill_value=0.0)
	return monthly.rename("scenario_longest_storm")


def build_real_monthly_longest_storm_mean(ww_threshold=4.0):
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

	daily = (
		daily_ww.groupby(["year", "day_of_year", "month"], as_index=False)
		.agg(
			n_locs=("weather_location_id", "nunique"),
			n_bad=("ww", lambda x: int((x < ww_threshold).sum())),
		)
	)
	daily = daily[daily["n_locs"] == len(TARGET_WL_IDS)].copy()
	daily["all_bad"] = daily["n_bad"] == len(TARGET_WL_IDS)

	rows = []
	for (year, month), group in daily.groupby(["year", "month"]):
		group = group.sort_values("day_of_year")
		longest = longest_consecutive_bad_days(
			group["day_of_year"].to_numpy(),
			group["all_bad"].to_numpy(),
		)
		rows.append({"year": int(year), "month": int(month), "longest_storm": float(longest)})

	if not rows:
		return pd.Series(dtype=float, name="real_longest_storm")

	monthly = pd.DataFrame(rows).groupby("month")["longest_storm"].mean()
	monthly = monthly.reindex(range(1, 13), fill_value=0.0)
	return monthly.rename("real_longest_storm")


def plot_monthly_longest_storm(ww_threshold=4.0):
	scenario_monthly = build_scenario_monthly_longest_storm_mean(ww_threshold=ww_threshold)
	real_monthly = build_real_monthly_longest_storm_mean(ww_threshold=ww_threshold)

	months = np.arange(1, 13)
	labels = [pd.Timestamp(2011, m, 1).strftime("%b") for m in months]

	plt.figure(figsize=(12, 5))
	plt.plot(months, scenario_monthly.values, marker="o", linewidth=2, label="Scenarioer")
	plt.plot(months, real_monthly.values, marker="x", linewidth=2, label="Ekte weather (år)")
	plt.xticks(months, labels)
	plt.xlabel("Måned")
	plt.ylabel(f"Gjennomsnittlig lengste stormperiode (dager, WW < {ww_threshold})")
	plt.title(
		f"Månedlig gj.snitt av lengste stormperiode ({TARGET_VESSEL}, wl_ids={locations_label()})"
	)
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig("figures/monthly_longest_storm.svg", dpi=100, bbox_inches="tight")
	plt.close()


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


def _safe_relative_diff(simulated, observed):
	if pd.isna(observed) or observed == 0:
		return np.nan
	return 100.0 * (simulated - observed) / observed


def _fast_acf(x, max_lag=168):
	x = np.asarray(x, dtype=float)
	x = x[np.isfinite(x)]
	if x.size == 0:
		return np.full(max_lag + 1, np.nan)

	x = x - x.mean()
	var = np.dot(x, x)
	if var <= 0:
		out = np.zeros(max_lag + 1, dtype=float)
		out[0] = 1.0
		return out

	n = len(x)
	nfft = 1 << (2 * n - 1).bit_length()
	f = np.fft.rfft(x, n=nfft)
	acf_full = np.fft.irfft(f * np.conjugate(f), n=nfft)[:n]
	acf = acf_full[: max_lag + 1] / var
	acf[0] = 1.0
	return acf


def _month_name(m):
	return pd.Timestamp(2011, int(m), 1).strftime("%b")


def load_real_weather_raw():
	df = pd.read_csv("data/weather/weather.csv")
	df = df[df["weather_location_id"].isin(TARGET_WL_IDS)].copy()
	df["time"] = pd.to_datetime(df["time"])
	df["month"] = df["time"].dt.month
	df = df.rename(columns={"weather_location_id": "wl_id"})
	return df[["time", "wl_id", "month", "speed", "height"]]


def load_scenario_weather_raw():
	df = pd.read_parquet(
		SCENARIO_DATA_DIR / "weather",
		columns=["s", "wl_id", "d", "hour", "speed", "height"],
	)
	df = df.dropna(subset=["s", "wl_id", "d", "hour", "speed", "height"]).copy()
	df["s"] = df["s"].astype(int)
	df["wl_id"] = df["wl_id"].astype(int)
	df["d"] = df["d"].astype(int)
	df["hour"] = df["hour"].astype(int)
	df["speed"] = df["speed"].astype(float)
	df["height"] = df["height"].astype(float)
	df = df[df["wl_id"].isin(TARGET_WL_IDS)].copy()

	first_ids = sorted(df["s"].unique())[: min(MAX_SCENARIOS, RAW_WEATHER_MAX_SCENARIOS)]
	df = df[df["s"].isin(first_ids)].copy()
	df["month"] = df["d"].map(day_to_month)
	return df


def build_monthly_summary_tables(real_weather, scenario_weather):
	months = pd.Index(range(1, 13), name="month")

	tables = {}
	for var in ["speed", "height"]:
		real_mean = real_weather.groupby("month")[var].mean().reindex(months)
		real_std = real_weather.groupby("month")[var].std().reindex(months)

		sim_mean = scenario_weather.groupby("month")[var].mean().reindex(months)
		sim_std = scenario_weather.groupby("month")[var].std().reindex(months)

		table = pd.DataFrame(
			{
				("mean", "observed"): real_mean,
				("mean", "simulated"): sim_mean,
				("mean", "relative_difference_pct"): [_safe_relative_diff(s, o) for s, o in zip(sim_mean, real_mean)],
				("std", "observed"): real_std,
				("std", "simulated"): sim_std,
				("std", "relative_difference_pct"): [_safe_relative_diff(s, o) for s, o in zip(sim_std, real_std)],
			}
		)
		table.index = [_month_name(m) for m in table.index]
		table.index.name = "month"
		tables[var] = table

	return tables


def build_real_weather_windows_daily():
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
	daily_ww = daily_ww.rename(columns={"weather_location_id": "wl_id"})
	return daily_ww


def plot_weather_validation_suite(ww_threshold_good=8.0, ww_threshold_bad=4.0, acf_max_lag=168):
	real_weather = load_real_weather_raw()
	scenario_weather = load_scenario_weather_raw()

	# 1) Empirical distributions of wind speed and wave height.
	fig, axes = plt.subplots(1, 2, figsize=(14, 5))
	for ax, var, title in zip(
		axes,
		["speed", "height"],
		["Empirical distribution: wind speed", "Empirical distribution: wave height"],
	):
		v_real = real_weather[var].to_numpy()
		v_sim = scenario_weather[var].to_numpy()
		vmin = float(min(np.nanmin(v_real), np.nanmin(v_sim)))
		vmax = float(max(np.nanmax(v_real), np.nanmax(v_sim)))
		bins = np.linspace(vmin, vmax, 60)

		ax.hist(v_real, bins=bins, density=True, alpha=0.45, label="Observed", edgecolor="none")
		ax.hist(v_sim, bins=bins, density=True, alpha=0.45, label="Simulated", edgecolor="none")
		ax.set_title(title)
		ax.set_xlabel(var)
		ax.set_ylabel("Density")
		ax.grid(True, alpha=0.25)

	axes[0].legend()
	fig.suptitle(f"Observed vs simulated raw weather ({TARGET_VESSEL}, wl_ids={locations_label()})", y=0.98)
	fig.tight_layout(rect=[0, 0, 1, 0.95])
	plt.savefig("figures/weather_empirical_distributions.svg", dpi=100, bbox_inches="tight")
	plt.close()

	# 2) Monthly mean/std summary tables with relative differences.
	tables = build_monthly_summary_tables(real_weather, scenario_weather)
	Path("results").mkdir(parents=True, exist_ok=True)
	for var, table in tables.items():
		table.to_csv(f"results/monthly_{var}_summary.csv")
		print(f"\n=== Monthly summary ({var}) ===")
		print(table.round(3).to_string())

	# 3) ACF comparison for speed and height.
	real_weather_sorted = real_weather.sort_values("time")
	scenario_weather_sorted = scenario_weather.sort_values(["s", "d", "hour"])

	acf_real_speed = _fast_acf(real_weather_sorted["speed"].to_numpy(), max_lag=acf_max_lag)
	acf_sim_speed = _fast_acf(scenario_weather_sorted["speed"].to_numpy(), max_lag=acf_max_lag)
	acf_real_height = _fast_acf(real_weather_sorted["height"].to_numpy(), max_lag=acf_max_lag)
	acf_sim_height = _fast_acf(scenario_weather_sorted["height"].to_numpy(), max_lag=acf_max_lag)

	lags = np.arange(acf_max_lag + 1)
	fig, axes = plt.subplots(1, 2, figsize=(14, 5))
	for ax, r, s, title in zip(
		axes,
		[acf_real_speed, acf_real_height],
		[acf_sim_speed, acf_sim_height],
		["ACF speed", "ACF height"],
	):
		ax.plot(lags, r, linewidth=2, label="Observed")
		ax.plot(lags, s, linewidth=2, label="Simulated")
		ax.set_title(title)
		ax.set_xlabel("Lag (hours)")
		ax.set_ylabel("ACF")
		ax.set_ylim(-0.1, 1.05)
		ax.grid(True, alpha=0.25)

	axes[0].legend()
	fig.suptitle(f"ACF comparison ({TARGET_VESSEL}, wl_ids={locations_label()})", y=0.98)
	fig.tight_layout(rect=[0, 0, 1, 0.95])
	plt.savefig("figures/weather_acf_comparison.svg", dpi=100, bbox_inches="tight")
	plt.close()

	# 4) and 5) Mean number of days with WW >= 8 and WW < 4.
	scen_ww = pd.read_parquet(
		SCENARIO_DATA_DIR / "weather_windows",
		columns=["h", "wl_id", "s", "d", "ww"],
	)
	scen_ww = scen_ww.dropna(subset=["h", "wl_id", "s", "d", "ww"]).copy()
	scen_ww["h"] = scen_ww["h"].astype(str)
	scen_ww["wl_id"] = scen_ww["wl_id"].astype(int)
	scen_ww["s"] = scen_ww["s"].astype(int)
	scen_ww["d"] = scen_ww["d"].astype(int)
	scen_ww["ww"] = scen_ww["ww"].astype(float)
	scen_ww = keep_first_scenarios(scen_ww)
	scen_ww = scen_ww[(scen_ww["h"] == TARGET_VESSEL) & (scen_ww["wl_id"].isin(TARGET_WL_IDS))].copy()

	obs_ww_daily = build_real_weather_windows_daily()
	obs_ww_daily = obs_ww_daily[obs_ww_daily["wl_id"].isin(TARGET_WL_IDS)].copy()

	scen_good = (
		scen_ww[scen_ww["ww"] >= ww_threshold_good]
		.groupby(["s", "wl_id"])
		.size()
		.groupby("s")
		.sum()
	)
	obs_good = (
		obs_ww_daily[obs_ww_daily["ww"] >= ww_threshold_good]
		.groupby(["year", "wl_id"])
		.size()
		.groupby("year")
		.sum()
	)

	scen_bad = (
		scen_ww[scen_ww["ww"] < ww_threshold_bad]
		.groupby(["s", "wl_id"])
		.size()
		.groupby("s")
		.sum()
	)
	obs_bad = (
		obs_ww_daily[obs_ww_daily["ww"] < ww_threshold_bad]
		.groupby(["year", "wl_id"])
		.size()
		.groupby("year")
		.sum()
	)

	good_vals = [obs_good.mean() if len(obs_good) else 0.0, scen_good.mean() if len(scen_good) else 0.0]
	bad_vals = [obs_bad.mean() if len(obs_bad) else 0.0, scen_bad.mean() if len(scen_bad) else 0.0]

	fig, axes = plt.subplots(1, 2, figsize=(12, 5))
	axes[0].bar(["Observed", "Simulated"], good_vals, color=["tab:orange", "tab:blue"], alpha=0.8)
	axes[0].set_title(f"Mean #days with WW >= {ww_threshold_good}")
	axes[0].set_ylabel("Days (location-days)")
	axes[0].grid(True, axis="y", alpha=0.25)

	axes[1].bar(["Observed", "Simulated"], bad_vals, color=["tab:orange", "tab:blue"], alpha=0.8)
	axes[1].set_title(f"Mean #days with WW < {ww_threshold_bad}")
	axes[1].set_ylabel("Days (location-days)")
	axes[1].grid(True, axis="y", alpha=0.25)

	fig.suptitle(f"Weather-window threshold counts ({TARGET_VESSEL}, wl_ids={locations_label()})", y=0.98)
	fig.tight_layout(rect=[0, 0, 1, 0.95])
	plt.savefig("figures/weather_window_threshold_counts.svg", dpi=100, bbox_inches="tight")
	plt.close()

	# 6) Distribution of weather-window durations.
	obs_durations = obs_ww_daily["ww"].to_numpy()
	scen_durations = scen_ww["ww"].to_numpy()
	vmin = int(min(np.nanmin(obs_durations), np.nanmin(scen_durations)))
	vmax = int(max(np.nanmax(obs_durations), np.nanmax(scen_durations)))
	bins = np.arange(vmin, vmax + 2)

	plt.figure(figsize=(10, 5))
	plt.hist(obs_durations, bins=bins, density=True, alpha=0.5, label="Observed", edgecolor="black")
	plt.hist(scen_durations, bins=bins, density=True, alpha=0.5, label="Simulated", edgecolor="black")
	plt.xlabel("Weather window duration (hours)")
	plt.ylabel("Density")
	plt.title(f"Distribution of weather-window durations ({TARGET_VESSEL}, wl_ids={locations_label()})")
	plt.grid(True, axis="y", alpha=0.25)
	plt.legend()
	plt.tight_layout()
	plt.savefig("figures/weather_window_duration_distribution.svg", dpi=100, bbox_inches="tight")
	plt.close()


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
	plot_monthly_longest_storm(ww_threshold=4.0)
	plot_weather_validation_suite(ww_threshold_good=8.0, ww_threshold_bad=4.0, acf_max_lag=168)

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

