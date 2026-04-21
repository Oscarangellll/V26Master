"""
Rank-based stratified scenario sampling.

Workflow:
1) Build one score per scenario ("kindness axis").
2) Sort scenarios by score (kind -> harsh).
3) For tree size n, split ranked scenarios into n near-equal contiguous bins.
4) Draw exactly one scenario uniformly from each bin.
5) Assign equal scenario weights (1 / n).
"""

from collections import Counter
from typing import Dict, List, Sequence

import numpy as np


DEFAULT_WINTER_PERIODS = ("Oct", "Nov", "Dec", "Jan", "Feb", "Mar")


def _center_out_indices(n: int) -> List[int]:
    if n <= 0:
        return []

    if n % 2 == 1:
        center = n // 2
        order = [center]
        step = 1
        while len(order) < n:
            left = center - step
            right = center + step
            if left >= 0:
                order.append(left)
            if right < n:
                order.append(right)
            step += 1
        return order

    left_center = (n // 2) - 1
    right_center = n // 2
    order = [left_center, right_center]
    step = 1
    while len(order) < n:
        left = left_center - step
        right = right_center + step
        if left >= 0:
            order.append(left)
        if right < n:
            order.append(right)
        step += 1
    return order


def _near_equal_bin_sizes(total: int, n_bins: int) -> List[int]:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    if total < n_bins:
        raise ValueError(
            f"Cannot split {total} scenarios into {n_bins} non-empty bins."
        )

    base = total // n_bins
    remainder = total % n_bins
    sizes = [base] * n_bins

    for idx in _center_out_indices(n_bins)[:remainder]:
        sizes[idx] += 1

    return sizes


def split_ranked_into_n_bins(scores: Dict[int, float], n_bins: int) -> List[List[int]]:
    ranked_ids = [sid for sid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
    n = len(ranked_ids)
    if n == 0:
        raise ValueError("Cannot split empty score set.")

    sizes = _near_equal_bin_sizes(total=n, n_bins=n_bins)

    bins: List[List[int]] = []
    start = 0
    for size in sizes:
        end = start + size
        bins.append([int(s) for s in ranked_ids[start:end]])
        start = end

    return bins


def _winter_days(case, winter_periods: Sequence[str]) -> List[int]:
    days: List[int] = []
    for p in winter_periods:
        if p in case.D_t:
            days.extend(case.D_t[p])

    if not days:
        raise ValueError(
            "No winter periods found in case.D_t for requested periods: "
            f"{list(winter_periods)}"
        )

    return sorted(set(int(d) for d in days))


def build_kindness_scores(
    case,
    scenario_ids: Sequence[int],
    weather_windows,
    metric: str = "count_location_days_over_threshold",
    ww_threshold: float = 8.0,
    vessel_name: str | None = None,
    winter_periods: Sequence[str] = DEFAULT_WINTER_PERIODS,
) -> Dict[int, float]:
    """
    Build a scalar kindness score per scenario.

    Supported metrics:
    - count_location_days_over_threshold:
        Count of (location, day) with ww >= threshold in winter periods.
        Higher score => kinder scenario.
    - total_window_hours:
        Sum of ww across (location, day) in winter periods.
        Higher score => kinder scenario.
    - count_location_days_under_threshold:
        Count of (location, day) with ww < threshold in winter periods.
        Higher count means harsher scenario, so score = -count.
        Higher score => kinder scenario.
    - max_bad_streak_under_threshold:
        Longest consecutive streak of days where system-wide total ww is below
        threshold * n_locations in winter periods.
        Longer streak means harsher scenario, so score = -streak.
        Higher score => kinder scenario.
    """
    if vessel_name is None:
        vessel_name = case.vessel_types[0].name

    location_ids = sorted({w.weather_location_id for w in case.wind_farms})
    winter_days = _winter_days(case, winter_periods)

    scores: Dict[int, float] = {}
    for s in scenario_ids:
        scenario_map = weather_windows[s]

        if metric == "count_location_days_over_threshold":
            score = 0.0
            for wl_id in location_ids:
                for d in winter_days:
                    ww = float(scenario_map.get((vessel_name, wl_id, d), 0.0))
                    if ww >= ww_threshold:
                        score += 1.0
        elif metric == "total_window_hours":
            score = 0.0
            for wl_id in location_ids:
                for d in winter_days:
                    ww = float(scenario_map.get((vessel_name, wl_id, d), 0.0))
                    score += ww
        elif metric == "count_location_days_under_threshold":
            bad_count = 0.0
            for wl_id in location_ids:
                for d in winter_days:
                    ww = float(scenario_map.get((vessel_name, wl_id, d), 0.0))
                    if ww < ww_threshold:
                        bad_count += 1.0
            score = -bad_count
        elif metric == "max_bad_streak_under_threshold":
            # System-wide bad day: aggregate ww across selected locations is low.
            bad_day_limit = ww_threshold * max(1, len(location_ids))
            day_totals = []
            for d in winter_days:
                total_ww = 0.0
                for wl_id in location_ids:
                    total_ww += float(scenario_map.get((vessel_name, wl_id, d), 0.0))
                day_totals.append(total_ww)

            max_streak = 0
            current = 0
            for total_ww in day_totals:
                if total_ww < bad_day_limit:
                    current += 1
                    if current > max_streak:
                        max_streak = current
                else:
                    current = 0

            score = -float(max_streak)
        else:
            raise ValueError(
                "Unsupported metric. Use 'count_location_days_over_threshold' "
                "or 'total_window_hours' or 'count_location_days_under_threshold' "
                "or 'max_bad_streak_under_threshold'."
            )

        scores[int(s)] = float(score)

    return scores


def sample_stratified_scenarios(
    rng: np.random.Generator,
    case,
    scenario_ids: Sequence[int],
    weather_windows,
    n_samples: int,
    metric: str = "count_location_days_over_threshold",
    ww_threshold: float = 8.0,
    vessel_name: str | None = None,
    winter_periods: Sequence[str] = DEFAULT_WINTER_PERIODS,
):
    """
    End-to-end rank-bin stratified sampling.

    Returns:
      selected_ids, weights, details
    where details includes scores, bins, and selected bin labels.
    """
    scores = build_kindness_scores(
        case=case,
        scenario_ids=scenario_ids,
        weather_windows=weather_windows,
        metric=metric,
        ww_threshold=ww_threshold,
        vessel_name=vessel_name,
        winter_periods=winter_periods,
    )

    rank_bins = split_ranked_into_n_bins(scores=scores, n_bins=n_samples)

    selected_ids: List[int] = []
    selected_bins: List[str] = []
    for idx, bin_ids in enumerate(rank_bins):
        chosen = int(rng.choice(bin_ids))
        selected_ids.append(chosen)
        selected_bins.append(f"bin_{idx + 1}")

    equal_weight = 1.0 / float(n_samples)
    weights = {int(s): equal_weight for s in selected_ids}

    details = {
        "scores": scores,
        "bins": {f"bin_{i + 1}": b for i, b in enumerate(rank_bins)},
        "selected_bins": selected_bins,
        "bin_counts": dict(Counter(selected_bins)),
    }

    return selected_ids, weights, details
