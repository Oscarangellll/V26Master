"""
Stratified scenario sampling based on a kindness/harshness ranking axis.

Workflow:
1) Build one score per scenario ("kindness axis").
2) Split ranked scenarios into three bins: kind, normal, harsh.
3) Draw bin first using configured bin probabilities.
4) Draw scenario uniformly from the selected bin.
5) Assign scenario weights using selected bins and bin probabilities.
"""

from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np


BIN_KIND = "kind"
BIN_NORMAL = "normal"
BIN_HARSH = "harsh"
BIN_ORDER = (BIN_KIND, BIN_NORMAL, BIN_HARSH)
DEFAULT_WINTER_PERIODS = ("Oct", "Nov", "Dec", "Jan", "Feb", "Mar")


def _normalize_bin_probabilities(bin_probabilities: Dict[str, float]) -> Dict[str, float]:
    probs = {k: float(bin_probabilities.get(k, 0.0)) for k in BIN_ORDER}
    if any(v < 0 for v in probs.values()):
        raise ValueError("Bin probabilities must be non-negative.")

    total = sum(probs.values())
    if total <= 0:
        raise ValueError("At least one bin probability must be positive.")

    return {k: v / total for k, v in probs.items()}


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


def split_into_bins(
    scores: Dict[int, float],
    tail_fraction: float,
) -> Dict[str, List[int]]:
    """
    Split scenarios into kind/normal/harsh bins by ranking scores descending.

    tail_fraction is y in [0, 0.5].
    - top y% => kind
    - bottom y% => harsh
    - middle => normal
    """
    if not (0.0 <= tail_fraction <= 0.5):
        raise ValueError("tail_fraction must be in [0.0, 0.5].")

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    n = len(ranked)
    if n == 0:
        raise ValueError("Cannot split empty score set.")

    tail_n = int(round(n * tail_fraction))
    if tail_fraction > 0 and tail_n == 0:
        tail_n = 1

    if 2 * tail_n >= n:
        tail_n = max(0, (n - 1) // 2)

    kind_ids = [sid for sid, _ in ranked[:tail_n]]
    harsh_ids = [sid for sid, _ in ranked[n - tail_n :]] if tail_n > 0 else []
    normal_ids = [sid for sid, _ in ranked[tail_n : n - tail_n]]

    return {
        BIN_KIND: kind_ids,
        BIN_NORMAL: normal_ids,
        BIN_HARSH: harsh_ids,
    }


def sample_from_bins(
    rng: np.random.Generator,
    bins: Dict[str, Sequence[int]],
    n_samples: int,
    bin_probabilities: Dict[str, float],
) -> Tuple[List[int], List[str]]:
    """
    Draw scenarios without replacement, choosing bin first for each draw.
    """
    probs = _normalize_bin_probabilities(bin_probabilities)

    available = {
        b: [int(s) for s in bins.get(b, [])]
        for b in BIN_ORDER
    }

    total_available = sum(len(v) for v in available.values())
    if n_samples > total_available:
        raise ValueError(
            f"Requested {n_samples} samples, but only {total_available} scenarios are available."
        )

    selected_ids: List[int] = []
    selected_bins: List[str] = []

    for _ in range(n_samples):
        eligible_bins = [b for b in BIN_ORDER if len(available[b]) > 0]
        eligible_probs = np.array([probs[b] for b in eligible_bins], dtype=float)
        eligible_probs = eligible_probs / eligible_probs.sum()

        chosen_bin = str(rng.choice(eligible_bins, p=eligible_probs))

        idx = int(rng.integers(low=0, high=len(available[chosen_bin])))
        scenario_id = available[chosen_bin].pop(idx)

        selected_ids.append(int(scenario_id))
        selected_bins.append(chosen_bin)

    return selected_ids, selected_bins


def compute_weights_from_selected_bins(
    selected_ids: Sequence[int],
    selected_bins: Sequence[str],
    bin_probabilities: Dict[str, float],
) -> Dict[int, float]:
    """
    Weight formula following requested logic:

    1) Let B_sel be bins represented among selected scenarios.
    2) Normalize bin probabilities over B_sel only.
    3) Split each selected bin mass equally among selected scenarios in that bin.

    Example:
      p(kind)=0.2, p(normal)=0.6, p(harsh)=0.2
      selected bins: kind + harsh
      normalized masses: kind=0.2/(0.2+0.2)=0.5, harsh=0.5
    """
    if len(selected_ids) != len(selected_bins):
        raise ValueError("selected_ids and selected_bins must have the same length.")

    probs = _normalize_bin_probabilities(bin_probabilities)
    represented_bins = sorted(set(selected_bins))
    if not represented_bins:
        return {}

    denom = sum(probs[b] for b in represented_bins)
    if denom <= 0:
        raise ValueError("Selected bins have zero combined probability.")

    bin_mass = {b: probs[b] / denom for b in represented_bins}
    counts = Counter(selected_bins)

    weights: Dict[int, float] = {}
    for sid, b in zip(selected_ids, selected_bins):
        weights[int(sid)] = float(bin_mass[b] / counts[b])

    return weights


def sample_stratified_scenarios(
    rng: np.random.Generator,
    case,
    scenario_ids: Sequence[int],
    weather_windows,
    n_samples: int,
    tail_fraction: float,
    bin_probabilities: Dict[str, float],
    metric: str = "count_location_days_over_threshold",
    ww_threshold: float = 8.0,
    vessel_name: str | None = None,
    winter_periods: Sequence[str] = DEFAULT_WINTER_PERIODS,
):
    """
    End-to-end stratified sampling.

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

    bins = split_into_bins(scores=scores, tail_fraction=tail_fraction)
    selected_ids, selected_bins = sample_from_bins(
        rng=rng,
        bins=bins,
        n_samples=n_samples,
        bin_probabilities=bin_probabilities,
    )

    weights = compute_weights_from_selected_bins(
        selected_ids=selected_ids,
        selected_bins=selected_bins,
        bin_probabilities=bin_probabilities,
    )

    details = {
        "scores": scores,
        "bins": bins,
        "selected_bins": selected_bins,
        "bin_counts": dict(Counter(selected_bins)),
    }

    return selected_ids, weights, details
