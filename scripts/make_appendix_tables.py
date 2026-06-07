from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_scripts.plot_case_studies import (  # noqa: E402
    _add_vessel_mix_metrics,
    _all_subsets,
    _charter_summary,
    _coalition_key,
    _display_coalition,
    _has_nonempty_core,
    _members,
    _parse_solution_group,
)
from plot_scripts.plot_stability_cv import compute_iss_cv, compute_oss_cv  # noqa: E402
from plot_scripts.plot_stratified_comparison import FILES as STRATIFIED_FILES  # noqa: E402
from plot_scripts.plot_stratified_comparison import compute_weighted_avg  # noqa: E402
from data.fixed_data import data  # noqa: E402


CASES = ["1W1B", "2W2B", "3W2B", "4W3B"]
TREE_SIZES = [1, 3, 5, 7, 10, 15, 20]
RESULTS_DIR = ROOT / "results"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "appendix_tables"
FALLBACK_WIND_FARMS = {
    "G": {"lat": 53.3, "lon": 1.37, "n_turbines": 60},
    "H": {"lat": 53.99, "lon": 0.48, "n_turbines": 80},
}


def _round_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if col.endswith("_percent"):
            df[col] = df[col].round(2)
        elif col.endswith("_meur"):
            df[col] = df[col].round(3)
        elif col.endswith("_km"):
            df[col] = df[col].round(1)
        elif "share" in col:
            df[col] = df[col].round(3)
        elif "runtime" in col:
            df[col] = df[col].round(2)
        else:
            df[col] = df[col].round(3)
    return df


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _round_frame(df).to_csv(path, index=False)
    print(f"Wrote {path.relative_to(ROOT)}")


def _weighted_objective(path: Path, min_count: int = 10) -> pd.DataFrame:
    df = pd.read_csv(path)[["tree_size", "count", "objective"]]
    rows = []
    for tree_size, group in df.groupby("tree_size"):
        count = group["count"].sum()
        if count < min_count:
            continue
        avg = (group["objective"] * group["count"]).sum() / count
        rows.append(
            {
                "tree_size": tree_size,
                "count": count,
                "objective_meur": avg / 1e6,
            }
        )
    return pd.DataFrame(rows).sort_values("tree_size")


def _runtime_average(path: Path, column: str, min_count: int = 19) -> pd.DataFrame:
    df = pd.read_csv(path)[["tree_size", column]]
    rows = []
    for tree_size, group in df.groupby("tree_size"):
        valid = group[column].dropna()
        if len(valid) < min_count:
            continue
        rows.append(
            {
                "tree_size": tree_size,
                "count": len(valid),
                "runtime_seconds": valid.mean(),
                "runtime_minutes": valid.mean() / 60,
            }
        )
    return pd.DataFrame(rows).sort_values("tree_size")


def build_sampling_table() -> pd.DataFrame:
    rows = []
    for method, path in STRATIFIED_FILES.items():
        df = compute_weighted_avg(path)
        for row in df.to_dict("records"):
            rows.append(
                {
                    "sampling_method": method,
                    "tree_size": row["tree_size"],
                    "count": row["count"],
                    "oss_objective_meur": row["weighted_avg"] / 1e6,
                }
            )
    return pd.DataFrame(rows).sort_values(["sampling_method", "tree_size"])


def build_consensus_objective_table() -> pd.DataFrame:
    rows = []
    for case in CASES:
        mip = _weighted_objective(RESULTS_DIR / "stability" / case / "mip" / "OSS.csv")
        con = _weighted_objective(RESULTS_DIR / "stability" / case / "con_mp" / "OSS.csv")
        mip = mip.rename(columns={"objective_meur": "mip_objective_meur", "count": "mip_count"})
        con = con.rename(columns={"objective_meur": "consensus_objective_meur", "count": "consensus_count"})
        merged = mip.merge(con, on="tree_size", how="outer")
        for row in merged.to_dict("records"):
            mip_obj = row.get("mip_objective_meur")
            con_obj = row.get("consensus_objective_meur")
            rel_gap = np.nan
            if pd.notna(mip_obj) and mip_obj:
                rel_gap = 100 * (con_obj - mip_obj) / mip_obj
            rows.append(
                {
                    "instance": case,
                    "tree_size": row["tree_size"],
                    "mip_count": row.get("mip_count"),
                    "consensus_count": row.get("consensus_count"),
                    "mip_oss_objective_meur": mip_obj,
                    "consensus_oss_objective_meur": con_obj,
                    "relative_difference_percent": rel_gap,
                }
            )
    return pd.DataFrame(rows).sort_values(["instance", "tree_size"])


def build_consensus_runtime_table() -> pd.DataFrame:
    rows = []
    for case in CASES:
        for method, folder, column in [
            ("Direct MIP", "mip", "MIP_runtime"),
            ("Consensus heuristic", "con_mp", "Con_total runtime"),
        ]:
            path = RESULTS_DIR / "stability" / case / folder / "ISS.csv"
            if not path.exists():
                continue
            df = _runtime_average(path, column)
            for row in df.to_dict("records"):
                rows.append(
                    {
                        "instance": case,
                        "method": method,
                        **row,
                    }
                )
    return pd.DataFrame(rows).sort_values(["instance", "method", "tree_size"])


def build_stability_table() -> pd.DataFrame:
    rows = []
    for case in CASES:
        iss_path = RESULTS_DIR / "stability" / case / "con_mp" / "ISS.csv"
        oss_path = RESULTS_DIR / "stability" / case / "con_mp" / "OSS.csv"
        iss = pd.read_csv(iss_path)[["tree_size", "objective"]]
        iss_rows = []
        for tree_size, group in iss.groupby("tree_size"):
            valid = group["objective"].dropna()
            if len(valid) >= 19:
                iss_rows.append(
                    {
                        "tree_size": tree_size,
                        "iss_count": len(valid),
                        "iss_objective_meur": valid.mean() / 1e6,
                    }
                )
        iss_df = pd.DataFrame(iss_rows)

        oss_df = _weighted_objective(oss_path, min_count=19).rename(
            columns={"count": "oss_count", "objective_meur": "oss_objective_meur"}
        )
        iss_cv = compute_iss_cv(iss_path).rename(columns={"cv": "iss_cv"})
        oss_cv = compute_oss_cv(oss_path).rename(columns={"cv": "oss_cv"})

        merged = iss_df.merge(oss_df, on="tree_size", how="outer")
        merged = merged.merge(iss_cv, on="tree_size", how="outer")
        merged = merged.merge(oss_cv, on="tree_size", how="outer")
        for row in merged.to_dict("records"):
            rows.append(
                {
                    "instance": case,
                    "tree_size": row["tree_size"],
                    "iss_count": row.get("iss_count"),
                    "oss_count": row.get("oss_count"),
                    "iss_objective_meur": row.get("iss_objective_meur"),
                    "oss_objective_meur": row.get("oss_objective_meur"),
                    "iss_cv_percent": row.get("iss_cv") * 100 if pd.notna(row.get("iss_cv")) else np.nan,
                    "oss_cv_percent": row.get("oss_cv") * 100 if pd.notna(row.get("oss_cv")) else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["instance", "tree_size"])


def _normalize(series: pd.Series) -> pd.Series:
    minimum = series.min(skipna=True)
    maximum = series.max(skipna=True)
    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series(0.5, index=series.index)
    return (series - minimum) / (maximum - minimum)


def _partitions(items: tuple[str, ...]):
    items = tuple(items)
    if not items:
        yield ()
        return

    first, rest = items[0], items[1:]
    for partition in _partitions(rest):
        yield ((first,),) + partition
        for idx, block in enumerate(partition):
            new_block = tuple(sorted(block + (first,)))
            yield partition[:idx] + (new_block,) + partition[idx + 1 :]


def _canonical_partition(partition) -> tuple[str, ...]:
    return tuple(sorted((_coalition_key(block) for block in partition), key=lambda x: (len(x), x)))


def _best_partition(coalition: str, cost: dict[str, float]) -> str:
    members = _members(coalition)
    best_config = None
    best_cost = np.inf
    seen = set()

    for partition in _partitions(members):
        config = _canonical_partition(partition)
        if config in seen or any(block not in cost for block in config):
            continue
        seen.add(config)
        total_cost = sum(cost[block] for block in config)
        if total_cost < best_cost:
            best_cost = total_cost
            best_config = config

    if best_config is None:
        return ""

    return " + ".join(_display_coalition(block) for block in best_config)


def _wind_farm_locations() -> dict[str, tuple[float, float]]:
    locations = {w.name: (w.lat, w.lon) for w in data.wind_farms}
    for name, values in FALLBACK_WIND_FARMS.items():
        locations.setdefault(name, (values["lat"], values["lon"]))
    return locations


def _base_locations() -> dict[str, tuple[float, float]]:
    return {b.name: (b.lat, b.lon) for b in data.bases}


def _base_display_map() -> dict[str, str]:
    base_ids = sorted(_base_locations(), key=lambda value: int(value) if str(value).isdigit() else str(value))
    return {base_id: str(idx) for idx, base_id in enumerate(base_ids, start=1)}


def _wind_farm_turbines() -> dict[str, float]:
    turbines = {w.name: w.n_turbines for w in data.wind_farms}
    for name, values in FALLBACK_WIND_FARMS.items():
        turbines.setdefault(name, values["n_turbines"])
    return turbines


def _distance_features(coalition: str) -> dict[str, float]:
    from haversine import Unit, haversine

    locations = _wind_farm_locations()
    members = _members(coalition)
    if any(member not in locations for member in members):
        return {"avg_pairwise_distance": np.nan, "max_pairwise_distance": np.nan}

    distances = [
        haversine(locations[a], locations[b], unit=Unit.KILOMETERS)
        for a, b in combinations(members, 2)
    ]
    if not distances:
        return {"avg_pairwise_distance": 0.0, "max_pairwise_distance": 0.0}
    return {
        "avg_pairwise_distance": float(np.mean(distances)),
        "max_pairwise_distance": float(np.max(distances)),
    }


def _turbine_features(coalition: str) -> dict[str, float]:
    turbines = _wind_farm_turbines()
    values = [turbines.get(member, np.nan) for member in _members(coalition)]
    if any(pd.isna(value) for value in values):
        return {"total_turbines": np.nan, "turbine_range": np.nan}
    return {
        "total_turbines": float(np.sum(values)),
        "turbine_range": float(np.max(values) - np.min(values)),
    }


def _selected_bases_from_row(row: dict) -> list[str]:
    selected = [
        str(key[0])
        for key, value in _parse_solution_group(row.get("eta", ""))
        if value > 0.5
    ]
    if selected:
        return selected

    bases = row.get("bases", "")
    if isinstance(bases, str) and bases.strip():
        return [base.strip() for base in bases.split(";") if base.strip()]
    return []


def _display_bases(base_ids: list[str]) -> str:
    display_map = _base_display_map()
    return ";".join(display_map.get(str(base_id), str(base_id)) for base_id in base_ids)


def _base_distance_features(row: dict) -> dict[str, float]:
    from haversine import Unit, haversine

    wind_farms = _wind_farm_locations()
    bases = _base_locations()
    selected_bases = [base for base in _selected_bases_from_row(row) if base in bases]
    members = _members(row.get("coalition", ""))

    if not selected_bases or any(member not in wind_farms for member in members):
        return {
            "avg_distance_to_selected_base": np.nan,
            "max_distance_to_selected_base": np.nan,
            "range_distance_to_selected_base": np.nan,
        }

    nearest_distances = [
        min(
            haversine(wind_farms[member], bases[base], unit=Unit.KILOMETERS)
            for base in selected_bases
        )
        for member in members
    ]
    return {
        "avg_distance_to_selected_base": float(np.mean(nearest_distances)),
        "max_distance_to_selected_base": float(np.max(nearest_distances)),
        "range_distance_to_selected_base": float(np.max(nearest_distances) - np.min(nearest_distances)),
    }


def _load_coalition_metrics_for_tables(path: Path) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    df = pd.read_csv(path)
    df["coalition"] = df["coalition"].map(_coalition_key)

    numeric_cols = [
        "objective",
        "first_stage_cost",
        "second_stage_cost",
        "charter_cost_ST",
        "charter_cost_LT",
        "charter_cost_mob",
        "downtime_cost",
        "travel_cost_S",
        "travel_cost_M",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.sort_values(["coalition", "objective"])
        .drop_duplicates(subset=["coalition"], keep="first")
        .reset_index(drop=True)
    )

    cost = dict(zip(df["coalition"], df["objective"]))
    standalone = {
        coalition: sum(cost[_coalition_key(w)] for w in _members(coalition))
        for coalition in df["coalition"]
        if all(_coalition_key(w) in cost for w in _members(coalition))
    }
    savings = {
        coalition: standalone[coalition] - cost[coalition]
        for coalition in standalone
    }
    synergy = {
        coalition: savings[coalition] / cost[coalition]
        for coalition in savings
        if cost[coalition]
    }

    df["standalone_cost"] = df["coalition"].map(standalone)
    df["savings"] = df["coalition"].map(savings)
    df["synergy"] = df["coalition"].map(synergy)

    charter_df = pd.DataFrame([_charter_summary(row) for row in df.to_dict("records")])
    df = pd.concat([df, charter_df], axis=1)
    df = _add_vessel_mix_metrics(df)

    df = pd.concat([df, pd.DataFrame([_distance_features(c) for c in df["coalition"]])], axis=1)
    df = pd.concat([df, pd.DataFrame([_base_distance_features(row) for row in df.to_dict("records")])], axis=1)
    df = pd.concat([df, pd.DataFrame([_turbine_features(c) for c in df["coalition"]])], axis=1)
    df["coalition_size"] = df["coalition"].str.len()
    df["has_stable_core"] = df["coalition"].map(
        lambda coalition: _has_nonempty_core(_members(coalition), savings)
    )

    return df, cost, savings


def _minmax_allocation_fallback(
    players: tuple[str, ...],
    values: dict[str, float],
    costs: dict[str, float],
) -> dict | None:
    grand = _coalition_key(players)
    total_savings = values.get(grand)
    if total_savings is None:
        return None

    n_players = len(players)
    coalitions = [coalition for coalition in _all_subsets(players) if coalition != grand]
    p_idx = n_players
    slack_start = n_players + 1
    n_vars = n_players + 1 + len(coalitions)

    a_ub = []
    b_ub = []
    for slack_offset, coalition in enumerate(coalitions):
        row = np.zeros(n_vars)
        for idx, player in enumerate(players):
            if player in coalition:
                row[idx] = -1.0
        row[p_idx] = costs.get(coalition, 0.0)
        row[slack_start + slack_offset] = 1.0
        a_ub.append(row)
        b_ub.append(-values.get(coalition, 0.0))

    a_eq = np.zeros((1, n_vars))
    a_eq[0, :n_players] = 1.0
    b_eq = np.array([total_savings])
    bounds = [(None, None)] * (n_players + 1) + [(0.0, None)] * len(coalitions)

    primary_objective = np.zeros(n_vars)
    primary_objective[p_idx] = -1.0

    primary = linprog(
        primary_objective,
        A_ub=np.array(a_ub),
        b_ub=np.array(b_ub),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not primary.success:
        return None

    secondary_objective = np.zeros(n_vars)
    for slack_offset, coalition in enumerate(coalitions):
        coalition_cost = costs.get(coalition, 0.0)
        if coalition_cost:
            secondary_objective[slack_start + slack_offset] = -1.0 / coalition_cost

    fixed_p_row = np.zeros((1, n_vars))
    fixed_p_row[0, p_idx] = 1.0
    secondary_a_eq = np.vstack([a_eq, fixed_p_row])
    secondary_b_eq = np.concatenate([b_eq, np.array([primary.x[p_idx]])])

    secondary = linprog(
        secondary_objective,
        A_ub=np.array(a_ub),
        b_ub=np.array(b_ub),
        A_eq=secondary_a_eq,
        b_eq=secondary_b_eq,
        bounds=bounds,
        method="highs",
    )
    if not secondary.success:
        return None

    return {
        "allocation": {player: secondary.x[idx] for idx, player in enumerate(players)},
        "p": secondary.x[p_idx],
        "normalized_slack": -secondary.fun,
    }


def _minmax_allocation_for_tables(
    players: tuple[str, ...],
    values: dict[str, float],
    costs: dict[str, float],
) -> dict | None:
    return _minmax_allocation_fallback(players, values, costs)


def build_coalition_size_table(base_coalition_path: Path) -> pd.DataFrame:
    df, cost, _ = _load_coalition_metrics_for_tables(base_coalition_path)
    df = df.copy()
    df["normalized_sov_share"] = _normalize(df["SOV_share"])
    rows = []
    df = df.sort_values(["coalition_size", "coalition"])
    for row in df.to_dict("records"):
        rows.append(
            {
                "coalition": _display_coalition(row["coalition"]),
                "selected bases": _display_bases(_selected_bases_from_row(row)),
                "Objval [MEUR]": row["objective"] / 1e6,
                "savings [MEUR]": row["savings"] / 1e6,
                "synergy [%]": row["synergy"] * 100,
                "stable core": row["has_stable_core"],
                "best collaboration": _best_partition(row["coalition"], cost),
                "sov share": row["SOV_share"],
                "avg distance [km]": row["avg_pairwise_distance"],
                "n turbines": row["total_turbines"],
            }
        )
    return pd.DataFrame(rows)


def build_structural_synergy_table(base_coalition_path: Path) -> pd.DataFrame:
    df, _, _ = _load_coalition_metrics_for_tables(base_coalition_path)
    df = df[df["coalition_size"] == 3].copy()
    df["normalized_sov_share"] = _normalize(df["SOV_share"])
    return (
        df[
            [
                "coalition",
                "coalition_size",
                "avg_pairwise_distance",
                "max_pairwise_distance",
                "avg_distance_to_selected_base",
                "max_distance_to_selected_base",
                "total_turbines",
                "turbine_range",
                "savings",
                "synergy",
                "has_stable_core",
                "SOV_share",
                "normalized_sov_share",
            ]
        ]
        .rename(
            columns={
                "avg_pairwise_distance": "avg_pairwise_distance_km",
                "max_pairwise_distance": "max_pairwise_distance_km",
                "avg_distance_to_selected_base": "avg_distance_to_selected_base_km",
                "max_distance_to_selected_base": "max_distance_to_selected_base_km",
                "savings": "savings_meur",
                "synergy": "synergy_percent",
                "has_stable_core": "stable_core",
                "SOV_share": "sov_share",
            }
        )
        .assign(
            display_coalition=lambda x: x["coalition"].map(_display_coalition),
            savings_meur=lambda x: x["savings_meur"] / 1e6,
            synergy_percent=lambda x: x["synergy_percent"] * 100,
        )
        .sort_values("coalition")
    )


def _case_group_dirs(group_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in group_dir.iterdir()
        if path.is_dir()
        and (path / "coalition_oos.csv").exists()
        and (path / "windfarm_oos.csv").exists()
    )


DISTANCE_CASE_LABELS = {
    "BCD_close": "ABC",
    "BCG_cluster_far": "ABF",
    "BEG_spread": "ADF",
}
N_TURBINES_CASE_LABELS = {
    "BCD_high": "ABC high",
    "BCD_low": "ABC low",
    "BCD_mixed": "ABC mixed",
}


def _display_case_name(case_group: str, case_dir_name: str, coalition: str) -> str:
    if case_group == "distance":
        return DISTANCE_CASE_LABELS.get(case_dir_name, _display_coalition(coalition))
    if case_group == "n_turbines":
        return N_TURBINES_CASE_LABELS.get(case_dir_name, _display_coalition(coalition))
    return _display_coalition(coalition)


def _case_allocation_rows(case_group: str, case_dir: Path) -> list[dict]:
    coalition_path = case_dir / "coalition_oos.csv"
    windfarm_path = case_dir / "windfarm_oos.csv"
    df, cost, savings = _load_coalition_metrics_for_tables(coalition_path)
    grand_row = df.sort_values("coalition_size", ascending=False).iloc[0]
    coalition = grand_row["coalition"]
    players = _members(coalition)
    allocation = _minmax_allocation_for_tables(players, savings, cost)
    allocated = allocation["allocation"] if allocation else {}
    p_value = allocation["p"] if allocation else np.nan
    best_collaboration = _best_partition(coalition, cost)

    wf = pd.read_csv(windfarm_path)
    wf["coalition"] = wf["coalition"].map(lambda value: "".join(sorted(str(value))))
    wf = wf[wf["coalition"] == coalition].copy()

    rows = []
    case_name = _display_case_name(case_group, case_dir.name, coalition)
    for row in wf.to_dict("records"):
        wind_farm = str(row["wind_farm"])
        rows.append(
            {
                "case group": "distance" if case_group == "distance" else "n turbines",
                "case": case_name,
                "wind farm": _display_coalition(wind_farm),
                "SOV share": grand_row["SOV_share"],
                "Stable core": grand_row["has_stable_core"],
                "Best collaboration structure": best_collaboration,
                "allocated savings [MEUR]": allocated.get(wind_farm, np.nan) / 1e6,
                "minmax p": p_value,
                "time availability [%]": row.get("time_based_availability", np.nan) * 100,
                "value availability [%]": row.get("value_based_availability", np.nan) * 100,
                "downtime cost [MEUR]": row.get("downtime_cost", np.nan) / 1e6,
                "potential revenue [MEUR]": row.get("potential_revenue", np.nan) / 1e6,
            }
        )
    return rows


def build_allocation_availability_table() -> pd.DataFrame:
    rows = []
    for case_group in ["distance", "n_turbines"]:
        group_dir = RESULTS_DIR / "case_studies" / case_group
        if not group_dir.exists():
            continue
        for case_dir in _case_group_dirs(group_dir):
            rows.extend(_case_allocation_rows(case_group, case_dir))
    return pd.DataFrame(rows).sort_values(["case group", "case", "wind farm"])


def _case_variant_coalition_rows(case_group: str, case_dir: Path) -> list[dict]:
    coalition_path = case_dir / "coalition_oos.csv"
    windfarm_path = case_dir / "windfarm_oos.csv"
    df, cost, _ = _load_coalition_metrics_for_tables(coalition_path)
    wf = pd.read_csv(windfarm_path)
    wf["coalition"] = wf["coalition"].map(lambda value: "".join(sorted(str(value))))
    grand_row = df.sort_values("coalition_size", ascending=False).iloc[0]
    case_name = _display_case_name(case_group, case_dir.name, grand_row["coalition"])
    group_name = "distance" if case_group == "distance" else "n turbines"

    rows = []
    for row in df.sort_values(["coalition_size", "coalition"]).to_dict("records"):
        rows.append(
            {
                "case group": group_name,
                "case": case_name,
                "coalition": _display_coalition(row["coalition"]),
                "selected bases": _display_bases(_selected_bases_from_row(row)),
                "Objval [MEUR]": row["objective"] / 1e6,
                "savings [MEUR]": row["savings"] / 1e6,
                "synergy [%]": row["synergy"] * 100,
                "Stable core": row["has_stable_core"],
                "Best collaboration structure": _best_partition(row["coalition"], cost),
                "SOV share": row["SOV_share"],
                "avg distance [km]": row["avg_pairwise_distance"],
                "n turbines": wf.loc[wf["coalition"] == row["coalition"], "n_turbines"].sum(),
            }
        )
    return rows


def build_case_variant_coalition_table() -> pd.DataFrame:
    rows = []
    for case_group in ["distance", "n_turbines"]:
        group_dir = RESULTS_DIR / "case_studies" / case_group
        if not group_dir.exists():
            continue
        for case_dir in _case_group_dirs(group_dir):
            rows.extend(_case_variant_coalition_rows(case_group, case_dir))
    df = pd.DataFrame(rows)
    df["_coalition_size"] = df["coalition"].str.len()
    df = df.sort_values(["case group", "case", "_coalition_size", "coalition"])
    return df.drop(columns="_coalition_size")


def _format_member_values(values: dict[str, float], digits: int = 3) -> str:
    parts = []
    for member in sorted(values):
        value = values[member]
        if pd.isna(value):
            formatted = ""
        else:
            formatted = f"{value:.{digits}f}"
        parts.append(f"{_display_coalition(member)}: {formatted}")
    return "; ".join(parts)


def _case_combined_rows(case_group: str, case_dir: Path) -> list[dict]:
    coalition_path = case_dir / "coalition_oos.csv"
    windfarm_path = case_dir / "windfarm_oos.csv"
    df, cost, savings = _load_coalition_metrics_for_tables(coalition_path)
    wf = pd.read_csv(windfarm_path)
    wf["coalition"] = wf["coalition"].map(lambda value: "".join(sorted(str(value))))
    grand_row = df.sort_values("coalition_size", ascending=False).iloc[0]
    case_name = _display_case_name(case_group, case_dir.name, grand_row["coalition"])
    group_name = "distance" if case_group == "distance" else "n turbines"

    rows = []
    for row in df.sort_values(["coalition_size", "coalition"]).to_dict("records"):
        coalition = row["coalition"]
        players = _members(coalition)
        wf_rows = wf[wf["coalition"] == coalition].copy()
        allocation = None
        if len(players) > 1:
            allocation = _minmax_allocation_for_tables(players, savings, cost)
        allocated = (
            allocation["allocation"]
            if allocation
            else {player: 0.0 for player in players}
        )
        time_availability = {
            str(wf_row["wind_farm"]): wf_row.get("time_based_availability", np.nan) * 100
            for wf_row in wf_rows.to_dict("records")
        }
        value_availability = {
            str(wf_row["wind_farm"]): wf_row.get("value_based_availability", np.nan) * 100
            for wf_row in wf_rows.to_dict("records")
        }

        rows.append(
            {
                "case": case_name,
                "coalition": _display_coalition(coalition),
                "selected bases": _display_bases(_selected_bases_from_row(row)),
                "savings [MEUR]": row["savings"] / 1e6,
                "synergy [%]": row["synergy"] * 100,
                "SOV share": row["SOV_share"],
                "allocated savings [MEUR]": _format_member_values(
                    {player: allocated.get(player, np.nan) / 1e6 for player in players}
                ),
                "time availability [%]": _format_member_values(time_availability),
                "value availability [%]": _format_member_values(value_availability),
            }
        )
    return rows


def build_case_combined_results_table() -> pd.DataFrame:
    rows = []
    for case_group in ["distance", "n_turbines"]:
        group_dir = RESULTS_DIR / "case_studies" / case_group
        if not group_dir.exists():
            continue
        for case_dir in _case_group_dirs(group_dir):
            rows.extend(_case_combined_rows(case_group, case_dir))
    df = pd.DataFrame(rows)
    df["_coalition_size"] = df["coalition"].str.len()
    df = df.sort_values(["case", "_coalition_size", "coalition"])
    return df.drop(columns="_coalition_size")


def build_case_combined_results_table_for_group(case_group: str) -> pd.DataFrame:
    rows = []
    group_dir = RESULTS_DIR / "case_studies" / case_group
    if group_dir.exists():
        for case_dir in _case_group_dirs(group_dir):
            rows.extend(_case_combined_rows(case_group, case_dir))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_coalition_size"] = df["coalition"].str.len()
    df = df.sort_values(["case", "_coalition_size", "coalition"])
    return df.drop(columns="_coalition_size")


def build_appendix_tables(output_dir: Path) -> None:
    base_coalition_path = RESULTS_DIR / "case_studies" / "base" / "coalition_oos.csv"

    _write_csv(build_sampling_table(), output_dir / "method_sampling_results.csv")
    _write_csv(build_consensus_objective_table(), output_dir / "method_consensus_validation_objectives.csv")
    _write_csv(build_consensus_runtime_table(), output_dir / "method_consensus_validation_runtime.csv")
    _write_csv(build_stability_table(), output_dir / "method_stability_results.csv")
    _write_csv(build_coalition_size_table(base_coalition_path), output_dir / "case_coalition_size_results.csv")
    _write_csv(build_structural_synergy_table(base_coalition_path), output_dir / "case_structural_synergy_results.csv")
    _write_csv(build_allocation_availability_table(), output_dir / "case_allocation_availability_results.csv")
    _write_csv(build_case_variant_coalition_table(), output_dir / "case_variant_coalition_results.csv")
    _write_csv(build_case_combined_results_table_for_group("distance"), output_dir / "case_distance_combined_results.csv")
    _write_csv(build_case_combined_results_table_for_group("n_turbines"), output_dir / "case_turbines_combined_results.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned CSV tables for the thesis appendix.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where appendix CSV files are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_appendix_tables(args.output_dir)


if __name__ == "__main__":
    main()
