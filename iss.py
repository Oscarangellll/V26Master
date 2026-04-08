import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import CaseConfig, ScenarioConfig
from config.scenario_reduction import perform_scenario_reduction
from optimization_models import OptimizationModel, ConsensusModelMP


ISS_COLUMNS = [
    "instance_id",
    "tree_size",
    "objective",
    "first_stage_cost",
    "second_stage_cost",
    "charter_cost_ST",
    "charter_cost_LT",
    "charter_cost_mob",
    "downtime_cost",
    "travel_cost_S",
    "travel_cost_M",
    "MIP_runtime",
    "Con_total runtime",
    "Con_eta runtime", 
    "Con_gamma_LT_runtime", 
    "Con_gamma_ST_runtime",
    "Con_Master_runtime",
    "Con_eta_n_iters",
    "Con_eta_vars_fixed",
    "Con_eta_judges_solved_min",
    "Con_eta_gap_p90_max",
    "Con_eta_unanimous_count",
    "Con_gamma_LT_n_iters",
    "Con_gamma_LT_vars_fixed",
    "Con_gamma_LT_judges_solved_min",
    "Con_gamma_LT_gap_p90_max",
    "Con_gamma_LT_unanimous_count",
    "Con_fix_history_json",
    "MIPGap",
    "has_solution",
    "solve_status_code",
    "solve_status",
    "solve_outcome",
    "gap_prune_enabled",
    "gap_pruned",
    "optimize_error",
    "eta",
    "gamma_LT",
    "gamma_ST",
    "alpha",
    "scenarios",
    "param_signature",
    "seed",
    "iss_pool",
    "oos_pool",
]


GRB_STATUS = {
    1: "LOADED",
    2: "OPTIMAL",
    3: "INFEASIBLE",
    4: "INF_OR_UNBD",
    5: "UNBOUNDED",
    6: "CUTOFF",
    7: "ITERATION_LIMIT",
    8: "NODE_LIMIT",
    9: "TIME_LIMIT",
    10: "SOLUTION_LIMIT",
    11: "INTERRUPTED",
    12: "NUMERIC",
    13: "SUBOPTIMAL",
    14: "INPROGRESS",
    15: "USER_OBJ_LIMIT",
}


def _status_label(status_code):
    if status_code is None:
        return None
    try:
        return GRB_STATUS.get(int(status_code), f"UNKNOWN_{status_code}")
    except Exception:
        return str(status_code)


def _safe_eval(getter):
    try:
        return getter()
    except Exception:
        return None


def _solve_outcome(has_solution, gap, gap_prune_threshold, optimize_error, status_label):
    if optimize_error:
        return "optimize_error"
    if has_solution:
        if math.isfinite(gap) and gap <= gap_prune_threshold:
            return "solution_within_gap"
        return "solution_gap_high_or_unknown"
    if status_label == "TIME_LIMIT":
        return "no_solution_time_limit"
    return "no_solution"


def _summarize_con_mp_insights(model):
    iteration_rows = getattr(model, "fix_iteration_summaries", None)
    if not iteration_rows:
        return {}, {}

    by_group = {}
    for row in iteration_rows:
        group = row.get("group")
        if group not in by_group:
            by_group[group] = []
        by_group[group].append(row)

    group_summaries = {}
    for group, rows in by_group.items():
        n_iters = len(rows)
        vars_fixed = sum(r.get("fixed_this_iter", 0) for r in rows)
        judges_solved_vals = [r.get("n_judges_solved") for r in rows if r.get("n_judges_solved") is not None]
        judges_solved_min = min(judges_solved_vals) if judges_solved_vals else None
        gap_p90_vals = [r.get("judge_gap_p90") for r in rows if r.get("judge_gap_p90") is not None]
        gap_p90_max = max(gap_p90_vals) if gap_p90_vals else None
        unanimous_count = sum(1 for r in rows if r.get("unanimous"))

        group_summaries[group] = {
            "n_iters": n_iters,
            "vars_fixed": vars_fixed,
            "judges_solved_min": judges_solved_min,
            "gap_p90_max": gap_p90_max,
            "unanimous_count": unanimous_count,
        }

    compact_history = {
        group: [
            {
                "iter": r.get("iteration"),
                "solved": r.get("n_judges_solved"),
                "failed": r.get("n_judges_failed"),
                "gap_p90": round(r.get("judge_gap_p90")) if r.get("judge_gap_p90") is not None else None,
                "cache_hr": round(r.get("cache_hit_rate") * 100) if r.get("cache_hit_rate") is not None else None,
                "fixed": r.get("fixed_this_iter"),
                "unan": r.get("unanimous"),
                "key": r.get("critical_key")[:20] if r.get("critical_key") else None,
            }
            for r in by_group.get(group, [])
        ]
        for group in sorted(by_group.keys())
    }

    return group_summaries, compact_history


def _encode_key(key):
    if isinstance(key, tuple):
        return "|".join(map(str, key))
    return str(key)


def _encode_solution_group(solution, group):
    items = sorted(
        (
            (_encode_key(key), val)
            for (var_group, key), val in solution
            if var_group == group and val > 0
        ),
        key=lambda t: t[0],
    )
    return ";".join(f"{key}:{val}" for key, val in items)


def _encode_scenarios(scenarios):
    return ";".join(map(str, sorted(int(s) for s in scenarios)))


def _sample_scenarios(rng, iss_pool, tree_sizes):
    return {
        size: rng.choice(iss_pool, size=size, replace=False)
        for size in tree_sizes
    }

def _instance_pool(instance_id, pool_start, pool_end, pool_size):
    lo = pool_start + (instance_id - 1) * pool_size
    hi = lo + pool_size - 1
    if hi > pool_end:
        raise ValueError(
            "Not enough ISS scenarios for requested instances. "
            f"Instance {instance_id} requires [{lo}, {hi}] but iss_pool_end={pool_end}."
        )
    return list(range(lo, hi + 1))


def _preload_reduction_inputs(case, all_iss_scenarios):
    scenario_data_dir = Path(os.environ.get("SCENARIO_DATA_DIR", "data/scenario_data"))

    df_weather_windows = pd.read_parquet(
        scenario_data_dir / "weather_windows",
        filters=[
            ("wl_id", "in", [w.weather_location_id for w in case.wind_farms]),
            ("s", "in", all_iss_scenarios),
            ("h", "in", case.H),
        ],
    )
    weather_windows = defaultdict(dict)
    for r in df_weather_windows.itertuples():
        weather_windows[r.s][(r.h, r.wl_id, r.d)] = r.ww

    df_failures = pd.read_parquet(
        scenario_data_dir / "failures",
        filters=[
            ("w", "in", case.W),
            ("s", "in", all_iss_scenarios),
        ],
    )
    failures = defaultdict(dict)
    for r in df_failures.itertuples():
        failures[r.s][(r.w, r.m, r.d)] = r.failures

    df_downtime = pd.read_parquet(
        scenario_data_dir / "downtime_cost",
        filters=[
            ("w", "in", case.W),
            ("s", "in", all_iss_scenarios),
        ],
    )
    downtime_cost = defaultdict(dict)
    for r in df_downtime.itertuples():
        downtime_cost[r.s][(r.w, r.d)] = r.downtime_cost

    return weather_windows, failures, downtime_cost


def _prepare_iss_plan(args, case, rng, start_instance_id, scenario_tree_sizes):
    all_iss_scenarios = list(range(args.iss_pool_start, args.iss_pool_end + 1))

    scenario_ids_by_instance_tree = {}
    weights_by_instance_tree = {}

    weather_windows = failures = downtime_cost = None
    if args.scenario_reduction:
        weather_windows, failures, downtime_cost = _preload_reduction_inputs(
            case,
            all_iss_scenarios,
        )

    for instance_id in range(start_instance_id, start_instance_id + args.n_trees):
        instance_pool = _instance_pool(
            instance_id,
            args.iss_pool_start,
            args.iss_pool_end,
            args.instance_pool_size,
        )

        if args.scenario_reduction:
            for tree_size in scenario_tree_sizes:
                reduced_ids, reduced_weights = perform_scenario_reduction(
                    case,
                    instance_pool,
                    weather_windows,
                    downtime_cost,
                    failures,
                    n_reduced_scenarios=tree_size,
                    features_setting=args.features_setting,
                )
                scenario_ids_by_instance_tree[(instance_id, tree_size)] = [
                    int(s) for s in reduced_ids
                ]
                weights_by_instance_tree[(instance_id, tree_size)] = {
                    int(s): float(w) for s, w in reduced_weights.items()
                }
        else:
            sampled = _sample_scenarios(
                rng,
                np.array(instance_pool),
                scenario_tree_sizes,
            )
            for tree_size in scenario_tree_sizes:
                selected_ids = [int(s) for s in sampled[tree_size]]
                scenario_ids_by_instance_tree[(instance_id, tree_size)] = selected_ids
                print(f"selected_ids for instance {instance_id} tree size {tree_size}: {selected_ids}")
                weights_by_instance_tree[(instance_id, tree_size)] = {
                    int(s): 1.0 / len(selected_ids) for s in selected_ids
                }

    complete_scenario_pool = sorted(
        {
            s
            for ids in scenario_ids_by_instance_tree.values()
            for s in ids
        }
    )

    return scenario_ids_by_instance_tree, weights_by_instance_tree, complete_scenario_pool


def _param_signature(args, case):
    payload = {
        "case": args.case,
        "method": args.method,
        "vessel_types": [
            {
                "name": h.name,
                "required_capacity": h.required_capacity,
                "multiday": h.multiday,
                "day_rate_ST": h.day_rate_ST,
                "day_rate_LT": h.day_rate_LT,
                "mob_rate": h.mob_rate,
                "n_teams": h.n_teams,
                "travel_speed": h.travel_speed,
                "max_wave": h.max_wave,
                "cost_per_km": h.cost_per_km,
                "periodic_return": h.periodic_return,
            }
            for h in case.vessel_types
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _next_instance_id(output_path: Path) -> int:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return 1

    max_instance_id = 0
    with output_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("instance_id", "")
            if not raw:
                continue
            max_instance_id = max(max_instance_id, int(raw))

    return max_instance_id + 1


def run_iss(args) -> str:
    case = CaseConfig(f"cases/{args.case}.yaml")
    rng = np.random.default_rng(seed=args.seed)

    scenario_tree_sizes = sorted(args.scenario_tree_sizes)
    gap_prune_threshold = float(getattr(args, "gap_prune_threshold", 0.10))
    max_allowed_tree_size = max(scenario_tree_sizes)

    param_signature = _param_signature(args, case)

    if args.iss_output is None:
        output_path = Path("results/stability") / args.case / args.method / "ISS.csv"
    else:
        output_path = Path(args.iss_output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    append_mode = bool(getattr(args, "append", False))
    write_header = (not append_mode) or (not output_path.exists()) or output_path.stat().st_size == 0
    start_instance_id = _next_instance_id(output_path) if append_mode else 1

    with output_path.open("a" if append_mode else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(ISS_COLUMNS)

    var_groups = ["eta", "gamma_LT", "gamma_ST", "alpha"]

    scenario_ids_plan, scenario_weights_plan, complete_scenario_pool = _prepare_iss_plan(
        args,
        case,
        rng,
        start_instance_id,
        scenario_tree_sizes,
    )

    for instance_id in range(start_instance_id, start_instance_id + args.n_trees):
        for tree_size in scenario_tree_sizes:
            if tree_size > max_allowed_tree_size:
                continue

            scenario_ids = scenario_ids_plan[(instance_id, tree_size)]
            weights = scenario_weights_plan[(instance_id, tree_size)]

            model = None
            optimize_error = None
            try:
                if args.method == "mip":
                    scenario_cfg = ScenarioConfig(case, scenario_ids)
                    model = OptimizationModel(case, scenario_cfg, scenario_ids, weights)
                    model.Params.OutputFlag = 0
                    model.Params.Timelimit = 4 * 3600 
                    model.Params.MIPGap = 0.02
                    model.optimize()

                elif args.method == "con_mp":
                    model = ConsensusModelMP(case, scenario_ids, weights)
                    model.optimize()

                else:
                    raise ValueError(f"Unsupported method: {args.method}")
            except Exception as exc:
                optimize_error = f"{type(exc).__name__}: {exc}"
                print(
                    "[ISS error] "
                    f"Optimization failed for instance={instance_id}, tree_size={tree_size}, method={args.method}. "
                    f"error={optimize_error}"
                )

            has_solution = bool(model is not None and getattr(model, "SolCount", 0) > 0)
            if has_solution:
                solution = frozenset(
                    ((group, idx), int(var.X))
                    for group in var_groups
                    for idx, var in getattr(model, group).items()
                )
            else:
                solution = frozenset()
                print(
                    "[ISS warning] "
                    f"No incumbent found for instance={instance_id}, tree_size={tree_size}, method={args.method}, "
                    f"status={getattr(model, 'Status', None)}, error={optimize_error}."
                )

            status_code = getattr(model, "Status", None) if model is not None else None
            status_label = _status_label(status_code)

            try:
                gap = float(model.MIPGap) if model is not None else float("nan")
            except Exception:
                gap = float("nan")

            solve_outcome = _solve_outcome(
                has_solution=has_solution,
                gap=gap,
                gap_prune_threshold=gap_prune_threshold,
                optimize_error=optimize_error,
                status_label=status_label,
            )
            gap_bad = (not math.isfinite(gap)) or gap > gap_prune_threshold
            gap_pruned = gap_bad and tree_size < max_allowed_tree_size

            (
                group_summaries,
                compact_history,
            ) = (
                _summarize_con_mp_insights(model)
                if args.method == "con_mp"
                else ({}, {})
            )
            
            row = [
                instance_id,
                tree_size,
                _safe_eval(lambda: model.ObjVal) if has_solution else None,
                _safe_eval(lambda: model.first_obj.getValue()) if has_solution else None,
                _safe_eval(lambda: model.second_obj.getValue()) if has_solution else None,
                _safe_eval(lambda: model.charter_cost_ST.getValue()) if has_solution else None,
                _safe_eval(lambda: model.charter_cost_LT.getValue()) if has_solution else None,
                _safe_eval(lambda: model.charter_cost_mob.getValue()) if has_solution else None,
                _safe_eval(lambda: model.downtime_cost.getValue()) if has_solution else None,
                _safe_eval(lambda: model.travel_cost_S.getValue()) if has_solution else None,
                _safe_eval(lambda: model.travel_cost_M.getValue()) if has_solution else None,
                _safe_eval(lambda: model.Runtime) if args.method == "mip" and model is not None else None,
                _safe_eval(lambda: model.total_consensus_time) if args.method == "con_mp" and model is not None else None,
                _safe_eval(lambda: model.time_to_fix_eta) if args.method == "con_mp" and model is not None else None,
                _safe_eval(lambda: model.time_to_fix_gamma_LT) if args.method == "con_mp" and model is not None else None,
                _safe_eval(lambda: model.time_to_tighten_gamma_ST) if args.method == "con_mp" and model is not None else None,
                _safe_eval(lambda: model.Runtime) if args.method == "con_mp" and model is not None else None,
                group_summaries.get("eta", {}).get("n_iters") if group_summaries else None,
                group_summaries.get("eta", {}).get("vars_fixed") if group_summaries else None,
                group_summaries.get("eta", {}).get("judges_solved_min") if group_summaries else None,
                group_summaries.get("eta", {}).get("gap_p90_max") if group_summaries else None,
                group_summaries.get("eta", {}).get("unanimous_count") if group_summaries else None,
                group_summaries.get("gamma_LT", {}).get("n_iters") if group_summaries else None,
                group_summaries.get("gamma_LT", {}).get("vars_fixed") if group_summaries else None,
                group_summaries.get("gamma_LT", {}).get("judges_solved_min") if group_summaries else None,
                group_summaries.get("gamma_LT", {}).get("gap_p90_max") if group_summaries else None,
                group_summaries.get("gamma_LT", {}).get("unanimous_count") if group_summaries else None,
                json.dumps(compact_history, separators=(",", ":")) if compact_history else None,
                gap,
                has_solution,
                status_code,
                status_label,
                solve_outcome,
                gap_prune_enabled,
                gap_pruned,
                optimize_error,
                _encode_solution_group(solution, "eta"),
                _encode_solution_group(solution, "gamma_LT"),
                _encode_solution_group(solution, "gamma_ST"),
                _encode_solution_group(solution, "alpha"),
                _encode_scenarios(scenario_ids),
                param_signature,
                args.seed,
                f"{_instance_pool(instance_id, args.iss_pool_start, args.iss_pool_end, args.instance_pool_size)[0]}-{_instance_pool(instance_id, args.iss_pool_start, args.iss_pool_end, args.instance_pool_size)[-1]}",
                f"{args.oos_pool_start}-{args.oos_pool_end}",
            ]

            with output_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            if gap_pruned:
                max_allowed_tree_size = tree_size - 1
                print(
                    "[ISS pruning] "
                    f"instance={instance_id}, tree_size={tree_size}, gap={gap} > {gap_prune_threshold}. "
                    f"Skipping larger tree sizes (> {max_allowed_tree_size}) for this and future instances."
                )

    return str(output_path)

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ISS and write ISS.csv")
    parser.add_argument("-c", "--case", required=True)
    parser.add_argument("-m", "--method", default="mip", choices=["mip", "con", "con_mp"])
    parser.add_argument("-n", "--n_trees", type=int, required=True)
    parser.add_argument("-s", "--scenario_tree_sizes", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--iss_pool_start", type=int, default=1)
    parser.add_argument("--iss_pool_end", type=int, default=3000)
    parser.add_argument("--oos_pool_start", type=int, default=3001)
    parser.add_argument("--oos_pool_end", type=int, default=4000)
    parser.add_argument(
        "--gap_prune_threshold",
        type=float,
        default=0.10,
        help="Skip larger scenario tree sizes when an evaluated size exceeds this MIPGap threshold.",
    )
    parser.add_argument(
        "--scenario_reduction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use k-medoids scenario reduction within each instance pool",
    )
    parser.add_argument(
        "--instance_pool_size",
        type=int,
        default=100,
        help="Number of ISS scenarios per instance",
    )
    parser.add_argument(
        "--nested_trees",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append ISS rows to existing ISS.csv",
    )
    parser.add_argument("--iss_output", default=None)
    parser.add_argument("--oos_output", default=None)
    #add parser for features: should have int between 1 and 3 (1 for only weather features, 2 for weather + downtime, 3 for weather + downtime + failures)
    parser.add_argument(
        "--features_setting",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Features to use for scenario reduction: 1 for only weather features, 2 for weather + downtime, 3 for weather + downtime + failures",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    path = run_iss(args)
    print(f"ISS written to: {path}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn")

    main()
