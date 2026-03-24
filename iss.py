import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from config.case_config import CaseConfig
from config.scenario_config import ScenarioConfig
from config.scenario_reduction import perform_scenario_reduction
from optimization_models.optimization_model import OptimizationModel
from optimization_models.consensus_model import ConsensusModel
from optimization_models.consensus_model_multiprocessing import ConsensusModelMP


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
    "runtime",
    "MasterSolve_runtime",
    "MIPGap",
    "eta",
    "gamma_LT",
    "gamma_ST",
    "alpha",
    "scenarios",
    "param_signature",
    "seed",
    "iss_pool",
    "oos_pool",
    "nested_trees",
]


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


def _sample_scenarios(rng, iss_pool, tree_sizes, nested):
    if nested:
        draw = rng.choice(iss_pool, size=max(tree_sizes), replace=False)
        return {size: draw[:size] for size in tree_sizes}

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
        scenario_data_dir / "weather_windows.parquet",
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
        scenario_data_dir / "failures.parquet",
        filters=[
            ("w", "in", case.W),
            ("s", "in", all_iss_scenarios),
        ],
    )
    failures = defaultdict(dict)
    for r in df_failures.itertuples():
        failures[r.s][(r.w, r.m, r.d)] = r.failures

    df_downtime = pd.read_parquet(
        scenario_data_dir / "downtime_cost.parquet",
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
                args.nested_trees,
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
    scenario_cfg = ScenarioConfig(
        case,
        complete_scenario_pool,
    )

    for instance_id in range(start_instance_id, start_instance_id + args.n_trees):
        for tree_size in scenario_tree_sizes:
            scenario_ids = scenario_ids_plan[(instance_id, tree_size)]
            scenario_cfg.weights = scenario_weights_plan[(instance_id, tree_size)]

            if args.method == "mip":
                model = OptimizationModel(case, scenario_cfg, scenario_ids)
                model.Params.OutputFlag = 0
                model.Params.MIPGap = 0.02
                model.optimize()
            elif args.method == "con":
                judge_seeds = scenario_ids
                master_scenarios = judge_seeds[:]
                
                cm = ConsensusModel(
                    case, 
                    scenario_cfg, 
                    judge_seeds_1scenario_each=judge_seeds,
                    mip_gap_judges=0.02,
                    log=False
                )
                
                model, runtime = cm.optimize(
                    master_scenarios=master_scenarios,
                    eta_max_iters=50,
                    lt_max_iters=200,
                    top_k_eta=1,
                    top_k_lt=1,
                    min_p=0.55,
                    max_p=0.95,
                    aggregator="mean",
                    tighten_ub_st=True,
                    unanim_fix_zero_st=True,
                    mip_gap_master=0.02
                )
            elif args.method == "con_mp":
                judge_seeds = scenario_ids
                master_scenarios = judge_seeds[:]
                
                cm = ConsensusModelMP(
                    case, 
                    scenario_cfg, 
                    judge_seeds_1scenario_each=judge_seeds,
                    mip_gap_judges=0.02,
                    cap_workers=14,
                    log=False
                )
                
                model, runtime = cm.optimize(
                    master_scenarios=master_scenarios,
                    eta_max_iters=50,
                    lt_max_iters=200,
                    top_k_eta=1,
                    top_k_lt=1,
                    min_p=0.60,
                    max_p=0.99,
                    aggregator="mean",
                    tighten_ub_st=True,
                    unanim_fix_zero_st=True,
                    mip_gap_master=0.02
                )
                
            else:
                raise ValueError(f"Unsupported method: {args.method}")
            

            solution = frozenset(
                ((group, key), int(var.X))
                for group in var_groups
                for key, var in getattr(model, group).items()
            )

            row = [
                instance_id,
                tree_size,
                model.ObjVal,
                model.first_obj.getValue(),
                model.second_obj.getValue(),
                model.charter_cost_ST.getValue(),
                model.charter_cost_LT.getValue(),
                model.charter_cost_mob.getValue(),
                model.downtime_cost.getValue(),
                model.travel_cost_S.getValue(),
                model.travel_cost_M.getValue(),
                model.Runtime if args.method == "mip" else runtime,
                model.Runtime if not args.method == "mip" else None,
                model.MIPGap,
                _encode_solution_group(solution, "eta"),
                _encode_solution_group(solution, "gamma_LT"),
                _encode_solution_group(solution, "gamma_ST"),
                _encode_solution_group(solution, "alpha"),
                _encode_scenarios(scenario_ids),
                param_signature,
                args.seed,
                f"{_instance_pool(instance_id, args.iss_pool_start, args.iss_pool_end, args.instance_pool_size)[0]}-{_instance_pool(instance_id, args.iss_pool_start, args.iss_pool_end, args.instance_pool_size)[-1]}",
                f"{args.oos_pool_start}-{args.oos_pool_end}",
                int(args.nested_trees),
            ]

            with output_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)

    return str(output_path)

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ISS and write ISS.csv")
    parser.add_argument("-c", "--case", required=True)
    parser.add_argument("-m", "--method", default="mip", choices=["mip", "con", "con_mp"])
    parser.add_argument("-n", "--n_trees", type=int, required=True)
    parser.add_argument("-s", "--scenario_tree_sizes", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--iss_pool_start", type=int, default=1)
    parser.add_argument("--iss_pool_end", type=int, default=50)
    parser.add_argument("--oos_pool_start", type=int, default=51)
    parser.add_argument("--oos_pool_end", type=int, default=300)
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
        default=True,
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append ISS rows to existing ISS.csv",
    )
    parser.add_argument("--iss_output", default=None)
    parser.add_argument("--oos_output", default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    path = run_iss(args)
    print(f"ISS written to: {path}")


if __name__ == "__main__":
    main()
