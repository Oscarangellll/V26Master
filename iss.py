import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from config.case_config import CaseConfig
from config.scenario_config import ScenarioConfig
from optimization_models.optimization_model import OptimizationModel


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
    iss_pool = np.arange(args.iss_pool_start, args.iss_pool_end + 1)

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

    for instance_id in range(start_instance_id, start_instance_id + args.n_trees):
        sampled = _sample_scenarios(
            rng,
            iss_pool,
            scenario_tree_sizes,
            args.nested_trees,
        )

        for tree_size in scenario_tree_sizes:
            scenario_ids = [int(s) for s in sampled[tree_size]]
            scenario_cfg = ScenarioConfig(case, scenario_ids)
            model = OptimizationModel(case, scenario_cfg, scenario_ids)
            model.Params.OutputFlag = 0
            model.optimize()

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
                model.Runtime,
                model.MIPGap,
                _encode_solution_group(solution, "eta"),
                _encode_solution_group(solution, "gamma_LT"),
                _encode_solution_group(solution, "gamma_ST"),
                _encode_solution_group(solution, "alpha"),
                _encode_scenarios(scenario_ids),
                param_signature,
                args.seed,
                f"{args.iss_pool_start}-{args.iss_pool_end}",
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
    parser.add_argument("-m", "--method", default="mip", choices=["mip"])
    parser.add_argument("-n", "--n_trees", type=int, required=True)
    parser.add_argument("-s", "--scenario_tree_sizes", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--iss_pool_start", type=int, default=1)
    parser.add_argument("--iss_pool_end", type=int, default=50)
    parser.add_argument("--oos_pool_start", type=int, default=51)
    parser.add_argument("--oos_pool_end", type=int, default=300)
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
