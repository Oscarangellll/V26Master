import argparse
import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd

from config.case_config import CaseConfig
from config.scenario_config import ScenarioConfig
from optimization_models.optimization_model import OptimizationModel


OSS_COLUMNS = [
    "tree_size",
    "count",
    "instances",
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
    "param_signature",
    "seed",
    "iss_pool",
    "oos_pool",
    "nested_trees",
    "iss_file",
]


def _parse_solution_group(group_name, encoded):
    if not isinstance(encoded, str) or encoded.strip() == "":
        return []

    entries = []
    for item in encoded.split(";"):
        key_str, val_str = item.rsplit(":", 1)
        val = int(val_str)

        if group_name == "eta":
            key = key_str
        elif group_name == "gamma_LT":
            h, b = key_str.split("|", 1)
            key = (h, b)
        elif group_name in {"gamma_ST", "alpha"}:
            h, b, t = key_str.split("|", 2)
            key = (h, b, t)
        else:
            raise ValueError(f"Unknown solution group: {group_name}")

        entries.append(((group_name, key), val))

    return entries


def _decode_solution(row):
    solution = []
    for group_name in ["eta", "gamma_LT", "gamma_ST", "alpha"]:
        solution.extend(_parse_solution_group(group_name, row.get(group_name, "")))
    return frozenset(solution)


def _solution_key(row_dict):
    return (
        row_dict.get("eta", ""),
        row_dict.get("gamma_LT", ""),
        row_dict.get("gamma_ST", ""),
        row_dict.get("alpha", ""),
    )


def _parse_instances(encoded):
    if not isinstance(encoded, str) or encoded.strip() == "":
        return set()
    return {int(x) for x in encoded.split(";") if x.strip()}


def _fix_solution(model, solution):
    for group_name in ["eta", "gamma_LT", "gamma_ST", "alpha"]:
        for _, var in getattr(model, group_name).items():
            var.LB = 0
            var.UB = 0

    for (group_name, key), value in solution:
        var = getattr(model, group_name)[key]
        var.LB = value
        var.UB = value


def _evaluate_solution(case, solution, oos_scenarios, scenario_cfg):
    totals = {
        "objective": 0.0,
        "first_stage_cost": 0.0,
        "second_stage_cost": 0.0,
        "charter_cost_ST": 0.0,
        "charter_cost_LT": 0.0,
        "charter_cost_mob": 0.0,
        "downtime_cost": 0.0,
        "travel_cost_S": 0.0,
        "travel_cost_M": 0.0,
        "runtime": 0.0,
        "MIPGap": 0.0,
    }

    for scenario_id in oos_scenarios:
        scenario_ids = [int(scenario_id)]
        model = OptimizationModel(case, scenario_cfg, scenario_ids)
        model.Params.OutputFlag = 0

        _fix_solution(model, solution)
        model.optimize()

        totals["objective"] += model.ObjVal
        totals["first_stage_cost"] += model.first_obj.getValue()
        totals["second_stage_cost"] += model.second_obj.getValue()
        totals["charter_cost_ST"] += model.charter_cost_ST.getValue()
        totals["charter_cost_LT"] += model.charter_cost_LT.getValue()
        totals["charter_cost_mob"] += model.charter_cost_mob.getValue()
        totals["downtime_cost"] += model.downtime_cost.getValue()
        totals["travel_cost_S"] += model.travel_cost_S.getValue()
        totals["travel_cost_M"] += model.travel_cost_M.getValue()
        totals["runtime"] += model.Runtime
        totals["MIPGap"] += model.MIPGap

    n = len(oos_scenarios)
    return {k: v / n for k, v in totals.items()}


def run_oos(args, iss_file: str):
    case = CaseConfig(f"cases/{args.case}.yaml")
    oos_scenarios = list(range(args.oos_pool_start, args.oos_pool_end + 1))

    iss_df = pd.read_csv(iss_file)

    grouped = defaultdict(list)
    for row in iss_df.itertuples(index=False):
        grouped[int(row.tree_size)].append(row)

    if args.oos_output is None:
        output_path = Path("results/stability") / args.case / args.method / "OSS.csv"
    else:
        output_path = Path(args.oos_output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    append_oos = bool(getattr(args, "append_oos", False))
    write_header = (not append_oos) or (not output_path.exists()) or output_path.stat().st_size == 0

    eval_cache = {}
    existing_instance_coverage = {}

    if append_oos and output_path.exists() and output_path.stat().st_size > 0:
        existing_df = pd.read_csv(output_path)
        for row in existing_df.itertuples(index=False):
            row_dict = row._asdict()
            key = _solution_key(row_dict)

            eval_cache[key] = {
                "objective": float(row.objective),
                "first_stage_cost": float(row.first_stage_cost),
                "second_stage_cost": float(row.second_stage_cost),
                "charter_cost_ST": float(row.charter_cost_ST),
                "charter_cost_LT": float(row.charter_cost_LT),
                "charter_cost_mob": float(row.charter_cost_mob),
                "downtime_cost": float(row.downtime_cost),
                "travel_cost_S": float(row.travel_cost_S),
                "travel_cost_M": float(row.travel_cost_M),
                "runtime": float(row.runtime),
                "MIPGap": float(row.MIPGap),
            }

            coverage_key = (int(row.tree_size), key)
            covered = existing_instance_coverage.setdefault(coverage_key, set())
            covered.update(_parse_instances(row_dict.get("instances", "")))

    with output_path.open("a" if append_oos else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(OSS_COLUMNS)

    for tree_size, rows in grouped.items():
        sol_to_instances = defaultdict(list)
        sol_meta = {}

        for row in rows:
            solution = _decode_solution(row._asdict())
            sol_to_instances[solution].append(int(row.instance_id))
            sol_meta[solution] = row

        for solution, instance_ids in sol_to_instances.items():
            row_like = {
                "eta": sol_meta[solution].eta,
                "gamma_LT": sol_meta[solution].gamma_LT,
                "gamma_ST": sol_meta[solution].gamma_ST,
                "alpha": sol_meta[solution].alpha,
            }
            key = _solution_key(row_like)

            sorted_ids = sorted(set(int(i) for i in instance_ids))
            coverage_key = (int(tree_size), key)
            old_ids = existing_instance_coverage.get(coverage_key, set()) if append_oos else set()
            delta_ids = [i for i in sorted_ids if i not in old_ids]

            if append_oos and len(delta_ids) == 0:
                continue

            if key in eval_cache:
                result = eval_cache[key]
            else:
                scenario_cfg = ScenarioConfig(case, oos_scenarios)
                result = _evaluate_solution(case, solution, oos_scenarios, scenario_cfg)
                eval_cache[key] = result

            meta = sol_meta[solution]
            row = [
                tree_size,
                len(delta_ids) if append_oos else len(sorted_ids),
                ";".join(map(str, delta_ids if append_oos else sorted_ids)),
                result["objective"],
                result["first_stage_cost"],
                result["second_stage_cost"],
                result["charter_cost_ST"],
                result["charter_cost_LT"],
                result["charter_cost_mob"],
                result["downtime_cost"],
                result["travel_cost_S"],
                result["travel_cost_M"],
                result["runtime"],
                result["MIPGap"],
                meta.eta,
                meta.gamma_LT,
                meta.gamma_ST,
                meta.alpha,
                meta.param_signature,
                int(meta.seed),
                meta.iss_pool,
                f"{args.oos_pool_start}-{args.oos_pool_end}",
                int(meta.nested_trees),
                str(iss_file),
            ]

            with output_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            covered = existing_instance_coverage.setdefault(coverage_key, set())
            covered.update(delta_ids if append_oos else sorted_ids)

    return str(output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OOS from ISS.csv and write OSS.csv")
    parser.add_argument("-c", "--case", required=True)
    parser.add_argument("-m", "--method", default="mip", choices=["mip"])
    parser.add_argument("--iss_file", required=True)
    parser.add_argument("--oos_pool_start", type=int, default=51)
    parser.add_argument("--oos_pool_end", type=int, default=300)
    parser.add_argument(
        "--append_oos",
        action="store_true",
        help="Append only new OOS rows and reuse prior evaluations from existing OSS.csv",
    )
    parser.add_argument("--oos_output", default=None)
    parser.add_argument("--seed", type=int, default=99)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    class Adapter:
        pass

    adapted = Adapter()
    adapted.case = args.case
    adapted.method = args.method
    adapted.oos_pool_start = args.oos_pool_start
    adapted.oos_pool_end = args.oos_pool_end
    adapted.append_oos = args.append_oos
    adapted.oos_output = args.oos_output

    path = run_oos(adapted, args.iss_file)
    print(f"OSS written to: {path}")


if __name__ == "__main__":
    main()
