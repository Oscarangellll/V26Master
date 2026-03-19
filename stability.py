import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.case_config import CaseConfig
from config.scenario_config import ScenarioConfig
from optimization_models.optimization_model import OptimizationModel

parser = argparse.ArgumentParser()

parser.add_argument(
    "-c", "--case",
    required=True,
    help="Case name"
)

parser.add_argument(
    "-m", "--method",
    required=True,
    choices=["mip", "con"],
    help="Solution method"
)

parser.add_argument(
    "-n", "--n_trees",
    type=int,
    required=True,
    help="Number of trees to solve for each scenario tree size"
)

parser.add_argument(
    "-s", "--scenario_tree_sizes",
    type=int,
    nargs="+",
    required=True,
    help="List of scenario tree sizes (e.g. 10 20 50)"
)

parser.add_argument(
    "--seed",
    type=int,
    default=99,
    help="Random seed for scenario sampling"
)

parser.add_argument(
    "--iss_pool_start",
    type=int,
    default=1,
    help="First scenario ID in in-sample pool (inclusive)"
)

parser.add_argument(
    "--iss_pool_end",
    type=int,
    default=50,
    help="Last scenario ID in in-sample pool (inclusive)"
)

parser.add_argument(
    "--oos_pool_start",
    type=int,
    default=51,
    help="First scenario ID in out-of-sample pool (inclusive)"
)

parser.add_argument(
    "--oos_pool_end",
    type=int,
    default=300,
    help="Last scenario ID in out-of-sample pool (inclusive)"
)

parser.add_argument(
    "--nested_trees",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use nested in-sample trees per replication (size k extends size k-1). Default: True. Use --no-nested_trees to disable."
)

parser.add_argument(
    "--append",
    action="store_true",
    help="Append to existing ISS/OSS files instead of overwriting. Automatically continues from the next instance_id."
)

args = parser.parse_args()

case = CaseConfig(f"cases/{args.case}.yaml")

rng = np.random.default_rng(seed=args.seed)

scenario_tree_sizes = sorted(args.scenario_tree_sizes)
n_trees = args.n_trees

iss_pool = np.arange(args.iss_pool_start, args.iss_pool_end + 1)
oos_pool = np.arange(args.oos_pool_start, args.oos_pool_end + 1)

if len(set(iss_pool).intersection(set(oos_pool))) > 0:
    raise ValueError("In-sample and out-of-sample scenario pools overlap. Use disjoint ranges.")

if max(scenario_tree_sizes) > len(iss_pool):
    raise ValueError(
        f"Largest tree size ({max(scenario_tree_sizes)}) exceeds in-sample pool size ({len(iss_pool)})."
    )


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
        key=lambda t: t[0]
    )
    return ";".join(f"{key}:{val}" for key, val in items)


def _encode_scenarios(scenarios):
    return ";".join(map(str, sorted(int(s) for s in scenarios)))


param_signature_payload = {
    "case": args.case,
    "method": args.method,
    "vessel_types": [
        {
            "name": h.name,
            "required_capacity": h.required_capacity,
            "multiday": h.multiday,
            "day_rate": h.day_rate,
            "mob_rate": h.mob_rate,
            "n_teams": h.n_teams,
            "travel_speed": h.travel_speed,
            "max_wind": h.max_wind,
            "max_wave": h.max_wave,
            "cost_per_km": h.cost_per_km,
            "periodic_return": h.periodic_return,
        }
        for h in case.vessel_types
    ],
}
param_signature = hashlib.sha256(
    json.dumps(param_signature_payload, sort_keys=True).encode("utf-8")
).hexdigest()[:16]


def _sample_scenarios_for_replication():
    if args.nested_trees:
        draw = rng.choice(iss_pool, size=max(scenario_tree_sizes), replace=False)
        return {st_size: draw[:st_size] for st_size in scenario_tree_sizes}

    return {
        st_size: rng.choice(iss_pool, size=st_size, replace=False)
        for st_size in scenario_tree_sizes
    }


# cache: {st_size: {solution: [instance_ids]}}
cache = {st_size: {} for st_size in scenario_tree_sizes}

var_groups = ["eta", "gamma_LT", "gamma_ST", "alpha"]

results_path = Path("results/stability") / args.case / args.method / "ISS.csv"
results_path.parent.mkdir(parents=True, exist_ok=True)

# Determine starting instance_id and handle append mode
start_instance_id = 1
if args.append and results_path.exists():
    iss_data = pd.read_csv(results_path)
    start_instance_id = int(iss_data["instance_id"].max()) + 1
    print(f"Appending mode: continuing from instance_id={start_instance_id}")
    
    # Warm up RNG to skip to the correct point in the sequence
    # This ensures we get NEW scenarios, not repeats of old ones
    skip_count = start_instance_id - 1
    if args.nested_trees:
        # Nested: 1 rng.choice() call per instance
        for _ in range(skip_count):
            _ = rng.choice(iss_pool, size=max(scenario_tree_sizes), replace=False)
    else:
        # Non-nested: len(scenario_tree_sizes) rng.choice() calls per instance
        for _ in range(skip_count * len(scenario_tree_sizes)):
            _ = rng.choice(iss_pool, size=max(scenario_tree_sizes), replace=False)
    print(f"RNG warmed up: skipped {skip_count} instance(s) to sync with seed and sequence")
else:
    # Overwrite mode: start fresh
    with results_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "instance_id", "tree_size", "objective", "first_stage_cost", "second_stage_cost", "charter_cost_ST", "charter_cost_LT", "charter_cost_mob","downtime_cost", "travel_cost_S", "travel_cost_M", "runtime", "MIPGap", "gamma_LT", "gamma_ST", "scenarios", "param_signature", "seed", "iss_pool", "oos_pool", "nested_trees"
        ])
    if not args.append:
        print(f"Fresh start: instance_id=1")

for j_offset in range(n_trees):
    j = start_instance_id - 1 + j_offset
    actual_instance_id = j + 1
    
    sampled_scenarios = _sample_scenarios_for_replication()

    for st_size in scenario_tree_sizes:
        s = sampled_scenarios[st_size]
        scenario = ScenarioConfig(case, s)

        if args.method == "mip":
            model = OptimizationModel(case, scenario)

            model.Params.OutputFlag = 0

            model.optimize()

            runtime = model.Runtime

        solution = frozenset(
            ((var_group, key), int(var.X))
            for var_group in var_groups
            for key, var in getattr(model, var_group).items()
        )
            
        gamma_LT_str = _encode_solution_group(solution, "gamma_LT")
        gamma_ST_str = _encode_solution_group(solution, "gamma_ST")
            
        if solution not in cache[st_size]:
            cache[st_size][solution] = []
        cache[st_size][solution].append(actual_instance_id)
        
        with results_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([   
                actual_instance_id,
                st_size, 
                model.ObjVal,
                model.first_obj.getValue(),
                model.second_obj.getValue(),
                model.charter_cost_ST.getValue(),
                model.charter_cost_LT.getValue(),
                model.charter_cost_mob.getValue(),
                model.downtime_cost.getValue(),
                model.travel_cost_S.getValue(),
                model.travel_cost_M.getValue(),
                runtime,
                model.MIPGap,
                gamma_LT_str,
                gamma_ST_str,
                _encode_scenarios(s),
                param_signature,
                args.seed,
                f"{args.iss_pool_start}-{args.iss_pool_end}",
                f"{args.oos_pool_start}-{args.oos_pool_end}",
                int(args.nested_trees)
            ])


oss_results_path = Path("results/stability") / args.case / args.method / "OSS.csv"
oss_results_path.parent.mkdir(parents=True, exist_ok=True)

# In append mode, leave OSS.csv as is; otherwise (re)create it fresh
if not args.append or not oss_results_path.exists():
    with oss_results_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tree_size", "count", "instances", "objective", "first_stage_cost", "second_stage_cost", "charter_cost_ST", "charter_cost_LT", "downtime_cost", "travel_cost_S", "travel_cost_M", "gamma_LT", "gamma_ST", "param_signature", "seed", "iss_pool", "oos_pool", "nested_trees"
        ])

# The true distribution is the same for all solutions
true_distribution = oos_pool
def evaluate_solution(solution): 
    results = {
        "objective": 0,
        "first_stage_cost": 0,
        "second_stage_cost": 0,
        "charter_cost_ST": 0,
        "charter_cost_LT": 0,
        "downtime_cost": 0,
        "travel_cost_S": 0,
        "travel_cost_M": 0,
    }
    for scenario in true_distribution:
        scenario_cfg = ScenarioConfig(case, [scenario])
        model = OptimizationModel(case, scenario_cfg)
        model.Params.OutputFlag = 0
        
        for (group, key), val in solution:
            var = getattr(model, group)[key]
            var.LB = val
            var.UB = val

        model.optimize()
        results["objective"] += model.ObjVal
        results["first_stage_cost"] += model.first_obj.getValue()
        results["second_stage_cost"] += model.second_obj.getValue()
        results["downtime_cost"] += model.downtime_cost.getValue()
        results["travel_cost_S"] += model.travel_cost_S.getValue()
        results["travel_cost_M"] += model.travel_cost_M.getValue()
        results["charter_cost_ST"] += model.charter_cost_ST.getValue()
        results["charter_cost_LT"] += model.charter_cost_LT.getValue()

    # Divide all results by the number of scenarios to get average cost per scenario
    results = {key: val / len(true_distribution) for key, val in results.items()}

    return results

evaluated_solutions = {}
for tree_size, sol_dict in cache.items():
    
    for solution, instance_ids in sol_dict.items():
        count = len(instance_ids)
        instances_str = ";".join(map(str, sorted(instance_ids)))

        if solution in evaluated_solutions:
            results = evaluated_solutions[solution]
            
        else:
            results = evaluate_solution(solution)
            evaluated_solutions[solution] = results

        gamma_LT_str = _encode_solution_group(solution, "gamma_LT")
        gamma_ST_str = _encode_solution_group(solution, "gamma_ST")
        
        with oss_results_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                tree_size,
                count,
                instances_str,
                results["objective"],
                results["first_stage_cost"],
                results["second_stage_cost"],
                results["charter_cost_ST"],
                results["charter_cost_LT"],
                results["downtime_cost"],
                results["travel_cost_S"],
                results["travel_cost_M"],
                gamma_LT_str,
                gamma_ST_str,
                param_signature,
                args.seed,
                f"{args.iss_pool_start}-{args.iss_pool_end}",
                f"{args.oos_pool_start}-{args.oos_pool_end}",
                int(args.nested_trees)
            ])

if args.append:
    print(f"Results appended: instance_id {start_instance_id} to {start_instance_id + n_trees - 1}")
    print(f"Seed {args.seed} was automatically advanced to generate new independent scenarios.")
    print(f"To continue: use '--append' flag with same parameters")
else:
    print(f"Fresh results written. To continue appending later, use '--append' flag.")
