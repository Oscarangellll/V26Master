import argparse
from collections import Counter
import csv
from pathlib import Path

import numpy as np

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

args = parser.parse_args()

case = CaseConfig(f"cases/{args.case}.yaml")

rng = np.random.default_rng(seed=99)

scenario_tree_sizes = args.scenario_tree_sizes
n_trees = args.n_trees


cache = {st_size: Counter() for st_size in scenario_tree_sizes}

var_groups = ["eta", "gamma_LT", "gamma_ST", "alpha"]

results_path = Path("results/stability") / args.case / args.method / "ISS.csv"
results_path.parent.mkdir(parents=True, exist_ok=True)

with results_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "tree_size", "objective", "runtime", "MIPGap", "gamma_LT", "gamma_ST", "scenarios"
    ])

for st_size in scenario_tree_sizes:
    for j in range(n_trees):
        
        s = rng.choice(np.arange(1, 101), size=st_size, replace=False)    
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
            
        gamma_LT_str = ";".join(f"{key}:{val}"
            for (var_group, key), val in solution
            if var_group == "gamma_LT" and val > 0)

        gamma_ST_str = ";".join(f"{key}:{val}"
            for (var_group, key), val in solution
            if var_group == "gamma_ST" and val > 0)
            
        cache[st_size][solution] += 1
        
        with results_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([   
                st_size, 
                model.ObjVal, 
                runtime,
                model.MIPGap,
                gamma_LT_str,
                gamma_ST_str,
                ",".join(map(str, s))
            ])



results_path = Path("results/stability") / args.case / args.method / "OSS.csv"
results_path.parent.mkdir(parents=True, exist_ok=True)

with results_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "tree_size", "count", "objective", "gamma_LT", "gamma_ST"
    ])

# The true distribution is the same for all solutions
true_distribution = np.arange(101, 301)
scenario = ScenarioConfig(case, true_distribution)

model = OptimizationModel(case, scenario)

def evaluate_solution(solution):
    for (group, key), val in solution:
        var = getattr(model, group)[key]
        var.LB = val
        var.UB = val
        
    model.optimize()

    return model.ObjVal 


evaluated_solutions = {}
for tree_size, counter in cache.items():
    
    for solution, count in counter.items():
        if solution in evaluated_solutions:
            obj = evaluated_solutions[solution]
        else:
            obj = evaluate_solution(solution)
            evaluated_solutions[solution] = obj

        gamma_LT_str = ";".join(f"{key}:{val}"
            for (var_group, key), val in solution
            if var_group == "gamma_LT" and val > 0)

        gamma_ST_str = ";".join(f"{key}:{val}"
            for (var_group, key), val in solution
            if var_group == "gamma_ST" and val > 0)
        
        with results_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([tree_size, count, obj, gamma_LT_str, gamma_ST_str])
