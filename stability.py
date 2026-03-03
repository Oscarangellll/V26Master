import argparse
from collections import Counter
import csv
from pathlib import Path

import numpy as np

from config import CaseConfig, ScenarioConfig
from scenario_models import PriceModel, WeatherModel
from optimization_models.optimization_model1 import OptimizationModel

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

weather_model = WeatherModel()
price_model = PriceModel()

case = CaseConfig(f"cases/{args.case}.yaml")

master_seed = 30
master_rng = np.random.default_rng(master_seed)

scenario_tree_sizes = args.scenario_tree_sizes
n_trees = args.n_trees

# Each scenario tree size gets its own seed to ensure
# that the trees with different sizes are independent
scenario_tree_size_seeds = master_rng.choice(10_000, size=len(scenario_tree_sizes), replace=False)

cache = {st_size: Counter() for st_size in scenario_tree_sizes}

var_groups = ["eta", "gamma_LT", "gamma_ST", "alpha"]

results_path = Path("results/stability") / args.case / args.method / "ISS.csv"
results_path.parent.mkdir(parents=True, exist_ok=True)

with results_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "tree_size", "objective", "runtime", "MIPGap"
    ])

for i, scenario_tree_size in enumerate(scenario_tree_sizes):
     
    rng = np.random.default_rng(scenario_tree_size_seeds[i]) 
    for j in range(n_trees):
        
        # These scenario seeds are unique within a single model
        s = rng.choice(100, size=scenario_tree_size, replace=False)    
        
        scenario = ScenarioConfig(case, weather_model, price_model, s)

        model = OptimizationModel(case, scenario)

        model.Params.OutputFlag = 0
        model.Params.MIPGap = 0.01

        model.optimize()

        solution = frozenset(
            ((var_group, key), int(var.X))
            for var_group in var_groups
            for key, var in getattr(model, var_group).items()
        )
        
        cache[scenario_tree_size][solution] += 1
        
        with results_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([   
                scenario_tree_size, 
                model.model.ObjVal, 
                model.model.Runtime, 
                model.model.MIPGap
            ])


unique_counts = {
    st_size: len(counter)
    for st_size, counter in cache.items()
}
print(unique_counts)
for st_size, counter in cache.items():
    print(f"\nTree size {st_size}")
    for sol, count in counter.most_common():
        print(f"  count={count}")

results_path = Path("results/stability") / args.case / args.method / "OSS.csv"
results_path.parent.mkdir(parents=True, exist_ok=True)

with results_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "tree_size", "count", "objective"
    ])

# The true distribution is the same for all solutions
true_distribution = master_rng.choice(np.arange(101, 1_000), size=50, replace=False)
def evaluate_solution(solution):
    obj = 0
    for s in true_distribution:
        scenario = ScenarioConfig(case, weather_model, price_model, [s])
    
        model = OptimizationModel(case, scenario)

        for (group, key), val in solution:
            var = getattr(model, group)[key]
            var.LB = val
            var.UB = val
        model.Params.OutputFlag = 0
        model.Params.MIPGap = 0.01

        model.optimize()
        obj += model.ObjVal

    return obj / len(true_distribution) 


evaluated_solutions = {}
for tree_size, counter in cache.items():
    
    for solution, count in counter.items():
        if solution in evaluated_solutions:
            obj = evaluated_solutions[solution]
        else:
            obj = evaluate_solution(solution)
            evaluated_solutions[solution] = obj
    
        with results_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([tree_size, count, obj])
