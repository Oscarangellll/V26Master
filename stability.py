import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from config import BaseCaseConfig, CaseConfig, ScenarioConfig
from ml_models import PriceModel, WeatherModel
from optimization_models import OptimizationModel

parser = argparse.ArgumentParser()

parser.add_argument(
    "-c", "--case_path",
    required=True,
    help="Path to base case YAML file"
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

scenario_tree_sizes = args.scenario_tree_sizes
n_trees = args.n_trees

base_case = BaseCaseConfig(args.case_path)
case = CaseConfig(base_case)

master_rng = np.random.default_rng(300)
scenario_seeds = master_rng.integers(0, 100, len(scenario_tree_sizes))
cache = {st_size: Counter() for st_size in scenario_tree_sizes}

var_groups = ["eta", "gamma_LT", "gamma_ST", "alpha"]

for i, scenario_tree_size in enumerate(scenario_tree_sizes):
    
    rng = np.random.default_rng(scenario_seeds[i]) 
    for j in range(n_trees):

        s = rng.choice(100, size=scenario_tree_size, replace=False)    
        
        scenario = ScenarioConfig(case, weather_model, price_model, s)

        model = OptimizationModel(case, scenario)
        model.build_model()
        model.model.Params.OutputFlag = 0

        model.optimize()
        solution = frozenset(
            ((var_group, key), int(var.X))
            for var_group in var_groups
            for key, var in getattr(model, var_group).items()
        )


        cache[scenario_tree_size][solution] += 1


def evaluate_solution(solution, rng):
    obj = 0
    for i in range(20):
        s = rng.integers(100, 1000)
        scenario = ScenarioConfig(case, weather_model, price_model, [s])
    
        model = OptimizationModel(case, scenario)

        model.build_model()

        for (group, key), val in solution:
            var = getattr(model, group)[key]
            var.LB = val
            var.UB = val
        model.model.Params.OutputFlag = 0
        model.optimize()
        obj += model.model.ObjVal

    return obj / 20

for tree_size, counter in cache.items():
    total_count = sum(counter.values())
    weighted_sum = 0

    for solution, count in counter.items():
        obj = evaluate_solution(solution, master_rng)

        weighted_sum += count * obj

    average = weighted_sum / total_count

    print(tree_size, average)
