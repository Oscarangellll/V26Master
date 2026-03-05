import argparse
from collections import Counter
import csv
from pathlib import Path

import numpy as np

from config import CaseConfig, ScenarioConfig
from scenario_models import PriceModel, WeatherModel
from optimization_models import OptimizationModel, ConsensusModel

if __name__ == "__main__":

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

    master_seed = 50 
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

    file_exists = results_path.exists()
    mode = "a" if file_exists else "w"

    with results_path.open(mode, newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "tree_size", "objective", "runtime", "MIPGap"
            ])

    for i, scenario_tree_size in enumerate(scenario_tree_sizes):
         
        rng = np.random.default_rng(scenario_tree_size_seeds[i]) 
        for j in range(n_trees):
            
            s = rng.choice(1000, size=scenario_tree_size, replace=False)    
            scenario = ScenarioConfig(case, s)

            if args.method == "mip":
                model = OptimizationModel(case, scenario, s)
               
                model.build_model()
                model.model.Params.OutputFlag = 0
                model.model.Params.TimeLimit = 7200
                model.model.Params.MIPGap = 0.01

                model.optimize()

                runtime = model.model.Runtime

            elif args.method == "con":
                judge_seeds = scenario.S 
                master_scenarios = judge_seeds[:]
                
                cm = ConsensusModel(
                    case,
                    scenario,
                    judge_seeds_1scenario_each=judge_seeds,
                    mip_gap_judges = 0.01,
                )
                
                model, runtime = cm.optimize(master_scenarios)

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
                    runtime,
                    model.model.MIPGap
                ])


    results_path = Path("results/stability") / args.case / args.method / "OSS.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = results_path.exists()
    mode = "a" if file_exists else "w"

    with results_path.open(mode, newline="") as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                "tree_size", "count", "objective", "gamma_LT", "gamma_ST"
            ])

    # The true distribution is the same for all solutions
    true_distribution = master_rng.choice(np.arange(1_001, 10_000), size=100, replace=False)
    scenario = ScenarioConfig(case, true_distribution)
    def evaluate_solution(solution):
        obj = 0
        for s in true_distribution:
            
            model = OptimizationModel(case, scenario, [s])
            model.build_model()

            for (group, key), val in solution:
                var = getattr(model, group)[key]
                var.LB = val
                var.UB = val
                model.model.Params.OutputFlag = 0
            model.model.Params.TimeLimit = 7200
            model.model.Params.MIPGap = 0.01

            model.optimize()
            obj += model.model.ObjVal

        return obj / len(true_distribution) 


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
                



