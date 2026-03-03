import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
from config.case_config import CaseConfig
from config.scenario_config import ScenarioConfig
from scenario_models.weather_model import WeatherModel
from scenario_models.price_model import PriceModel
from optimization_models.optimization_model import OptimizationModel
from optimization_models.consensus_model_multiprocessing import ConsensusModel

def get_coalitions(items):
    """Return all non-empty subsets of items."""
    coalitions = []
    for r in range(1, len(items) + 1):
        for combo in combinations(items, r):
            #sort combo alphabetically to ensure consistent coalition naming
            coalitions.append(sorted(list(combo)))

    #sort coalitions by length (smallest to largest) and then alphabetically
    coalitions.sort(key=lambda x: (len(x), x))

    return coalitions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m", "--method",
        required=True,
        choices=["mip", "con"],
        help="Solution method"
    )

    parser.add_argument(
        "-c", "--case",
        required=True,
        help="Case name"
    )

    parser.add_argument(
        "-n", "--n_instances",
        type=int,
        required=True,
        help="Number of instances to solve"
    )

    parser.add_argument(
        "-s", "--n_scenarios",
        type=int,
        required=True,
        help="Number of scenarios to sample for each instance"
    )

    parser.add_argument(
        "--i", "--iterate_coalitions",
        action="store_true",
        help="Whether to iterate over coalitions"
    )

    args = parser.parse_args()
    case_path = Path(f"cases/{args.case}.yaml")

    full_case = CaseConfig(case_path=case_path)
    full_case_wind_farm_names = [w.name for w in full_case.wind_farms]
    if args.i:
        coalitions = get_coalitions(full_case_wind_farm_names)
        print(coalitions)
        folder = "coalitions"
    else:
        coalitions = [full_case_wind_farm_names]
        folder = "full_case_only"

    weather_model = WeatherModel()
    price_model = PriceModel()

    resultspath = Path("results") / "Case" / args.case / args.method / folder / (Path(f"{args.n_instances}N{args.n_scenarios}S").stem + ".csv")

    master_seed = 22
    master_rng = np.random.default_rng(master_seed)

    # Pre-generate scenario seeds so every coalition uses the same ones
    scenario_seeds = [  
        master_rng.choice(
            np.arange(1, 1000),
            size=args.n_scenarios, 
            replace=False
        ) for _ in range(args.n_instances)
    ]

    first_row = True
    for coalition in coalitions:
        case = CaseConfig(case_path=case_path, wind_farm_names=coalition)
        
        for instance in range(1, args.n_instances + 1):
            
            if args.method == "mip":
            
                scenarios = scenario_seeds[instance - 1]

                scenario = ScenarioConfig(case, weather_model, price_model, scenarios=scenarios)

                model = OptimizationModel(case, scenario)
                        
                model.build_model()
                
                model.model.setParam("OutputFlag", 0)
                model.model.setParam("MIPGap", 0.01) # 1% optimality gap
                model.model.setParam("TimeLimit", 14400) # 4 hours time limit per instance
                
                model.model.optimize()
                
                model.report_to_csv(resultspath, instance=instance, runtime=None, write_header=first_row)
                first_row = False
            
            elif args.method == "con":
                judge_seeds = scenario_seeds[instance - 1]
                master_scenarios = judge_seeds[:]

                cm = ConsensusModel(
                    case,
                    judge_seeds_1scenario_each=judge_seeds,
                    weather_model=weather_model,
                    price_model=price_model,
                    mip_gap_judges=0.01,
                    log=False,
                )

                model, fix, runtime = cm.optimize(
                    master_scenarios=master_scenarios,
                    eta_max_iters=50,
                    lt_max_iters=200,
                    top_k_eta=1,
                    top_k_lt=1,
                    min_p=0.6,
                    max_p=0.95,
                    aggregator="mean",
                    tighten_ub_st=True,
                    unanim_fix_zero_st=True,
                    mip_gap_master=0.002,
                )
                
                model.report_to_csv(resultspath, instance=instance, runtime=runtime, write_header=first_row)
                first_row = False 
