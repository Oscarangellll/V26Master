import argparse
import random
from itertools import combinations
from pathlib import Path

from config import CaseConfig, ScenarioConfig
from models import WeatherModel, PriceModel

from model.optimization_model import OptimizationModel
from model.consensus import ConsensusModel

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
    help="Path to case config"
)

parser.add_argument(
    "-n", "--n_instances",
    type=int,
    required=True,
    help="Number of instances to solve (with different scenario seeds)"
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

random.seed(98621454)

full_case = CaseConfig(case_path=case_path)
all_wind_farms = [w.name for w in full_case.wind_farms]
if args.i:
    coalitions = get_coalitions(all_wind_farms)
    folder = "coalitions"
else:
    coalitions = [all_wind_farms]
    folder = "full_case_only"

weather_model = WeatherModel()
price_model = PriceModel()

resultspath = Path("results") / "Case" / args.case / args.method / folder / (Path(f"{args.n_instances}N{args.n_scenarios}S").stem + ".csv")

# Pre-generate scenario seeds so every coalition uses the same ones
scenario_seeds = [random.sample(range(1, 1000), args.n_scenarios) for _ in range(args.n_instances)]

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
            model.model.setParam("MIPGap", 0.00002) # 0.2% optimality gap
            # model.model.setParam("Threads", 1) # use a single thread for more consistent runtimes across different machines
            #print model thread in use
            print(f"Using {model.model.params.Threads} thread(s) for optimization")
            model.model.optimize()
            print(f"Coalition {''.join(coalition)}, Instance {instance}: scenario seeds = {scenarios}")
            
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
                log=True,
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
        
        else:
            print("method not recognized")
