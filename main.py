import argparse
import random
from itertools import combinations
from pathlib import Path

from config import CaseConfig, ScenarioConfig
from models import WeatherModel, PriceModel

from model.optimization_model import OptimizationModel

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
    choices=["mip", "grasp"],
    help="Solution method"
)

parser.add_argument(
    "-c", "--case",
    required=True,
    help="Path to case config"
)

parser.add_argument(
    "-n", "--n-instances",
    type=int,
    required=True,
    help="Number of instances to solve (with different scenario seeds)"
)

parser.add_argument(
    "-s", "--n-scenarios",
    type=int,
    required=True,
    help="Number of scenarios to sample for each instance"
)

parser.add_argument(
    "--i", "--iterate-coalitions",
    action="store_true",
    help="Whether to iterate over coalitions"
)

args = parser.parse_args()

if args.method == "mip":
    
    random.seed(98621454)
    
    # Load full case to discover all wind farms
    full_case = CaseConfig(args.case)
    all_wind_farms = [w.name for w in full_case.wind_farms]
    if args.i:
        coalitions = get_coalitions(all_wind_farms)
    else:
        coalitions = [all_wind_farms]
    
    results_file = Path(args.case).stem + ".csv"
    
    # Pre-generate scenario seeds so every coalition uses the same ones
    scenario_seeds = [random.sample(range(1, 1000), args.n_scenarios) for _ in range(args.n_instances)]
    
    weather_model = WeatherModel()
    price_model = PriceModel()
    
    first_row = True
    for coalition in coalitions:
        case = CaseConfig(args.case, wind_farm_names=coalition)
        
        for instance in range(1, args.n_instances + 1):
            scenarios = scenario_seeds[instance - 1]

            scenario = ScenarioConfig(case, weather_model, price_model, scenarios=scenarios)

            model = OptimizationModel(case, scenario)
                    
            model.build_model()
            
            model.model.setParam("OutputFlag", 0)
            model.model.setParam("MIPGap", 0.002) # 0.2% optimality gap
            
            model.model.optimize()
            
            model.report_to_csv(results_file, instance=instance, write_header=first_row)
            first_row = False
            
            print(f"Coalition {''.join(coalition)}, Instance {instance}: scenario seeds = {scenarios}")


elif args.method == "grasp":
    print("grasp")
