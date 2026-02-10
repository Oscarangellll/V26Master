import argparse

from config.case_config import CaseConfig
from config.scenario_config import ScenarioConfig
from model.optimization_model import OptimizationModel

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

args = parser.parse_args()

if args.method == "mip":
    case = CaseConfig(args.case)
    
    scenario = ScenarioConfig(case, scenarios=[1, 2])

    model = OptimizationModel(case, scenario)
    model.build_model()
    model.optimize()
    
    for (h, b, t), var in model.gamma_ST.items():
        if var.X > 0:
            print(f"{h}{b}{t}")

elif args.method == "grasp":
    print("grasp")
