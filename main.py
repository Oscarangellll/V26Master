import argparse

from config import CaseConfig, ScenarioConfig
from models import WeatherModel, PriceModel

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
    
    weather_model = WeatherModel()
    price_model = PriceModel()

    scenario = ScenarioConfig(case, weather_model, price_model, scenarios=[1])
        
    down = scenario.make_downtime_costs()
    print(down)

    exit()
    model = OptimizationModel(case, scenario)
    
    model.build_model()
    # model.model.setParam("OutputFlag", 0)
    # model.optimize()
    
    # for (h, b, t), var in model.gamma_ST.items():
    #     if var.X > 0:
    #         print(f"{h}{b}{t}")

elif args.method == "grasp":
    print("grasp")
