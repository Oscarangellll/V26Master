import multiprocessing as mp
import numpy as np
import os
import time

from config import BaseCaseConfig, CaseConfig, ScenarioConfig
from ml_models import PriceModel, WeatherModel
from optimization_models import OptimizationModel

weather_model = WeatherModel()
price_model = PriceModel()

base_case = BaseCaseConfig("base_cases/mini.yaml")
case = CaseConfig(base_case)

"""
scenario_ids = [1, 2, 3, 4]
t0 = time.perf_counter()
for s in scenario_ids:
    
    scenario = ScenarioConfig(case, weather_model, price_model, [s])

    model = OptimizationModel(case, scenario)
    model.build_model()
    model.model.Params.OutputFlag = 0
    model.model.Params.Threads = 1
    model.optimize()


t1 = time.perf_counter()
print("Time:", t1 - t0)
"""
def solve_scenario(scenario_seed):
    # 1. Generate scenario
    scenario = ScenarioConfig(case, weather_model, price_model, [scenario_seed])
    
    # 2. Create model in this process
    model = OptimizationModel(case, scenario)
    model.build_model()
    model.model.Params.OutputFlag = 0
    model.model.Params.Threads = 1   # important!
    
    # 3. Solve
    model.optimize()
    
    # 4. Return whatever you need (objective, solution, etc.)
    return model.model.ObjVal


if __name__ == "__main__":
    scenario_ids = [1, 2, 3, 4]

    t0 = time.perf_counter()
    with mp.Pool(4) as pool:  # 4 processes
        results = pool.map(solve_scenario, scenario_ids)
    t1 = time.perf_counter()
    print("Results:", results)
    print("Time:", t1 - t0)
