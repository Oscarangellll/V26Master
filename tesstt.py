import numpy as np

from config import CaseConfig, ScenarioConfig
from optimization_models import OptimizationModel

case = CaseConfig("cases/mini.yaml")

rng = np.random.default_rng(seed=400)

for st in [4]:
    print(f"Scenario size {st}")

    for i in range(10):
        s = rng.choice(1_000, size=st, replace=False)
    
        scenario = ScenarioConfig(case, s)

        model = OptimizationModel(case, scenario, s)

        model.build_model()
        model.model.Params.OutputFlag = 0
        model.optimize()
        print(model.model.ObjVal)
