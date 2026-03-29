
from config import CaseConfig, ScenarioConfig
from optimization_models import OptimizationModel

case = CaseConfig("cases/3W3B.yaml")
scenario = ScenarioConfig(case, [10, 11])

model = OptimizationModel(case, scenario, [10], weights={10: 0.5})

model.Params.MIPGap = 0.02
model.Params.Threads = 1

model.optimize()

for key, var in model.gamma_LT.items():
    print(key, var.X)
for key, var in model.gamma_ST.items():
    print(key, var.X)
print(model.ObjVal)
