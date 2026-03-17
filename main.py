
from config.case_config import CaseConfig
from config.scenario_config import ScenarioConfig

from optimization_models.optimization_model import OptimizationModel

case = CaseConfig("cases/1W1B.yaml")
var_groups = ["gamma_ST", "gamma_LT", "alpha", "eta"]

scenario = ScenarioConfig(case, [6])

model = OptimizationModel(case, scenario)
model.Params.OutputFlag = 0
model.optimize()
print(f"Objective value: {model.ObjVal}")
solution = frozenset(
    ((var_group, key), int(var.X))
    for var_group in var_groups
    for key, var in getattr(model, var_group).items()
)
for (var_group, key), val in solution:
    if val > 0.5:
        print(var_group, key, val)


print("Fixing and resolving with true distribution")

scenario = ScenarioConfig(case, range(21, 101))

model = OptimizationModel(case, scenario)
for (var_group, key), val in solution:
    var = getattr(model, var_group)[key]
    var.LB = val
    var.UB = val

model.Params.OutputFlag = 0
model.optimize()
print(f"True solution: {model.ObjVal}")

