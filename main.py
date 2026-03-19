
from config.case_config import CaseConfig
from config.scenario_config import ScenarioConfig

from optimization_models.optimization_model import OptimizationModel

case = CaseConfig("cases/1W1B.yaml")
var_groups = ["gamma_ST", "gamma_LT", "alpha", "eta"]

scenario = ScenarioConfig(case, [10])

model = OptimizationModel(case, scenario)
model.Params.OutputFlag = 0
# model.addConstr(model.gamma_LT["SOV", "3"] == 0, name="fix_gamma_LT_1")
# # model.addConstr(model.gamma_LT["CTV", "3"] == 0, name="fix_gamma_LT_0")
# model.addConstrs(
#     (model.gamma_ST[h,"3", t] == 0 for h in model.case.H for t in case.periods),
#     name="fix_gamma_ST_0"
# )

model.optimize()
print(f"Objective value: {model.ObjVal}")
print(f"First stage cost: {model.first_obj.getValue()}")
print(f"Second stage cost: {model.second_obj.getValue()}")
print(f"Charter cost ST: {model.charter_cost_ST.getValue()}")
print(f"Charter cost LT: {model.charter_cost_LT.getValue()}")
print(f"Downtime cost: {model.downtime_cost.getValue()}")
print(f"Travel cost S: {model.travel_cost_S.getValue()}")
print(f"Travel cost M: {model.travel_cost_M.getValue()}")
solution = frozenset(
    ((var_group, key), int(var.X))
    for var_group in var_groups
    for key, var in getattr(model, var_group).items()
)
for (var_group, key), val in solution:
    if val > 0.5:
        print(var_group, key, val)
#plot model.b across the time periods for each maintenance category:
import matplotlib.pyplot as plt
for m in model.case.maintenance_categories:
    b_values = [model.b[w.name, m.name, d, s].X for d in model.case.D for w in model.case.wind_farms for s in model.S]
    plt.plot(model.case.D, b_values, label=m)
plt.xlabel("Time Period")
plt.ylabel("b value")
plt.title("b values over time for each maintenance category")
plt.legend()
plt.show()




# print("Fixing and resolving with true distribution")

# scenario = ScenarioConfig(case, range(21, 101))

# model = OptimizationModel(case, scenario)
# for (var_group, key), val in solution:
#     var = getattr(model, var_group)[key]
#     var.LB = val
#     var.UB = val

# model.Params.OutputFlag = 0
# model.optimize()
# print(f"True solution: {model.ObjVal}")

