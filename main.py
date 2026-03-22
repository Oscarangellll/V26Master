
from config.case_config import CaseConfig
from config.scenario_config import ScenarioConfig

from optimization_models.consensus_model import ConsensusModel
from optimization_models.consensus_model_multiprocessing import ConsensusModelMP
from optimization_models.optimization_model import OptimizationModel

case = CaseConfig("cases/1W1B.yaml")
var_groups = ["gamma_ST", "gamma_LT", "alpha", "eta"]
method = "con" # "mip" or "con"
scenario_list = [50, 51, 52, 53, 54, 55]

scenario = ScenarioConfig(case, scenario_list)

if method == "mip":
    model = OptimizationModel(case, scenario, scenario_list)
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0.02
    model.optimize()

elif method == "con":
    judge_seeds = scenario_list
    master_scenarios = judge_seeds[:]
    
    cm = ConsensusModel(
        case, 
        scenario, 
        judge_seeds_1scenario_each=judge_seeds,
        mip_gap_judges=0.2,
        log=False
    )
    
    model, runtime = cm.optimize(
        master_scenarios=master_scenarios,
        eta_max_iters=50,
        lt_max_iters=200,
        top_k_eta=1,
        top_k_lt=1,
        min_p=0.55,
        max_p=0.95,
        aggregator="mean",
        tighten_ub_st=True,
        unanim_fix_zero_st=True,
        mip_gap_master=0.02
    )
elif method == "con_mp":
    judge_seeds = scenario_list
    master_scenarios = judge_seeds[:]
    
    cm = ConsensusModel(
        case, 
        scenario, 
        judge_seeds_1scenario_each=judge_seeds,
        mip_gap_judges=0.2,
        log=False
    )
    
    model, runtime = cm.optimize(
        master_scenarios=master_scenarios,
        eta_max_iters=50,
        lt_max_iters=200,
        top_k_eta=1,
        top_k_lt=1,
        min_p=0.55,
        max_p=0.95,
        aggregator="mean",
        tighten_ub_st=True,
        unanim_fix_zero_st=True,
        mip_gap_master=0.02
    )
    
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
        #dont print alpha variables
        if not (var_group == "alpha"):
            print(var_group, key, val)

print("Fixing and resolving with true distribution")


def _fix_solution(eval_model, fixed_solution):
    for group_name in ["eta", "gamma_LT", "gamma_ST", "alpha"]:
        for _, var in getattr(eval_model, group_name).items():
            var.LB = 0
            var.UB = 0

    for (group_name, key), value in fixed_solution:
        var = getattr(eval_model, group_name)[key]
        var.LB = value
        var.UB = value


# Mini ad-hoc OOS evaluation of the current solution on 100 scenarios.
oos_pool_start = 200
oos_n_scenarios = 100
oos_scenarios = list(range(oos_pool_start, oos_pool_start + oos_n_scenarios))

totals = {
    "objective": 0.0,
    "first_stage_cost": 0.0,
    "second_stage_cost": 0.0,
    "charter_cost_ST": 0.0,
    "charter_cost_LT": 0.0,
    "charter_cost_mob": 0.0,
    "downtime_cost": 0.0,
    "travel_cost_S": 0.0,
    "travel_cost_M": 0.0,
    "runtime": 0.0,
    "MIPGap": 0.0,
}

for scenario_id in oos_scenarios:
    eval_scenario_ids = [scenario_id]
    eval_scenario = ScenarioConfig(case, eval_scenario_ids)
    eval_model = OptimizationModel(case, eval_scenario, eval_scenario_ids)
    eval_model.Params.OutputFlag = 0
    eval_model.Params.MIPGap = 0.02

    _fix_solution(eval_model, solution)
    eval_model.optimize()

    totals["objective"] += eval_model.ObjVal
    totals["first_stage_cost"] += eval_model.first_obj.getValue()
    totals["second_stage_cost"] += eval_model.second_obj.getValue()
    totals["charter_cost_ST"] += eval_model.charter_cost_ST.getValue()
    totals["charter_cost_LT"] += eval_model.charter_cost_LT.getValue()
    totals["charter_cost_mob"] += eval_model.charter_cost_mob.getValue()
    totals["downtime_cost"] += eval_model.downtime_cost.getValue()
    totals["travel_cost_S"] += eval_model.travel_cost_S.getValue()
    totals["travel_cost_M"] += eval_model.travel_cost_M.getValue()
    totals["runtime"] += eval_model.Runtime
    totals["MIPGap"] += eval_model.MIPGap

n_oos = len(oos_scenarios)
avg = {key: value / n_oos for key, value in totals.items()}

print("\n=== Mini ad-hoc OOS (fixed first-stage solution) ===")
print(f"Scenarios evaluated: {oos_scenarios[0]}-{oos_scenarios[-1]} (n={n_oos})")
print(f"Mean objective: {avg['objective']}")
print(f"Mean first stage cost: {avg['first_stage_cost']}")
print(f"Mean second stage cost: {avg['second_stage_cost']}")
print(f"Mean charter cost ST: {avg['charter_cost_ST']}")
print(f"Mean charter cost LT: {avg['charter_cost_LT']}")
print(f"Mean charter cost mob: {avg['charter_cost_mob']}")
print(f"Mean downtime cost: {avg['downtime_cost']}")
print(f"Mean travel cost S: {avg['travel_cost_S']}")
print(f"Mean travel cost M: {avg['travel_cost_M']}")
print(f"Mean runtime: {avg['runtime']}")
print(f"Mean MIPGap: {avg['MIPGap']}")
