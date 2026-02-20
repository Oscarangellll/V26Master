import argparse
import random
from itertools import combinations
from pathlib import Path

from sympy import group

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
    choices=["mip", "con"],
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

# --------------------------------------
# Helpers
# --------------------------------------
def majority_vote(values):
    # values: list of ints (etter rounding)
    return max(set(values), key=values.count)

def frac_agree(values, candidate):
    return values.count(candidate) / len(values)

def get_val(model, group, key):
    # model: OptimizationModel instance
    return getattr(model, group)[key].X

def is_consistent(sol_dict, fixed_dict):
    # fixed_dict: dict[(group,key)->value]
    # sol_dict:   dict[(group,key)->value]
    for k, v in fixed_dict.items():
        if sol_dict.get(k, None) != v:
            return False
    return True

def extract_first_stage_decisions(m: OptimizationModel, tol=1e-6):
    # returnerer dict[(group,key)->int]
    out = {}
    # eta (binary)
    for b in m.case.B:
        out[("eta", b)] = int(round(m.eta[b].X))
    # gamma_LT (integer)
    for h in m.case.H:
        for b in m.case.B:
            out[("gamma_LT", (h,b))] = int(round(m.gamma_LT[h,b].X))
    # gamma_ST (integer)
    for h in m.case.H:
        for b in m.case.B:
            for t in m.case.T:
                out[("gamma_ST", (h,b,t))] = int(round(m.gamma_ST[h,b,t].X))
    return out

# --------------------------------------
# Consensus Fixing
# --------------------------------------
if args.method == "con":
    threshold = 0.8
    random.seed(9862145)
    print("Entered consensus method with threshold", threshold)
    
    weather_model = WeatherModel()
    price_model = PriceModel()
    
    full_case = CaseConfig(args.case)
    
    all_wind_farms = [w.name for w in full_case.wind_farms]
    coalitions = get_coalitions(all_wind_farms) if args.i else [all_wind_farms]
        
    judges = [tuple(random.sample(range(1, 1000), 1)) for _ in range(args.n_scenarios)] 
    print(f"Judges (scenario seeds): {judges}")
    
    for coalition in coalitions:
        case = CaseConfig(args.case, wind_farm_names=coalition)
        unfixed = {
            "eta": {b for b in case.B},
            "gamma_LT": {(h, b) for h in case.H for b in case.B},
            "gamma_ST": {(h, b, t) for h in case.H for b in case.B for t in case.T},
            # kanskje legge til "alpha" 
        }
        fixed = {}  # dict[(group, keytuple) -> value]
        prev_sol = {}
        
        models = {}
        for judge in judges:
            scenario = ScenarioConfig(case, weather_model, price_model, scenarios=list(judge))
            m = OptimizationModel(case, scenario)
            m.build_model()
            m.model.setParam("OutputFlag", 0)
            m.model.setParam("MIPGap", 0.002) # 0.2% optimality gap
            models[judge] = m
            print("Model built for judge (scenario seed) ", judge)
        i = 1
        max_iterations = 10
        while any(unfixed[v_group] for v_group in unfixed) and i <= max_iterations:
            print(f"\n--- Iteration {i} ---")
            # 1) Solve or reuse per judge
            for judge, m in models.items():
                if judge in prev_sol and is_consistent(prev_sol[judge], fixed):
                    # gjenbruk løsning – IKKE re-solve
                    continue
                else:
                    m.update_fixed_decisions(fixed, strict=True, use_start=True)
                    m.model.optimize()
                    prev_sol[judge] = extract_first_stage_decisions(m)
                print(f"Iteration {i}, Judge {judge}, Obj {m.model.ObjVal}, Fixed {len(fixed)}, Unfixed {sum(len(s) for s in unfixed.values())}")
            
            # 2) Consensus-fixing per gruppe
            n_fixed_this_i = 0
            for v_group in ["eta","gamma_LT","gamma_ST"]:
                for key in list(unfixed[v_group]):
                    k = (v_group, key)
                    vals = [prev_sol[j][k] for j in models.keys()]
                    maj = max(set(vals), key=vals.count)
                    if vals.count(maj)/len(vals) >= threshold:
                        fixed[k] = maj
                        unfixed[v_group].remove(key)
                        n_fixed_this_i += 1
            print(f"After iteration {i}, fixed variables: {fixed}")
            if n_fixed_this_i == 0:
                threshold -= 0.1
                print(f"No variables fixed in iteration {i}. Lowering threshold to {threshold}")
            i += 1
        
    print(f"\nCoalition={coalition}")
    print(f"Iterations={i-1}")
    print(f"Fixed vars={len(fixed)}")
    if any(unfixed[g] for g in unfixed):
        print("Stopped early (no progress). Remaining unfixed sizes:",
            {g: len(unfixed[g]) for g in unfixed})
    else:
        print("All targeted first-stage vars fixed.")
    #print all active fixed first-stage decisions (those that were fixed to 1 or >0)
    print("Active fixed first-stage decisions:")
    for (group, key), value in fixed.items():
        if value > 0:
            print(f"  {group}: {key} = {value}")

    # give estimation of solution on 5 new scenarios for the coalition, using the fixed decisions and re-solving each judge's model with those decisions fixed
    print("\nEvaluating consensus solution on new scenarios:")
    random.seed(98621454)
    new_scenarios = [tuple(random.sample(range(1, 1000), 1)) for _ in range(args.n_scenarios)]
    sol = 0
    for scenario in new_scenarios:
        s = ScenarioConfig(case, weather_model, price_model, scenarios=list(scenario))
        print(f"Scenario seed: {scenario}")
        m = OptimizationModel(case, s)
        m.build_model()
        m.model.setParam("OutputFlag", 0)
        m.update_fixed_decisions(fixed, strict=True, use_start=True)
        m.model.optimize()
        print(f"Obj {m.model.ObjVal}")
        sol += m.model.ObjVal
    print(f"Average objective over new scenarios: {sol/len(new_scenarios)}")