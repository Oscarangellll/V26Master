import argparse
import random
import time
import csv
from itertools import combinations
from pathlib import Path

from config import CaseConfig, ScenarioConfig
from models import WeatherModel, PriceModel
from model.optimization_model import OptimizationModel


# ============================================================
# Coalition helper (samme som i mip-main)
# ============================================================

def get_coalitions(items):
    """Return all non-empty subsets of items."""
    coalitions = []
    for r in range(1, len(items) + 1):
        for combo in combinations(items, r):
            coalitions.append(sorted(list(combo)))
    coalitions.sort(key=lambda x: (len(x), x))
    return coalitions


# ============================================================
# Consensus helpers
# ============================================================

def is_consistent(sol_dict, fixed_dict):
    """Check if cached solution dict agrees with all fixed decisions."""
    for k, v in fixed_dict.items():
        if sol_dict.get(k, None) != v:
            return False
    return True


def extract_first_stage_decisions(m: OptimizationModel):
    """Return dict[(group, key)->int] for first-stage vars (rounded)."""
    out = {}

    for b in m.case.B:
        out[("eta", b)] = int(round(m.eta[b].X))

    for h in m.case.H:
        for b in m.case.B:
            out[("gamma_LT", (h, b))] = int(round(m.gamma_LT[h, b].X))

    for h in m.case.H:
        for b in m.case.B:
            for t in m.case.T:
                out[("gamma_ST", (h, b, t))] = int(round(m.gamma_ST[h, b, t].X))

    return out


def build_models(case, judges, weather_model, price_model, mip_gap, output_flag):
    """Build one OptimizationModel per judge (each has 1 scenario)."""
    models = {}
    for judge in judges:
        scenario = ScenarioConfig(case, weather_model, price_model, scenarios=list(judge))
        m = OptimizationModel(case, scenario)
        m.build_model()
        m.model.setParam("OutputFlag", output_flag)
        m.model.setParam("MIPGap", mip_gap)
        models[judge] = m
    return models


def solve_or_reuse(models, fixed, prev_sol):
    """Solve each judge model unless cached solution matches fixed."""
    for judge, m in models.items():
        if judge in prev_sol and is_consistent(prev_sol[judge], fixed):
            continue
        m.update_fixed_decisions(fixed, strict=True, use_start=True)
        m.model.optimize()
        if m.model.SolCount == 0:
            raise RuntimeError(f"No solution for judge {judge}. Status={m.model.Status}")
        prev_sol[judge] = extract_first_stage_decisions(m)


def frac_agree(vals, candidate):
    return vals.count(candidate) / len(vals)


def threshold_fix(group, unfixed_set, prev_sol, judges, fixed, threshold):
    """Fix vars in group when agreement >= threshold. Returns count fixed."""
    n_fixed = 0
    for key in list(unfixed_set):
        k = (group, key)
        vals = [prev_sol[j][k] for j in judges]
        maj = max(set(vals), key=vals.count)
        if frac_agree(vals, maj) >= threshold:
            fixed[k] = maj
            unfixed_set.remove(key)
            n_fixed += 1
    return n_fixed


def fix_zero_gamma_ST(unfixed_set, prev_sol, judges, fixed):
    """Fix gamma_ST to 0 only if ALL judges say 0. Returns count fixed."""
    n_fixed = 0
    for key in list(unfixed_set):
        k = ("gamma_ST", key)
        vals = [prev_sol[j][k] for j in judges]
        if all(v == 0 for v in vals):
            fixed[k] = 0
            unfixed_set.remove(key)
            n_fixed += 1
    return n_fixed


def run_consensus_for_case(case, judge_seeds_1scenario_each, master_scenarios, weather_model, price_model,
                           *, mip_gap_judges=0.008, mip_gap_master=0.001, output_flag=0):
    """
    Runs Step A/B on judge models (each has 1 scenario),
    then Step C master solve with master_scenarios (multi-scenario).
    Returns (master_model, master_first_stage_dict).
    """

    judges = [(s,) for s in judge_seeds_1scenario_each]  # tuple seeds for dict keys

    # Build judge models
    models = build_models(case, judges, weather_model, price_model, mip_gap_judges, output_flag)

    fixed = {}
    prev_sol = {}

    unfixed = {
        "eta": {b for b in case.B},
        "gamma_LT": {(h, b) for h in case.H for b in case.B},
        "gamma_ST": {(h, b, t) for h in case.H for b in case.B for t in case.T},
    }

    # Initial solve
    solve_or_reuse(models, fixed, prev_sol)

    # -------------------------
    # Step A: eta + gamma_LT
    # threshold starts at 1.0 and drops by 0.1 if no progress
    # -------------------------
    thr = 1.0
    while thr >= 0.0 and (unfixed["eta"] or unfixed["gamma_LT"]):
        nA = 0
        nA += threshold_fix("eta", unfixed["eta"], prev_sol, judges, fixed, thr)
        nA += threshold_fix("gamma_LT", unfixed["gamma_LT"], prev_sol, judges, fixed, thr)

        if nA > 0:
            solve_or_reuse(models, fixed, prev_sol)
        else:
            thr = round(thr - 0.1, 10)

    # -------------------------
    # Step B: only-fix-zero for gamma_ST
    # iterate until no new zeros fixed
    # -------------------------
    while True:
        nB = fix_zero_gamma_ST(unfixed["gamma_ST"], prev_sol, judges, fixed)
        if nB == 0:
            break
        solve_or_reuse(models, fixed, prev_sol)

    # -------------------------
    # Step C: master solve (multi-scenario)
    # -------------------------
    master_scenario_cfg = ScenarioConfig(case, weather_model, price_model, scenarios=master_scenarios)
    master = OptimizationModel(case, master_scenario_cfg)
    master.build_model()
    master.model.setParam("OutputFlag", output_flag)
    master.model.setParam("MIPGap", mip_gap_master)

    # apply fixed from A+B (eta, gamma_LT, forced-zero gamma_ST)
    master.update_fixed_decisions(fixed, strict=True, use_start=True)
    master.model.optimize()
    if master.model.SolCount == 0:
        raise RuntimeError(f"Master solve: no solution. Status={master.model.Status}")

    master_first = extract_first_stage_decisions(master)
    return master, master_first


# ============================================================
# CSV reporting (samme kolonne-struktur som report_to_csv)
# men runtime = wall-clock for consensus-run
# ============================================================

def report_consensus_row(filename, case, scenario_seeds, instance, obj, mip_gap,
                        runtime_wall, n_vars, n_constrs,
                        base_decision, gamma_lt_str, gamma_st_str,
                         *, write_header=False):

    row = {
        "case_id": f"W{len(case.W)}_B{len(case.B)}_V{case.max_multiday_vessels}_S{len(scenario_seeds)}_T{len(case.T)}",
        "case_name": str(case.name),
        "coalition": case.coalition,
        "n_scenarios": len(scenario_seeds),
        "instance": instance,
        "objective": obj,
        "mip_gap": mip_gap,
        "runtime": round(runtime_wall, 2),
        "n_variables": n_vars,
        "n_constraints": n_constrs,
        "base_decision": base_decision,
        "gamma_LT_decision": gamma_lt_str,
        "gamma_ST_decision": gamma_st_str,
        "wind_farms": ",".join(case.W),
        "bases": ",".join(case.B),
        "max_multiday_vessels": case.max_multiday_vessels,
        "scenario_seeds": ",".join(str(s) for s in scenario_seeds),
        "n_periods": len(case.T),
        "days_per_period": case.days_per_period,
        "one_base": case.one_base,
        "n_vessels_ub_ST": case.n_vessels_ub_ST,
        "n_vessels_ub_LT": case.n_vessels_ub_LT,
    }

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"

    with open(filename, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def summarize_master_solution(master: OptimizationModel, case):
    # base decision
    active_bases = [b for b in case.B if master.eta[b].X > 0.5]
    base_decision = ",".join(active_bases) if active_bases else "none"

    # gamma_LT string
    lt_parts = []
    for (h, b), var in master.gamma_LT.items():
        if var.X > 0.5:
            lt_parts.append(f"{h}@{b}:{int(round(var.X))}")
    gamma_lt_str = ", ".join(lt_parts) if lt_parts else "none"

    # gamma_ST string
    st_parts = []
    for t in case.T:
        period_parts = []
        for h in case.H:
            for b in case.B:
                val = master.gamma_ST[h, b, t].X
                if val > 0.5:
                    period_parts.append(f"{h}@{b}:{int(round(val))}")
        if period_parts:
            st_parts.append(f"{t}|{';'.join(period_parts)}")
    gamma_st_str = ", ".join(st_parts) if st_parts else "none"

    return base_decision, gamma_lt_str, gamma_st_str


# ============================================================
# MAIN (matcher MIP-CLI)
# ============================================================

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

if args.method != "con":
    raise RuntimeError("Use -m con for this file.")

random.seed(98621454)

# Load full case to discover all wind farms (samme som mip-main)
full_case = CaseConfig(args.case)
all_wind_farms = [w.name for w in full_case.wind_farms]

if args.i:
    coalitions = get_coalitions(all_wind_farms)
else:
    coalitions = [all_wind_farms]

# Separate results filename so it doesn't overwrite MIP
results_file = Path(args.case).stem + "_consensus.csv"

# Pre-generate scenario seeds so every coalition uses the same ones (samme som mip-main)
# NOTE: For consensus we use these seeds in two roles:
#   - judges: single-scenario models that "vote" (we take the first K seeds)
#   - master: multi-scenario solve (we use ALL seeds)
scenario_seeds = [random.sample(range(1, 1000), args.n_scenarios) for _ in range(args.n_instances)]

weather_model = WeatherModel()
price_model = PriceModel()

first_row = True

for coalition in coalitions:

    case = CaseConfig(args.case, wind_farm_names=coalition)

    for instance in range(1, args.n_instances + 1):

        seeds = scenario_seeds[instance - 1]  # length = args.n_scenarios

        # In consensus:
        # - judges each solve 1 scenario: we use all seeds (one per judge)
        # - master sees all seeds simultaneously
        judge_seeds = seeds
        master_scenarios = seeds

        t0 = time.perf_counter()

        master, _ = run_consensus_for_case(
            case,
            judge_seeds_1scenario_each=judge_seeds,
            master_scenarios=master_scenarios,
            weather_model=weather_model,
            price_model=price_model,
            mip_gap_judges=0.008,
            mip_gap_master=0.002,
            output_flag=0
        )

        t1 = time.perf_counter()
        runtime_wall = t1 - t0

        base_decision, gamma_lt_str, gamma_st_str = summarize_master_solution(master, case)

        report_consensus_row(
            results_file,
            case,
            scenario_seeds=seeds,
            instance=instance,
            obj=master.model.ObjVal,
            mip_gap=master.model.MIPGap,
            runtime_wall=runtime_wall,
            n_vars=master.model.NumVars,
            n_constrs=master.model.NumConstrs,
            base_decision=base_decision,
            gamma_lt_str=gamma_lt_str,
            gamma_st_str=gamma_st_str,
            write_header=first_row
        )

        first_row = False

        print(f"Coalition {''.join(coalition)}, Instance {instance}: scenario seeds = {seeds}")

print("Saved:", results_file)
