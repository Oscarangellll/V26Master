"""
Diagnostic 2: 
  (a) Check if gamma_ST / gamma_LT are fractional (missing vtype=INTEGER bug)
  (b) Run nested scenarios (1S ⊂ 2S ⊂ 4S) to test monotonicity cleanly
"""
import sys, time
import numpy as np
from pathlib import Path
from config.case_config import CaseConfig
from config.scenario_config import ScenarioConfig
from scenario_models.weather_model import WeatherModel
from scenario_models.price_model import PriceModel
from optimization_models.optimization_model import OptimizationModel


def decompose_objective(om):
    case = om.case
    scen = om.scenario
    S = om.scenario_ids

    H = case.H; H_S = case.H_S; H_M = case.H_M
    B = case.B; T = case.T; V = case.V
    W = case.W; M = case.M; D = case.D; L = case.L

    C_ST = case.C_ST; C_LT = case.C_LT; C_B = case.C_B
    C_RT = case.C_RT; C_T = case.C_T
    C_D = dict(scen.get_CD_for_scenarios(S))

    first = 0.0
    for h in H:
        for b in B:
            for t in T:
                first += C_ST[h, t] * om.gamma_ST[h, b, t].X
    for h in H:
        for b in B:
            first += C_LT[h] * om.gamma_LT[h, b].X
    for b in B:
        first += C_B[b] * om.eta[b].X

    downtime_raw = 0.0
    travel_s_raw = 0.0
    travel_m_raw = 0.0

    for w in W:
        for m in M:
            for d in D:
                for s in S:
                    downtime_raw += C_D[w, d, s] * om.b[w, m, d, s].X

    for h in H_S:
        for b in B:
            for w in W:
                for d in D:
                    for s in S:
                        travel_s_raw += C_RT[h, b, w] * om.x[h, b, w, d, s].X

    for h in H_M:
        for v in V[h]:
            for i in L:
                for j in L:
                    if i != j:
                        for d in D:
                            for s in S:
                                travel_m_raw += C_T[h, i, j] * om.f[v, i, j, d, s].X

    second_raw = downtime_raw + travel_s_raw + travel_m_raw
    second_avg = second_raw / len(S)

    return {
        "first_stage": first,
        "second_raw": second_raw,
        "second_avg": second_avg,
        "downtime_avg": downtime_raw / len(S),
        "travel_s_avg": travel_s_raw / len(S),
        "travel_m_avg": travel_m_raw / len(S),
        "total_obj": first + second_avg,
        "gurobi_obj": om.model.ObjVal,
    }


def check_fractional_gammas(om):
    """Check whether charter variables gamma_ST/gamma_LT are fractional."""
    tol = 1e-6
    fractional = []
    for (h, b, t), var in om.gamma_ST.items():
        val = var.X
        if abs(val - round(val)) > tol:
            fractional.append(("gamma_ST", h, b, t, val))
    for (h, b), var in om.gamma_LT.items():
        val = var.X
        if abs(val - round(val)) > tol:
            fractional.append(("gamma_LT", h, b, None, val))
    return fractional


def print_all_gammas(om):
    """Print all non-zero gamma values."""
    print("    gamma_ST values:")
    for (h, b, t), var in sorted(om.gamma_ST.items()):
        if var.X > 1e-9:
            int_tag = "" if abs(var.X - round(var.X)) < 1e-6 else " *** FRACTIONAL ***"
            print(f"      gamma_ST[{h},{b},{t}] = {var.X:.6f}{int_tag}")
    print("    gamma_LT values:")
    for (h, b), var in sorted(om.gamma_LT.items()):
        if var.X > 1e-9:
            int_tag = "" if abs(var.X - round(var.X)) < 1e-6 else " *** FRACTIONAL ***"
            print(f"      gamma_LT[{h},{b}] = {var.X:.6f}{int_tag}")


def main():
    case_path = Path("cases/1W3B.yaml")
    case = CaseConfig(case_path=case_path)
    wm = WeatherModel()
    pm = PriceModel()

    master_seed = 22
    n_instances = 5
    scenario_counts = [1, 2, 4]

    # =====================================================================
    # PART A: Check for fractional gammas using standard (non-nested) seeds
    # =====================================================================
    print("=" * 70)
    print("PART A: Checking for FRACTIONAL gamma_ST / gamma_LT values")
    print("=" * 70)

    rng = np.random.default_rng(master_seed)
    seeds_4s = [rng.choice(np.arange(1, 1000), size=4, replace=False) for _ in range(3)]

    for inst_idx, seeds in enumerate(seeds_4s):
        sc = ScenarioConfig(case, wm, pm, scenarios=seeds)
        om = OptimizationModel(case, sc, list(seeds))
        om.build_model()
        om.model.setParam("OutputFlag", 0)
        om.model.setParam("MIPGap", 0.002)
        om.model.optimize()

        frac = check_fractional_gammas(om)
        print(f"\n  Instance {inst_idx+1}, seeds={list(seeds)}")
        print_all_gammas(om)
        if frac:
            print(f"  *** FOUND {len(frac)} FRACTIONAL GAMMA(S)! ***")
            for item in frac:
                print(f"      {item}")
        else:
            print(f"  All gammas are integer-valued.")

    # =====================================================================
    # PART B: Nested scenario test 
    #         For each instance, draw 4 seeds. Then solve 1S, 2S, 4S using
    #         seeds[0:1], seeds[0:2], seeds[0:4] — purely nested.
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART B: NESTED SCENARIO TEST (1S ⊂ 2S ⊂ 4S)")
    print("=" * 70)

    rng2 = np.random.default_rng(master_seed)
    nested_seeds = [rng2.choice(np.arange(1, 1000), size=max(scenario_counts), replace=False)
                    for _ in range(n_instances)]

    results = {n_s: {"first": [], "second": [], "total": [], "downtime": []}
               for n_s in scenario_counts}

    for inst in range(n_instances):
        base_seeds = nested_seeds[inst]
        print(f"\n  Instance {inst+1}, base_seeds = {list(base_seeds)}")

        for n_s in scenario_counts:
            seeds = base_seeds[:n_s]
            sc = ScenarioConfig(case, wm, pm, scenarios=seeds)
            om = OptimizationModel(case, sc, list(seeds))
            om.build_model()
            om.model.setParam("OutputFlag", 0)
            om.model.setParam("MIPGap", 0.002)
            om.model.setParam("TimeLimit", 14400)

            t0 = time.time()
            om.model.optimize()
            elapsed = time.time() - t0

            d = decompose_objective(om)

            results[n_s]["first"].append(d["first_stage"])
            results[n_s]["second"].append(d["second_avg"])
            results[n_s]["total"].append(d["total_obj"])
            results[n_s]["downtime"].append(d["downtime_avg"])

            frac = check_fractional_gammas(om)
            frac_tag = f" FRAC={len(frac)}" if frac else ""

            print(f"    {n_s}S: first={d['first_stage']:>12,.0f}  "
                  f"sec_avg={d['second_avg']:>12,.0f}  "
                  f"total={d['total_obj']:>12,.0f}  "
                  f"gurobi={d['gurobi_obj']:>12,.0f}  "
                  f"t={elapsed:.1f}s{frac_tag}")

    # Summary
    print("\n" + "=" * 70)
    print("PART B SUMMARY: Nested averages over", n_instances, "instances")
    print("=" * 70)
    print(f"  {'|S|':>4}  {'Avg First':>14}  {'Avg SecondAvg':>14}  {'Avg Total':>14}  {'Monotonicity':>14}")
    prev_total = None
    for n_s in scenario_counts:
        avg_f = np.mean(results[n_s]["first"])
        avg_s = np.mean(results[n_s]["second"])
        avg_t = np.mean(results[n_s]["total"])
        mono = ""
        if prev_total is not None:
            mono = "OK (↑)" if avg_t >= prev_total else "VIOLATION (↓)"
        print(f"  {n_s:>4}  {avg_f:>14,.0f}  {avg_s:>14,.0f}  {avg_t:>14,.0f}  {mono}")
        prev_total = avg_t

    # Per-instance nested comparison
    print("\n  Per-instance total objectives (nested):")
    print(f"  {'Inst':>4}", end="")
    for n_s in scenario_counts:
        print(f"  {n_s}S Total".rjust(14), end="")
    print("  Mono?")
    for i in range(n_instances):
        print(f"  {i+1:>4}", end="")
        vals = []
        for n_s in scenario_counts:
            v = results[n_s]["total"][i]
            vals.append(v)
            print(f"  {v:>14,.0f}", end="")
        is_mono = all(vals[j] <= vals[j+1] for j in range(len(vals)-1))
        print(f"  {'YES' if is_mono else 'NO'}")


if __name__ == "__main__":
    main()
