"""
Diagnostic: decompose the MIP objective into first-stage and second-stage costs
for different scenario counts, to pinpoint the source of non-monotonicity.
"""
import sys, time
import numpy as np
from pathlib import Path
from config.case_config import CaseConfig
from config.scenario_config import ScenarioConfig
from scenario_models.weather_model import WeatherModel
from scenario_models.price_model import PriceModel
from optimization_models.optimization_model import OptimizationModel

def decompose_objective(om: OptimizationModel):
    """After solving, compute first-stage cost and second-stage cost separately."""
    case = om.case
    scen = om.scenario
    S = om.scenario_ids

    H = case.H; H_S = case.H_S; H_M = case.H_M
    B = case.B; T = case.T; V = case.V
    W = case.W; M = case.M; D = case.D; L = case.L

    C_ST = case.C_ST; C_LT = case.C_LT; C_B = case.C_B
    C_RT = case.C_RT; C_T = case.C_T
    C_D = dict(scen.get_CD_for_scenarios(S))

    # First stage cost
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

    # Second stage cost (the raw sum over scenarios, before dividing by |S|)
    downtime_raw = 0.0
    travel_single_raw = 0.0
    travel_multi_raw = 0.0

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
                        travel_single_raw += C_RT[h, b, w] * om.x[h, b, w, d, s].X

    for h in H_M:
        for v in V[h]:
            for i in L:
                for j in L:
                    if i != j:
                        for d in D:
                            for s in S:
                                travel_multi_raw += C_T[h, i, j] * om.f[v, i, j, d, s].X

    second_raw = downtime_raw + travel_single_raw + travel_multi_raw
    second_avg = second_raw / len(S)

    return {
        "first_stage": first,
        "second_raw": second_raw,
        "second_avg": second_avg,
        "downtime_raw": downtime_raw,
        "travel_single_raw": travel_single_raw,
        "travel_multi_raw": travel_multi_raw,
        "total_obj": first + second_avg,
        "gurobi_obj": om.model.ObjVal,
    }


def main():
    case_path = Path("cases/1W3B.yaml")
    case = CaseConfig(case_path=case_path)
    wm = WeatherModel()
    pm = PriceModel()
    master_seed = 22
    master_rng = np.random.default_rng(master_seed)

    n_instances = 5  # quick diagnostic
    scenario_counts = [1, 2, 4]

    # Pre-generate seeds for max scenario count across instances
    max_s = max(scenario_counts)
    all_seeds = [
        master_rng.choice(np.arange(1, 1000), size=max_s, replace=False)
        for _ in range(n_instances)
    ]

    for n_s in scenario_counts:
        print(f"\n{'='*70}")
        print(f"  SCENARIO COUNT = {n_s}")
        print(f"{'='*70}")

        # Regenerate seeds the same way main.py does (separate RNG)
        rng_for_this = np.random.default_rng(master_seed)
        seeds_for_this = [
            rng_for_this.choice(np.arange(1, 1000), size=n_s, replace=False)
            for _ in range(n_instances)
        ]

        firsts = []; seconds = []; totals = []; downtimes = []
        for inst in range(n_instances):
            seeds = seeds_for_this[inst]
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
            firsts.append(d["first_stage"])
            seconds.append(d["second_avg"])
            totals.append(d["total_obj"])
            downtimes.append(d["downtime_raw"] / len(seeds))

            print(f"  inst={inst+1}  seeds={list(seeds)}")
            print(f"    first_stage   = {d['first_stage']:>14,.0f}")
            print(f"    second_avg    = {d['second_avg']:>14,.0f}")
            print(f"    total (ours)  = {d['total_obj']:>14,.0f}")
            print(f"    gurobi obj    = {d['gurobi_obj']:>14,.0f}")
            print(f"    downtime_avg  = {d['downtime_raw']/len(seeds):>14,.0f}")
            print(f"    travel_s_avg  = {d['travel_single_raw']/len(seeds):>14,.0f}")
            print(f"    travel_m_avg  = {d['travel_multi_raw']/len(seeds):>14,.0f}")
            print(f"    runtime       = {elapsed:.1f}s")
            
            # Also print first-stage decisions
            bases = [b for b in case.B if om.eta[b].X > 0.5]
            lt_parts = []
            for (h, b), var in om.gamma_LT.items():
                if var.X > 0.5:
                    lt_parts.append(f"{h}@{b}:{int(round(var.X))}")
            print(f"    bases={bases}  LT={lt_parts}")

        print(f"\n  --- AVERAGES for {n_s}S (n={n_instances}) ---")
        print(f"    avg first_stage = {np.mean(firsts):>14,.0f}  (std={np.std(firsts):>10,.0f})")
        print(f"    avg second_avg  = {np.mean(seconds):>14,.0f}  (std={np.std(seconds):>10,.0f})")
        print(f"    avg total       = {np.mean(totals):>14,.0f}  (std={np.std(totals):>10,.0f})")
        print(f"    avg downtime    = {np.mean(downtimes):>14,.0f}  (std={np.std(downtimes):>10,.0f})")


if __name__ == "__main__":
    main()
