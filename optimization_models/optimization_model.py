import csv
from datetime import datetime
import os

import gurobipy as gp

from config import CaseConfig, ScenarioConfig

class OptimizationModel:
    def __init__(self, case: CaseConfig, scenario: ScenarioConfig):
        self.case = case
        self.scenario = scenario
    
    def build_model(self):
        
        model = gp.Model()
        model.setParam("OutputFlag", 0)  # silent mode

        # First stage sets
        H = self.case.H
        H_S = self.case.H_S
        H_M = self.case.H_M
        V = self.case.V
        B = self.case.B
        T = self.case.T
    
        # First stage parameters
        C_ST = self.case.C_ST
        C_LT = self.case.C_LT
        C_B = self.case.C_B
        K_MAX = self.case.K_MAX
        K_REQ = self.case.K_REQ

        # First stage variables
        gamma_ST = {}
        gamma_LT = {}
        for b in B:
            for t in T:
                for h in self.case.vessel_types:
                    if h.multiday:
                        gamma_ST[h.name, b, t] = model.addVar(ub=self.case.n_vessels_ub_ST_multi)
                        
                    else:
                        gamma_ST[h.name, b, t] = model.addVar(ub=self.case.n_vessels_ub_ST_single)
        for b in B:
            for h in self.case.vessel_types:
                if h.multiday:
                    gamma_LT[h.name, b] = model.addVar(ub=self.case.n_vessels_ub_LT_multi)
                else:
                    gamma_LT[h.name, b] = model.addVar(ub=self.case.n_vessels_ub_LT_single)

        alpha = model.addVars(
            ((v, b, t) 
            for h in H_M 
            for v in V[h] 
            for b in B 
            for t in T),
            vtype=gp.GRB.BINARY
        )

        eta = model.addVars(B, vtype=gp.GRB.BINARY)

        # First stage objective
        first_obj = (
            # ST charter costs
            gp.quicksum(C_ST[h, t] * gamma_ST[h, b, t] 
                for h in H 
                for b in B 
                for t in T
            )
            # LT charter costs
            + gp.quicksum(C_LT[h] * gamma_LT[h, b] 
                for h in H 
                for b in B
            )
            # Base costs
            + gp.quicksum(C_B[b] * eta[b] 
                for b in B
            )
        )

        # First stage constraints
        model.addConstrs(
            (gp.quicksum(K_REQ[h] * (gamma_ST[h, b, t] + gamma_LT[h, b]) for h in H) <= K_MAX[b] * eta[b]
            for b in B
            for t in T),
            name="base_capacity"
        )

        model.addConstrs(
            (gp.quicksum(alpha[v, b, t] for v in V[h]) <= gamma_ST[h, b, t] + gamma_LT[h, b]
            for h in H_M 
            for b in B 
            for t in T),
            name="binding_charter_vars"
        )

        model.addConstrs(
            (gp.quicksum(alpha[v, b, t] for b in B) <= 1
            for h in H_M
            for v in V[h]
            for t in T),
            name="allocate_to_one_base"
        )
        
        model.addConstrs(
            (gp.quicksum(alpha[v1, b, t] for b in B) >= gp.quicksum(alpha[v2, b, t] for b in B)
            for h in H_M
            for (v1, v2) in zip(V[h], V[h][1:])
            for t in T),
            name=f"symmetry_break_ST"
        )

        # Second stage sets
        W = self.case.W 
        L = self.case.L 
        M = self.case.M
        D = self.case.D
        D_t = self.case.D_t
        D_T = self.case.D_T
        K_S = self.scenario.K_S
        K_M = self.scenario.K_M
        S = self.scenario.S
        
        # Second stage parameters
        F = self.scenario.F
        N = self.case.N
        P = self.scenario.P
        C_D = self.scenario.C_D
        C_RT = self.case.C_RT 
        C_T = self.case.C_T
        R = self.case.R
        
        U = self.case.U

        # Second stage variables
        x = model.addVars(H_S, B, W, D, S, vtype=gp.GRB.INTEGER)

        delta = model.addVars(
            ((v, i, d, s)
            for h in H_M 
            for v in V[h] 
            for i in L 
            for d in D 
            for s in S),
            vtype=gp.GRB.BINARY
        )

        lmbd_S = model.addVars(
            ((h, b, w, d, k, s)     
            for h in H_S 
            for b in B 
            for w in W 
            for d in D 
            for s in S 
            for k in K_S[h, b, w, d, s]),
            vtype=gp.GRB.INTEGER
        )

        lmbd_M = model.addVars(
            ((h, w, d, k, s) 
            for h in H_M 
            for w in W 
            for d in D 
            for s in S 
            for k in K_M[h, w, d, s]),
            vtype=gp.GRB.INTEGER,
        )

        z = model.addVars(W, M, D, S, vtype=gp.GRB.INTEGER)

        b = model.addVars(W, M, [0] + D, S, vtype=gp.GRB.INTEGER)

        f = model.addVars(
            ((v, i, j, d, s) 
            for h in H_M 
            for v in V[h] 
            for i in L for j in L if i!=j 
            for d in D 
            for s in S),
            vtype=gp.GRB.BINARY,
        )

        r_S = model.addVars(
            ((v, b, d, s) 
            for h in H_M 
            for v in V[h] 
            for b in B 
            for d in D_T 
            for s in S),
            vtype=gp.GRB.BINARY,
        )

        r_E = model.addVars(
            ((v, b, d, s) 
            for h in H_M 
            for v in V[h] 
            for b in B 
            for d in D_T 
            for s in S),
            vtype=gp.GRB.BINARY,
        )

        # Second-stage objective
        second_obj = (
            # Downtime costs
            gp.quicksum(C_D[w, d, s] * b[w, m, d, s]
                for w in W 
                for m in M 
                for d in D 
                for s in S)
            # Travel cost singleday vessels
            + gp.quicksum(C_RT[h, b, w] * x[h, b, w, d, s] 
                for h in H_S 
                for b in B 
                for w in W 
                for d in D 
                for s in S)
            # Travel cost multiday vessels
            + gp.quicksum(C_T[h, i, j] * f[v, i, j, d, s] 
                for h in H_M 
                for v in V[h] 
                for i in L for j in L if i != j 
                for d in D 
                for s in S)
        ) / len(S)

        # Second stage constraints
        model.addConstrs(
            (gp.quicksum(x[h, b, w, d, s] for w in W) <= gamma_ST[h, b, t] + gamma_LT[h, b]
            for h in H_S
            for b in B
            for t in T
            for d in D_t[t]
            for s in S),
            name="vessels_available"
        )
    
        model.addConstrs(
            (gp.quicksum(delta[v, i, d, s] for i in L) <= gp.quicksum(alpha[v, b, t] for b in B)
            for h in H_M
            for v in V[h]
            for t in T
            for d in D_t[t]
            for s in S),
            name="M_vessels_available"
        )

        model.addConstrs(
            (delta[v, b, d, s] <= alpha[v, b, t]
            for h in H_M 
            for v in V[h]
            for b in B
            for t in T
            for d in D_t[t]
            for s in S),
            name="deactivate_unused_deltas"
        )
        
        model.addConstrs(
            (delta[v, b, d, s] == alpha[v, b, t]
            for h in H_M
            for v in V[h]
            for b in B
            for t in T
            for d in D_t[t] if d % R[h] == 0
            for s in S),
            name="base_visit"
        )

        model.addConstrs(
            (delta[v, b, 1, s] == alpha[v, b, T[0]]
            for h in H_M
            for v in V[h]
            for b in B
            for s in S),
            name="base_visit_day_first"
        )

        model.addConstrs(
            (delta[v, b, D[-1], s] == alpha[v, b, T[-1]]
            for h in H_M
            for v in V[h]
            for b in B
            for s in S),
            name="base_visit_day_last"
        )

        model.addConstrs(
            (gp.quicksum(lmbd_S[h, b, w, d, k, s] for k in K_S[h, b, w, d, s]) <= N[h] * x[h, b, w, d, s]
            for h in H_S 
            for w in W
            for b in B
            for d in D
            for s in S),
            name="pattern_performance_constraint_singleday"
        )

        model.addConstrs(
            (gp.quicksum(lmbd_M[h, w, d, k, s] for k in K_M[h, w, d, s]) <= gp.quicksum(N[h] * delta[v, w, d, s] for v in V[h])
            for h in H_M 
            for w in W
            for d in D
            for s in S),
            name="pattern_performance_constraint_multiday"
        )

        model.addConstrs(
            (z[w, m, d, s] <= gp.quicksum(P[m, k] * lmbd_M[h, w, d, k, s] for h in H_M for k in K_M[h, w, d, s]) + gp.quicksum(P[m, k] * lmbd_S[h, b, w, d, k, s] for h in H_S for b in B for k in K_S[h, b, w, d, s])
            for w in W
            for m in M
            for d in D
            for s in S),
            name="tasks_performed"
        )

        model.addConstrs(
            (b[w, m, d, s] == b[w, m, d-1, s] + F[w, m, d, s] - z[w, m, d, s]
            for w in W
            for m in M
            for d in D
            for s in S),
            name="backlog"
        )

        model.addConstrs(
            (b[w, m, 0, s] == 0
            for w in W
            for m in M
            for s in S),
            name="init_backlog"
        )

        model.addConstrs(
            (delta[v, i, d-1, s] + gp.quicksum(f[v, j, i, d-U[(h, i, j)], s] for j in L if (i!=j and d-U[(h, i, j)]>0)) - gp.quicksum(f[v, i, j, d-1, s] for j in L if i!=j) == delta[v, i, d, s]
            for h in H_M
            for v in V[h]
            for i in L
            for d in D if d!=1 and d not in D_T
            for s in S),
            name="flow"
        )

        model.addConstrs(
            (delta[v, w, d-1, s] + gp.quicksum(f[v, j, w, d-U[(h, w, j)], s] for j in L if (w!=j and d-U[(h, w, j)]>0)) - gp.quicksum(f[v, w, j, d-1, s] for j in L if w!=j) == delta[v, w, d, s]
            for h in H_M
            for v in V[h]
            for w in W
            for d in D if d in D_T
            for s in S),
            name="flow_transition_windfarm"
        )

        model.addConstrs(
            (delta[v, b, d-1, s] + gp.quicksum(f[v, j, b, d-U[(h, b, j)], s] for j in L if (b!=j and d-U[(h, b, j)]>0)) - gp.quicksum(f[v, b, j, d-1, s] for j in L if b!=j) == delta[v, b, d, s] + r_E[v, b, d, s] - r_S[v, b, d, s]
            for h in H_M
            for v in V[h]
            for b in B
            for d in D if d in D_T
            for s in S),
            name="flow_transition_base"
        )

        model.addConstrs(
            (gp.quicksum(r_S[v, b, d, s] for b in B) <= gp.quicksum(alpha[v, b, T[t]] for b in B)
            for h in H_M 
            for v in V[h] 
            for t in range(1, len(T))
            for d in D_t[T[t]] if d in D_T
            for s in S),
            name="ST_START_transition1"
        )

        model.addConstrs(
            (gp.quicksum(r_S[v, b, d, s] for b in B) <= 1 - gp.quicksum(alpha[v, b, T[t-1]] for b in B)
            for h in H_M 
            for v in V[h] 
            for t in range(1, len(T))
            for d in D_t[T[t]] if d in D_T
            for s in S),
            name="ST_START_transition2"
        )

        model.addConstrs(
            (gp.quicksum(r_E[v, b, d, s] for b in B) <= gp.quicksum(alpha[v, b, T[t-1]] for b in B)
            for h in H_M 
            for v in V[h] 
            for t in range(1, len(T))
            for d in D_t[T[t]] if d in D_T
            for s in S),
            name="ST_END_transition1"
        )

        model.addConstrs(
            (gp.quicksum(r_E[v, b, d, s] for b in B) <= 1 - gp.quicksum(alpha[v, b, T[t]] for b in B)
            for h in H_M 
            for v in V[h] 
            for t in range(1, len(T))
            for d in D_t[T[t]] if d in D_T
            for s in S),
            name="ST_END_transition2"
        )

        model.addConstrs(
            (r_S[v, b, d, s] <= delta[v, b, d, s]
            for h in H_M
            for v in V[h]
            for b in B
            for d in D_T
            for s in S),
            name="ST_START_base"
        )

        model.addConstrs(
            (r_E[v, b, d, s] <= delta[v, b, d-1, s]
            for h in H_M
            for v in V[h]
            for b in B
            for d in D_T
            for s in S),
            name="ST_END_base"
        ) 
        
        model.setObjective(first_obj + second_obj)
        
        if self.case.one_base:
            model.addConstr(
                gp.quicksum(eta[b] for b in B) <= 1
            )
            
        # #second stage only:
        # model.addConstr(
        #     (eta["1"] == 1)
        # )
        # model.addConstrs(
        #     (gamma_ST[h, b, t] == 0
        #     for h in H
        #     for b in B
        #     for t in T)
        # )
        # model.addConstr(
        #     (gamma_LT["CTV", "1"] == 2)
        # )
        # model.addConstr(
        #     (gamma_LT["SOV", "1"] == 4)
        # )
        # model.addConstrs(
        #     (gamma_LT["CTV", b] == 0
        #     for b in ["2", "3"])
        # )

        model.update()
        
        self.model = model
        
        self.gamma_ST = gamma_ST
        self.gamma_LT = gamma_LT
        self.alpha = alpha
        self.eta = eta
        self.x = x
        self.delta = delta
        self.lmbd_S = lmbd_S
        self.lmbd_M = lmbd_M
        self.z = z
        self.b = b
        self.f = f
        self.r_S = r_S
        self.r_E = r_E
    
    def optimize(self):
        self.model.optimize()

    def print_variables(self):
        #print active gamma variables
        for (h, b, t), var in self.gamma_ST.items():
            if var.X > 0:
                print(f"gamma_ST[{h}, {b}, {t}] = {var.X}")
        for (h, b), var in self.gamma_LT.items():
            if var.X > 0:
                print(f"gamma_LT[{h}, {b}] = {var.X}")
        # print active f variables
        for (v, i, j, d, s), var in self.f.items():
            if var.X > 0:
                print(f"f[{v}, {i}, {j}, {d}, {s}] = {var.X}")
        #print active delta variables
        for (v, i, d, s), var in self.delta.items():
            if var.X > 0:
                print(f"delta[{v}, {i}, {d}, {s}] = {var.X}")
                
    def report_to_csv(self, resultspath, instance=1, runtime=None, write_header=False):
        """Save a summary row of the solved model to a CSV file."""

        case = self.case
        scenario = self.scenario
        model = self.model

        # --- Case identification ---
        case_id = (
            f"W{len(case.W)}_B{len(case.B)}_V{case.n_vessels_ub_LT_multi + case.n_vessels_ub_ST_multi}"
            f"_S{len(scenario.scenarios)}_T{len(case.T)}"
        )

        # --- Active eta (base decisions) ---
        active_bases = [b for b in case.B if self.eta[b].X > 0.5]

        # --- Active gamma_LT ---
        lt_parts = []
        for (h, b), var in self.gamma_LT.items():
            if var.X > 0.5:
                lt_parts.append(f"{h}@{b}:{int(round(var.X))}")
        gamma_lt_str = ", ".join(lt_parts) if lt_parts else "none"

        # --- Active gamma_ST (per period) ---
        st_parts = []
        for t in case.T:
            period_parts = []
            for h in case.H:
                for b in case.B:
                    val = self.gamma_ST[h, b, t].X
                    if val > 0.5:
                        period_parts.append(f"{h}@{b}:{int(round(val))}")
            if period_parts:
                st_parts.append(f"{t}|{';'.join(period_parts)}")
        gamma_st_str = ", ".join(st_parts) if st_parts else "none"

        # --- Build row ---
        row = {
            "case_id": case_id,
            "case_name": str(case.name),
            "coalition": case.coalition,
            "n_scenarios": len(scenario.scenarios),
            "instance": instance,
            "objective": model.ObjVal if model.SolCount > 0 else None,
            "mip_gap": model.MIPGap if model.SolCount > 0 else None,
            "runtime": round(runtime, 2) if runtime is not None else round(model.Runtime, 2),
            "n_variables": model.NumVars,
            "n_constraints": model.NumConstrs,
            "base_decision": ",".join(active_bases) if active_bases else "none",
            "gamma_LT_decision": gamma_lt_str,
            "gamma_ST_decision": gamma_st_str,
            "wind_farms": ",".join(case.W),
            "bases": ",".join(case.B),
            "scenario_seeds": ",".join(str(s) for s in scenario.scenarios),
            "n_periods": len(case.T),
            "days_per_period": case.days_per_period,
            "one_base": case.one_base,
            "n_vessels_ub_ST_multi": case.n_vessels_ub_ST_multi,
            "n_vessels_ub_ST_single": case.n_vessels_ub_ST_single,
            "n_vessels_ub_LT_multi": case.n_vessels_ub_LT_multi,
            "n_vessels_ub_LT_single": case.n_vessels_ub_LT_single,

        }

        # --- Write (create or append) ---
        from pathlib import Path
        Path(resultspath).parent.mkdir(parents=True, exist_ok=True)

        mode = "w" if write_header else "a"
        with open(resultspath, mode, newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def update_fixed_decisions(self, fixed_decisions, *, strict=True, use_start=False):
        """
        fixed_decisions: dict[tuple[str, tuple], int|float]
        f.eks. {("eta", ("1",)): 1, ("gamma_LT", ("SOV","1")): 4, ("gamma_ST", ("CTV","1",2)): 0}
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        for (group, key), value in fixed_decisions.items():
            try:
                vardict = getattr(self, group)  # e.g. self.eta, self.gamma_LT
            except AttributeError:
                if strict:
                    raise ValueError(f"Unknown variable group: {group}")
                continue

            try:
                var = vardict[key]  # tupledict supports tuple keys
            except KeyError:
                if strict:
                    raise KeyError(f"Variable not found: {group}{key}")
                continue

            var.LB = value
            var.UB = value
            if use_start:
                var.Start = value

        self.model.update()

    def get_solution(self, which="all", *, include_zero=True, tol=1e-9):
        """
        Retrieve solution values.

        which:
            - "all": returns dict[varName -> value] for all vars in model
            - "first_stage": returns only gamma_ST, gamma_LT, alpha, eta
            - "second_stage": returns x, delta, lmbd_S, lmbd_M, z, b, f, r_S, r_E

        include_zero:
            If False: only include variables with |X| > tol (much smaller dicts)
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        # Ensure we actually have a solution
        if self.model.SolCount == 0:
            raise RuntimeError(f"No solution available. Status={self.model.Status}")

        def extract(var_dict):
            out = {}
            for key, var in var_dict.items():
                val = var.X
                if include_zero or abs(val) > tol:
                    out[var.VarName] = val
            return out

        if which == "all":
            if include_zero:
                return {v.VarName: v.X for v in self.model.getVars()}
            else:
                return {v.VarName: v.X for v in self.model.getVars() if abs(v.X) > tol}

        elif which == "first_stage":
            sol = {}
            sol.update(extract(self.gamma_ST))
            sol.update(extract(self.gamma_LT))
            sol.update(extract(self.alpha))
            sol.update(extract(self.eta))
            return sol

        elif which == "second_stage":
            sol = {}
            sol.update(extract(self.x))
            sol.update(extract(self.delta))
            sol.update(extract(self.lmbd_S))
            sol.update(extract(self.lmbd_M))
            sol.update(extract(self.z))
            sol.update(extract(self.b))
            sol.update(extract(self.f))
            sol.update(extract(self.r_S))
            sol.update(extract(self.r_E))
            return sol

        else:
            raise ValueError("which must be one of: 'all', 'first_stage', 'second_stage'.")
