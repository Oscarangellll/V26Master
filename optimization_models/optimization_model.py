
import gurobipy as gp

class OptimizationModel:
    def __init__(self, case, scenario):
        
        self.case = case
        self.scenario = scenario

        model = gp.Model()
        
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
        gamma_ST = model.addVars(H, B, T, vtype=gp.GRB.INTEGER)
        gamma_LT = model.addVars(H, B, vtype=gp.GRB.INTEGER)
        
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
        charter_cost_ST = gp.quicksum(
            C_ST[h, t] * gamma_ST[h, b, t]
            for h in H
            for b in B
            for t in T
        )
        charter_cost_LT = gp.quicksum(
            C_LT[h] * gamma_LT[h, b]
            for h in H
            for b in B
        )
        base_cost = gp.quicksum(
            C_B[b] * eta[b]
            for b in B
        )
        first_obj = charter_cost_ST + charter_cost_LT + base_cost

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

        if self.case.one_base:
            model.addConstr(
                gp.quicksum(eta[b] for b in B) <= 1
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
        downtime_cost = gp.quicksum(
            C_D[w, d, s] * b[w, m, d, s]
            for w in W 
            for m in M 
            for d in D
            for s in S
        ) / len(S)
        travel_cost_S = gp.quicksum(
            C_RT[h, b, w] * x[h, b, w, d, s] 
            for h in H_S
            for b in B
            for w in W
            for d in D
            for s in S
        ) / len(S)
        travel_cost_M = gp.quicksum(
            C_T[h, i, j] * f[v, i, j, d, s] 
            for h in H_M 
            for v in V[h] 
            for i in L for j in L if i != j 
            for d in D
            for s in S
        ) / len(S)
        second_obj = downtime_cost + travel_cost_S + travel_cost_M

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

        
        self.model = model

        self.gamma_ST = gamma_ST
        self.gamma_LT = gamma_LT
        self.alpha = alpha
        self.eta = eta

        self.charter_cost_ST = charter_cost_ST
        self.charter_cost_LT = charter_cost_LT
        self.base_cost = base_cost
        self.first_obj = first_obj



    def __getattr__(self, name):
        return getattr(self.model, name)
