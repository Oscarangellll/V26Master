from model.classes import VesselType, Vessel, Windfarm, Base, MaintenanceCategory
import gurobipy as gp
import numpy as np
from haversine import haversine, Unit

def init_model(
    name: str,
    days_per_ST_period: int,
    ST_periods_in_LT_horizon: list[str],
    vessel_types: list[VesselType],
    vessels: list[Vessel],
    windfarms: list[Windfarm],
    bases: list[Base],
    maintenance_categories: list[MaintenanceCategory],
    pattern_library: dict[tuple[str, int]: int], #(M, K)
    pattern_scenarios_S: dict[tuple[str, str, str, int, int], list[int]], # (H,B,W,D,S)
    pattern_scenarios_M: dict[tuple[str, str, int, int], list[int]], # (H,W,D,S)
    failure_scenarios: dict[tuple[str, str, int, int], int], # (W,M,D,S)
    downtime_cost_scenarios: dict[tuple[str, int, int], int], # (W,D,S)
):
    model = gp.Model()
    
    # First-stage sets
    H = [h.name for h in vessel_types]
    H_S = [h.name for h in vessel_types if not h.multiday]
    H_M = [h.name for h in vessel_types if h.multiday]
    V = {h: [v.name for v in vessels if v.vessel_type.name == h] for h in H_M}
    B = [b.name for b in bases]
    T = ST_periods_in_LT_horizon
    #First-stage parameters
    C_ST = {(h.name, t): h.calculate_ST_cost(days_per_ST_period) for h in vessel_types for t in ST_periods_in_LT_horizon}
    C_LT = {h.name: h.calculate_LT_cost(days_per_ST_period, len(ST_periods_in_LT_horizon)) for h in vessel_types}
    C_B = {b.name: b.cost for b in bases}
    KMAX = {b.name: b.max_capacity for b in bases}
    KREQ = {h.name: h.capacity_requirement for h in vessel_types}
    #First-stage desicion variables
    gamma_ST = model.addVars(H, B, T, vtype=gp.GRB.INTEGER, name="gamma_ST")
    gamma_LT = model.addVars(H, B, vtype=gp.GRB.INTEGER, ub=0, name="gamma_LT")
    alpha = model.addVars(((v, b, t) for h in H_M for v in V[h] for b in B for t in T), vtype=gp.GRB.BINARY, name="alpha")
    
    model.addConstr(
        (alpha["SOV1", "Base 1", "Jan"] == 1)
    )
    model.addConstr(
        (alpha["SOV1", "Base 1", "Feb"] == 0)
    )
    eta = model.addVars(B, vtype=gp.GRB.BINARY, lb=1, name="eta")
    #First-stage objective
    fs_obj = (gp.quicksum(C_B[b] * eta[b] for b in B) 
            + gp.quicksum(C_ST[h, t] * gamma_ST[h, b, t] for b in B for h in H for t in T) 
            + gp.quicksum(C_LT[h] * gamma_LT[h, b] for b in B for h in H))
    #First-stage constraints
    model.addConstrs(
        (gp.quicksum(KREQ[h] * (gamma_ST[h, b, t] + gamma_LT[h, b]) for h in H) <= KMAX[b] * eta[b]
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
    #Second-stage sets
    W = [w.name for w in windfarms]
    L = [i.name for i in bases + windfarms]
    M = [m.name for m in maintenance_categories]
    D = [d+1 for d in range(days_per_ST_period * len(ST_periods_in_LT_horizon))]
    D_t = {t: D[i * days_per_ST_period : (i+1) * days_per_ST_period] for i, t in enumerate(ST_periods_in_LT_horizon)} 
    D_T = [d for d in D if d % days_per_ST_period == 1 and d != 1]
    S = {s for (_,_,s) in downtime_cost_scenarios.keys()} #husk
    K_S = pattern_scenarios_S
    K_M = pattern_scenarios_M
    #Second-stage parameters
    F = failure_scenarios
    N = {h.name : h.n_teams for h in vessel_types}
    P = pattern_library
    C_D = downtime_cost_scenarios
    C_RT = {(h.name, b.name, w.name): 2*haversine((b.latitude, b.longitude), (w.latitude, w.longitude)) * h.travel_cost_per_km for h in vessel_types if h.name in H_S for b in bases for w in windfarms}
    C_T = {(h.name, i.name, j.name): 2*haversine((i.latitude, i.longitude), (j.latitude, j.longitude)) * h.travel_cost_per_km for h in vessel_types if h.name in H_M for i in bases + windfarms for j in bases + windfarms if i != j}
    R = {h.name: h.periodic_return for h in vessel_types if h.name in H_M}
    #Second-stage variables
    x = model.addVars(H_S, B, W, D, S, vtype=gp.GRB.INTEGER, name="x")
    delta = model.addVars(
        ((v, i, d, s) for h in H_M for v in V[h] for i in L for d in D for s in S),
        vtype=gp.GRB.BINARY,
        name="delta"
    )
    lmbd_S = model.addVars(
        ((h, b, w, d, k, s) for h in H_S for b in B for w in W for d in D for s in S for k in K_S[h, b, w, d, s]),
        vtype=gp.GRB.INTEGER,
        name="lambda"
    )
    lmbd_M = model.addVars(
        ((h, w, d, k, s) for h in H_M for w in W for d in D for s in S for k in K_M[h, w, d, s]),
        vtype=gp.GRB.INTEGER,
        name="lambda"
    )
    z = model.addVars(W, M, D, S, vtype=gp.GRB.INTEGER, name="z")
    b = model.addVars(W, M, [0] + D, S, vtype=gp.GRB.INTEGER, name="b")
    f = model.addVars(
        ((v, i, j, d, s) for h in H_M for v in V[h] for i in L for j in L if i!=j for d in D for s in S),
        vtype=gp.GRB.BINARY,
        name="f"
    )
    r_S = model.addVars(
        ((v, i, d, s) for h in H_M for v in V[h] for i in L for d in D_T for s in S),
        vtype=gp.GRB.BINARY,
        name="r_S"
    )
    r_E = model.addVars(
        ((v, i, d, s) for h in H_M for v in V[h] for i in L for d in D_T for s in S),
        vtype=gp.GRB.BINARY,
        name="r_E"
    )
    # Second-stage objective
    sec_obj = (
        gp.quicksum(C_D[w, d, s] * b[w, m, d, s] for w in W for m in M for d in D for s in S) + 
        gp.quicksum(C_RT[h, b, w] * x[h, b, w, d, s] for h in H_S for b in B for w in W for d in D for s in S) + 
        gp.quicksum(C_T[h, i, j] * f[v, i, j, d, s] for h in H_M for v in V[h] for i in L for j in L if i != j for d in D for s in S)
    ) / len(S)
    # Second-stage constraints
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
        (z[w, m, d, s] <= gp.quicksum(P[m,k] * lmbd_M[h, w, d, k, s] for h in H_M for k in K_M[h, w, d, s]) + gp.quicksum(P[m,k] * lmbd_S[h, b, w, d, k, s] for h in H_S for b in B for k in K_S[h, b, w, d, s])
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
    # model.addConstrs(
    #     (delta[v, i, d-1, s] + gp.quicksum(f[v, j, i, d-1, s] for j in L if i!=j) - gp.quicksum(f[v, i, j, d-1, s] for j in L if i!=j) == delta[v, i, d, s]
    #     for h in H_M
    #     for v in V[h]
    #     for i in L
    #     for d in D if d!=1 and d not in D_T
    #     for s in S),
    #     name="flow"
    # )
    # model.addConstrs(
    #     (delta[v, i, d-1, s] + gp.quicksum(f[v, j, i, d-1, s] for j in L if i!=j) - gp.quicksum(f[v, i, j, d-1, s] for j in L if i!=j) == delta[v, i, d, s]
    #     for h in H_M
    #     for v in V[h]
    #     for i in L
    #     for d in D if d in D_T
    #     for s in S),
    #     name="flow_transition"
    # )
    model.addConstrs(
        (gp.quicksum(r_S[v, i, d, s] for i in L) <= gp.quicksum(alpha[v, b, T[t]] for b in B)
        for h in H_M 
        for v in V[h] 
        for t in range(1, len(T))
        for d in D_t[T[t]] if d in D_T
        for s in S),
        name="ST_START_transition1"
    )
    model.addConstrs(
        (gp.quicksum(r_S[v, i, d, s] for i in L) <= 1 - gp.quicksum(alpha[v, b, T[t-1]] for b in B)
        for h in H_M 
        for v in V[h] 
        for t in range(1, len(T))
        for d in D_t[T[t]] if d in D_T
        for s in S),
        name="ST_START_transition2"
    )
    model.addConstrs(
        (gp.quicksum(r_E[v, i, d, s] for i in L) <= gp.quicksum(alpha[v, b, T[t-1]] for b in B)
        for h in H_M 
        for v in V[h] 
        for t in range(1, len(T))
        for d in D_t[T[t]] if d in D_T
        for s in S),
        name="ST_END_transition1"
    )
    model.addConstrs(
        (gp.quicksum(r_E[v, i, d, s] for i in L) <= 1 - gp.quicksum(alpha[v, b, T[t]] for b in B)
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
    model.addConstrs(
        (gp.quicksum(r_S[v, w, d, s] + r_E[v, w, d, s] for w in W) == 0
        for h in H_M
        for v in V[h]
        for d in D_T
        for s in S),
        name="only_transition_in_base"
    )
    #set objective
    model.setObjective(fs_obj + sec_obj)
    
    return model

#pseudo input for quick testrun
# vessel_types = [
#     VesselType(name="CTV", travel_speed=20, travel_cost_per_km=5, usage_cost_per_day=1000, n_teams=2, capacity_requirement=1.0, max_wind=10, max_wave=1.5, shift_length=12, day_rate=8000, mob_rate=2000, multiday=False, periodic_return=0),
#     VesselType(name="SOV", travel_speed=15, travel_cost_per_km=10, usage_cost_per_day=5000, n_teams=5, capacity_requirement=5.0, max_wind=15, max_wave=2.5, shift_length=24, day_rate=30000, mob_rate=10000, multiday=True, periodic_return=7)
# ]
# vessels = [
#     Vessel(name="SOV1", vessel_type=vessel_types[1]),
#     Vessel(name="SOV2", vessel_type=vessel_types[1])    
# ]
# windfarms = [
#     Windfarm(name="Wind Farm 1", latitude=54.55, longitude=-0.5, nTurbines=100, areaId=1)
# ]
# bases = [
#     Base(name="Base 1", latitude=53.7, longitude=7.4, cost=1000, max_capacity=20)
# ]
# maintenance_categories = [
#     MaintenanceCategory(name="Annual Service", failure_rate=5.0, duration=2, suitable_vessel_types=["CTV", "SOV"])
# ]
# ST_periods_in_LT_horizon = ["Jan"]
# model = init_model(
#     name="Wind Farm Maintenance Model",
#     days_per_ST_period=2,
#     ST_periods_in_LT_horizon=ST_periods_in_LT_horizon,
#     vessel_types=vessel_types,
#     vessels=vessels,
#     windfarms=windfarms,
#     bases=bases,
#     maintenance_categories=maintenance_categories,
#     pattern_scenarios_S={
#         ('CTV', 'Base 1', 'Wind Farm 1', 1, 1): [2, 5],
#         ('CTV', 'Base 1', 'Wind Farm 1', 1, 2): [1],
#         ('CTV', 'Base 1', 'Wind Farm 1', 2, 1): [2, 5],
#         ('CTV', 'Base 1', 'Wind Farm 1', 2, 2): [1]},
#     pattern_scenarios_M={
#         ('SOV', 'Wind Farm 1', 1, 1): [2, 5],
#         ('SOV', 'Wind Farm 1', 1, 2): [1],
#         ('SOV', 'Wind Farm 1', 2, 1): [2, 5],
#         ('SOV', 'Wind Farm 1', 2, 2): [1]},
#     pattern_library={
#         ("Annual Service", 1): 1,
#         ("Annual Service", 2): 2,
#         ("Annual Service", 5): 3
#         },
#     failure_scenarios={
#         ('Wind Farm 1', 'Annual Service', 1, 1): 0,
#         ('Wind Farm 1', 'Annual Service', 1, 2): 0,
#         ('Wind Farm 1', 'Annual Service', 2, 1): 0,
#         ('Wind Farm 1', 'Annual Service', 2, 2): 0,
#         },
#     downtime_cost_scenarios={
#         ("Wind Farm 1", 1, 1): 100,
#         ("Wind Farm 1", 1, 2): 100,
#         ("Wind Farm 1", 2, 1): 100,
#         ("Wind Farm 1", 2, 2): 100,
#         },
# )
