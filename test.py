import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.model import init_model
from model.classes import VesselType, Vessel, Windfarm, Base, MaintenanceCategory

#pseudo input for quick testrun
ST_periods_in_LT_horizon = ["Jan", "Feb"]
days_per_ST_period = 3
vessel_types = [
    # VesselType(name="CTV", travel_speed=20, travel_cost_per_km=5, usage_cost_per_day=100, n_teams=2, capacity_requirement=1.0, max_wind=10, max_wave=1.5, shift_length=12, day_rate=50, mob_rate=200, multiday=False, periodic_return=0),
    VesselType(name="SOV", travel_speed=15, travel_cost_per_km=5, usage_cost_per_day=0, n_teams=5, capacity_requirement=5.0, max_wind=15, max_wave=2.5, shift_length=24, day_rate=50, mob_rate=200, multiday=True, periodic_return=14)
]
vessels = [
    Vessel(name="SOV1", vessel_type=vessel_types[0]),
    # Vessel(name="SOV2", vessel_type=vessel_types[0])    
]
windfarms = [
    Windfarm(name="Wind Farm 1", latitude=54, longitude=7, nTurbines=100, areaId=1)
]
bases = [
    Base(name="Base 1", latitude=53.7, longitude=7.4, cost=100, max_capacity=20),
    # Base(name="Base 2", latitude=55, longitude=8.33, cost=100, max_capacity=15)
]
maintenance_categories = [
    MaintenanceCategory(name="Annual Service", failure_rate=5.0, duration=2, suitable_vessel_types=["CTV", "SOV"])
]
pattern_scenarios_S={
    ('CTV', 'Base 1', 'Wind Farm 1', 1, 1): [1],
    ('CTV', 'Base 1', 'Wind Farm 1', 2, 1): [1],
    ('CTV', 'Base 1', 'Wind Farm 1', 3, 1): [1],
    ('CTV', 'Base 1', 'Wind Farm 1', 4, 1): [1],
    ('CTV', 'Base 1', 'Wind Farm 1', 5, 1): [1],
    ('CTV', 'Base 1', 'Wind Farm 1', 6, 1): [1],
    ('CTV', 'Base 2', 'Wind Farm 1', 1, 1): [1],
    ('CTV', 'Base 2', 'Wind Farm 1', 2, 1): [1],
    ('CTV', 'Base 2', 'Wind Farm 1', 3, 1): [1],
    ('CTV', 'Base 2', 'Wind Farm 1', 4, 1): [1],
    ('CTV', 'Base 2', 'Wind Farm 1', 5, 1): [1],
    ('CTV', 'Base 2', 'Wind Farm 1', 6, 1): [1]
    }
pattern_scenarios_M={
    ('SOV', 'Wind Farm 1', 1, 1): [1],
    ('SOV', 'Wind Farm 1', 2, 1): [1],
    ('SOV', 'Wind Farm 1', 3, 1): [1],
    ('SOV', 'Wind Farm 1', 4, 1): [1],
    ('SOV', 'Wind Farm 1', 5, 1): [1],
    ('SOV', 'Wind Farm 1', 6, 1): [1],
    ('SOV', 'Wind Farm 1', 1, 1): [1],
    ('SOV', 'Wind Farm 1', 2, 1): [1],
    ('SOV', 'Wind Farm 1', 3, 1): [1],
    ('SOV', 'Wind Farm 1', 4, 1): [1],
    ('SOV', 'Wind Farm 1', 5, 1): [1],
    ('SOV', 'Wind Farm 1', 6, 1): [1]
    }
pattern_library={
    ("Annual Service", 1): 10,
    # ("Annual Service", 2): 2,
    # ("Annual Service", 5): 3
    }
failure_scenarios={
    ('Wind Farm 1', "Annual Service", 1, 1): 0,
    ('Wind Farm 1', "Annual Service", 2, 1): 10,
    ('Wind Farm 1', "Annual Service", 3, 1): 0,
    ('Wind Farm 1', "Annual Service", 4, 1): 0,
    ('Wind Farm 1', "Annual Service", 5, 1): 0,
    ('Wind Farm 1', "Annual Service", 6, 1): 0,
    # ('Wind Farm 1', "Annual Service", 7, 1): 1,
    # ('Wind Farm 1', "Annual Service", 8, 1): 0,
    # ('Wind Farm 1', "Annual Service", 9, 1): 0,
    # ('Wind Farm 1', "Annual Service", 10, 1): 0,
    }
downtime_cost_scenarios={
    ("Wind Farm 1", 1, 1): 500,
    ("Wind Farm 1", 2, 1): 500,
    ("Wind Farm 1", 3, 1): 500,
    ("Wind Farm 1", 4, 1): 500,
    ("Wind Farm 1", 5, 1): 500,
    ("Wind Farm 1", 6, 1): 500,
    # ("Wind Farm 1", 7, 1): 500,
    # ("Wind Farm 1", 8, 1): 500,
    # ("Wind Farm 1", 9, 1): 500,
    # ("Wind Farm 1", 10, 1): 500,
    }
# cost hvis ikke leie: 1300
# cost hvis st januar: 874 + 400 --> 1274
# cost hvis st februar: 200 + 5*50 + 424 + 500--> 1374
# cost hvis long term: 200 + 10*50 + 2*424 --> 1548
model = init_model(
    name="Wind Farm Maintenance Model",
    days_per_ST_period=days_per_ST_period,
    ST_periods_in_LT_horizon=ST_periods_in_LT_horizon,
    vessel_types=vessel_types,
    vessels=vessels,
    windfarms=windfarms,
    bases=bases,
    maintenance_categories=maintenance_categories,
    pattern_scenarios_S=pattern_scenarios_S,
    pattern_scenarios_M=pattern_scenarios_M,
    pattern_library=pattern_library,
    failure_scenarios=failure_scenarios,
    downtime_cost_scenarios=downtime_cost_scenarios,
)
model.setParam('OutputFlag', 1)
model.optimize()
#print all active eta variables
for v in model.getVars():
    if v.VarName.startswith("eta") and v.X > 0:
        print(f"{v.VarName}: {v.X}")
print("-----")
#print all active gamma variables
for v in model.getVars():
    if v.VarName.startswith("gamma") and v.X > 0:
        print(f"{v.VarName}: {v.X}")
print("-----")
#print all active alpha variables
for v in model.getVars():
    if v.VarName.startswith("alpha") and v.X > 0:
        print(f"{v.VarName}: {v.X}")
#print all active x variables
for v in model.getVars():
    if v.VarName.startswith("x") and v.X > 0:
        print(f"{v.VarName}: {v.X}")    
print("-----")
#print all active r variables
for v in model.getVars():
    if v.VarName.startswith("r_E") and v.X > 0:
        print(f"{v.VarName}: {v.X}")    
print("-----")
#print all active z variables
for v in model.getVars():
    if v.VarName.startswith("z") and v.X > 0:
        print(f"{v.VarName}: {v.X}")
print("-----")
print(f"objective value: {model.ObjVal}")
#print amount of variables
print(f"number of variables: {model.NumVars}")
#print amount of constraints
print(f"number of constraints: {model.NumConstrs}")