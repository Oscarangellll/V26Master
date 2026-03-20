import csv

import numpy as np
import pandas as pd

from config import CaseConfig, ScenarioConfig
from optimization_models import OptimizationModel

case = CaseConfig("cases/2W1B.yaml")

seed=30
rng = np.random.default_rng()

scenario_tree_sizes = [1, 2, 3, 4, 5]
n_instances = 20 

scenario = ScenarioConfig(case, np.arange(1, 51))

var_groups = ["gamma_ST", "gamma_LT", "eta"]

fieldnames = [
    "instance", "tree_size", "objective", "runtime",
    "var_group", "h", "b", "t", "value",
    "seed", "scenarios"
]

with open("test.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

rows = []

for j in range(1, n_instances + 1):
    for st_size in scenario_tree_sizes:
        s = rng.choice(range(1, 51), size=st_size, replace=False)
        s = [int(i) for i in s]
        print(j, s) 
        model = OptimizationModel(case, scenario, s)
        model.Params.OutputFlag = 0
        model.Params.MIPGap = 0.01
        model.optimize()
        with open("test.csv", "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            for var_group in var_groups:
                for key, var in getattr(model, var_group).items():
                    if var.X > 0.5:

                        if var_group == "gamma_ST":
                            h, b, t = key
                        elif var_group == "gamma_LT":
                            h, b = key
                            t = None
                        elif var_group == "eta":
                            b = key
                            h, t = None, None

                        writer.writerow({
                            "instance": j,
                            "tree_size": st_size,
                            "objective": model.ObjVal,
                            "runtime": model.Runtime,
                            "var_group": var_group,
                            "h": h,
                            "b": b,
                            "t": t,
                            "value": int(var.X),
                            "seed": seed,
                            "scenarios": list(s)
                        })


exit()
model = OptimizationModel(case, scenario, [1, 2, 3, 4, 5])

df = pd.read_csv("test.csv", dtype={"b": str})
for var_group in var_groups:
    for key, var in getattr(model, var_group).items():
        var.LB = 0
        var.UB = 0

for idx, row in df.iterrows():
    var_group = row["var_group"]
    value = row["value"]
    
    # reconstruct the key depending on group
    if var_group == "gamma_ST":
        key = (row["h"], row["b"], row["t"])
    elif var_group == "gamma_LT":
        key = (row["h"], row["b"])
    elif var_group == "eta":
        key = row["b"]

    # fix the variable in the model
    var = getattr(model, var_group)[key]
    var.LB = value
    var.UB = value

# now optimize if needed
model.optimize()

for var_group in var_groups:
    for key, var in getattr(model, var_group).items():
        if var.X > 0.5:
            print(key, var.X)
