

import pandas as pd


class ScenarioConfig:
    def __init__(self, case, scenario_ids):
        
        # Second stage sets
        self.S = scenario_ids
        
        # Second stage parameters
        df_F = pd.read_parquet(
            "data/scenario_data/failures.parquet", 
            filters=[("s", "in", scenario_ids)]
        )
        self.F = {(r.w, r.m, r.d, r.s): r.failures for r in df_F.itertuples()}
       
        C_D_df = pd.read_parquet(
            "data/scenario_data/downtime_cost.parquet", 
            filters=[("s", "in", scenario_ids)]
        )
        self.C_D = {(r.w, r.d, r.s): r.downtime_cost for r in C_D_df.itertuples()}


        self.P = {
            ("Annual Service", 1): 1,
            ("Manual Reset", 1): 1, 
            ("Minor Repair", 1): 0,
            ("Medium Repair", 1): 1,
            ("Severe Repair", 1): 0,

            ("Annual Service", 2): 1,
            ("Manual Reset", 2): 1,
            ("Minor Repair", 2): 1,
            ("Medium Repair", 2): 1,
            ("Severe Repair", 2): 1
        }

        self.K_S = {
            (h, b, w, d, s): [1] 
            for h in case.H 
            for b in case.B 
            for w in case.W 
            for d in case.D 
            for s in self.S
        }

        self.K_M = {
            (h, w, d, s): [1, 2] 
            for h in case.H 
            for w in case.W 
            for d in case.D 
            for s in self.S
        }

