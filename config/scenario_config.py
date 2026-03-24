from collections import defaultdict
import os
from pathlib import Path

import pandas as pd

class ScenarioConfig:
    def __init__(self, case, scenario_ids):
        scenario_ids = sorted(int(s) for s in scenario_ids)
        if len(scenario_ids) == 0:
            raise ValueError("ScenarioConfig requires at least one scenario id")
        
        scenario_data_dir = Path(os.environ.get("SCENARIO_DATA_DIR", "data/scenario_data"))


        # Second stage sets
        df_K_S = pd.read_parquet(
            scenario_data_dir / "singleday_pattern_set.parquet",
            filters=[
                ("h", "in", case.H_S),
                ("b", "in", case.B),
                ("w", "in", case.W),
                ("s", "in", scenario_ids)
            ]
        )
        K_S = defaultdict(dict)
        for r in df_K_S.itertuples():
            K_S[r.s][(r.h, r.b, r.w, r.d)] = r.patterns
        
        df_K_M = pd.read_parquet(
            scenario_data_dir / "multiday_pattern_set.parquet",
            filters=[
                ("h", "in", case.H_M),
                ("w", "in", case.W),
                ("s", "in", scenario_ids)
            ]
        )
        K_M = defaultdict(dict)
        for r in df_K_M.itertuples():
            K_M[r.s][(r.h, r.w, r.d)] = r.patterns
            
        # Second stage parameters
        df_P = pd.read_parquet(scenario_data_dir / "patterns.parquet")
        self.P = {(r.m, r.k): r.task_count for r in df_P.itertuples()}
        
        df_F = pd.read_parquet(
            scenario_data_dir / "failures.parquet", 
            filters=[
                ("w", "in", case.W),
                ("s", "in", scenario_ids)
            ]
        )
        F = defaultdict(dict)
        for r in df_F.itertuples():
            F[r.s][(r.w, r.m, r.d)] = r.failures 

        df_C_D = pd.read_parquet(
            scenario_data_dir / "downtime_cost.parquet", 
            filters=[
                ("w", "in", case.W),
                ("s", "in", scenario_ids)
            ]
        )
        C_D = defaultdict(dict)
        for r in df_C_D.itertuples():
            C_D[r.s][(r.w, r.d)] = r.downtime_cost
        
        self.K_S = K_S
        self.K_M = K_M
        self.F = F
        self.C_D = C_D