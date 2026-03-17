

import pandas as pd
from config.patterns import gen_patterns


class ScenarioConfig:
    def __init__(self, case, scenario_ids):
        
        # Second stage sets
        self.S = scenario_ids
        
        df_ww = pd.read_parquet(
            "data/scenario_data/weather_windows.parquet", 
            filters=[("s", "in", scenario_ids)]
        )
        ww = {(r.h, r.wl_id, r.d, r.s): r.ww for r in df_ww.itertuples()}
        self.K_S, self.K_M, self.P = gen_patterns(ww, case, scenario_ids)

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