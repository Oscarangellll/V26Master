# libraries
import numpy as np
import pickle
import pandas as pd
# modules
from gen_weather import gen_weather
from gen_failures import gen_failures
from gen_powerprices import gen_powerprices

def generate_scenarios(num_scenarios, master_seed, scenarios_db, windfarms, wf_to_loc):
    #num_scenarios is a number
    #seed is a number
    #scenarios_db is a read csv using pandas, simply containing a bunch of seeds to choose from
    rng = np.random.default_rng(master_seed)
    
    # load and params from pkl file:
    with open("model_params.pkl", "rb") as f:
        model_params = pickle.load(f)

    # pick random seeds from scenarios_db
    scenario_seeds = rng.choice(scenarios_db["seed"].to_numpy(), size=num_scenarios, replace=False)
    
    # Only generate for locations actually used in this case
    locs = sorted(set(wf_to_loc[wf] for wf in windfarms))
    
    weather_rows, electricity_rows, failure_rows = [], [], []
    
    for scenario_id, seed_i in enumerate(scenario_seeds): #kan være vi må sørge for at seed_i er en int       
        for loc in locs:
            weather_params = model_params[loc]["weather"]
            electricity_params = model_params[loc]["electricity"]
            
            wdf = gen_weather(seed_i, weather_params)
            wdf["scenario_id"] = scenario_id  # add scenario_id column
            wdf["seed"] = seed_i  # add seed column
            wdf["location"] = loc  # add location column
            weather_rows.append(wdf)
            
            edf = gen_powerprices(seed_i, wdf, electricity_params)
            edf["scenario_id"] = scenario_id  # add scenario_id column
            edf["seed"] = seed_i  # add seed column
            edf["location"] = loc  # add location column
            electricity_rows.append(edf)
            
        for wf in windfarms:
            fdf = gen_failures(seed_i, wf)
            fdf["scenario_id"] = scenario_id  # add scenario_id column
            fdf["seed"] = seed_i  # add seed column
            fdf["windfarm"] = wf  # add windfarm column
            failure_rows.append(fdf)
            
    # concatenate all dataframes
    weather_df = pd.concat(weather_rows, ignore_index=True)
    electricity_df = pd.concat(electricity_rows, ignore_index=True)
    failure_df = pd.concat(failure_rows, ignore_index=True)
    
    meta = {
        "scenario_seeds": {i: scenario_seeds[i] for i in range(num_scenarios)},
        "wf_to_loc": dict(wf_to_loc),
        "time_def": {"M":12, "D":30, "H":24, "dayID_range": (1,360)} 
    }
        
    return weather_df, electricity_df, failure_df, meta