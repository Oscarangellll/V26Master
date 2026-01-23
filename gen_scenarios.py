# libraries
import numpy as np
import pickle
import pandas as pd
# modules
from gen_weather import gen_weather
from gen_failures import gen_failures
from gen_powerprices import gen_powerprices

def generate_scenarios(num_scenarios, master_seed, scenarios_db, windfarms, wf_to_loc, maintenance_categories):
    #num_scenarios is a number
    #seed is a number
    #scenarios_db is a read csv using pandas, simply containing a bunch of seeds to choose from
    rng = np.random.default_rng(master_seed)
    
    # load and params from pkl file:
    with open("models.pkl", "rb") as f:
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
            
            edf = gen_powerprices(seed_i, electricity_params, wdf)
            edf["scenario_id"] = scenario_id  # add scenario_id column
            edf["seed"] = seed_i  # add seed column
            edf["location"] = loc  # add location column
            electricity_rows.append(edf)
            
        for wf in windfarms:
            fdf = gen_failures(seed_i, wf, maintenance_categories)
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



from dataclasses import dataclass
@dataclass
class MaintenanceCategory:
    name: str
    failure_rate: float  # per year
    
maintenance_categories = [
    MaintenanceCategory("Gearbox", 0.12),
    MaintenanceCategory("Generator", 0.08),
    MaintenanceCategory("Pitch", 0.10),
]

weather_df, electricity_df, failure_df, meta = generate_scenarios(
    num_scenarios=5,
    master_seed=42,
    scenarios_db=pd.read_csv("scenarios_db.csv"),
    windfarms=["WF1", "WF2"],
    wf_to_loc={"WF1": 1, "WF2": 2},
    maintenance_categories=maintenance_categories
)
    
print(weather_df.head())
print(electricity_df.head())
print(failure_df.head())
#print size of each dataframe in MB
print(f"Weather DF size: {weather_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
print(f"Electricity DF size: {electricity_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
print(f"Failure DF size: {failure_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
# print(meta)

#plot a sample of the weather data for location 1, scenario 0
import matplotlib.pyplot as plt
sample_wdf = weather_df[(weather_df["location"] == 1) & (weather_df["scenario_id"] == 0)]
plt.figure(figsize=(12,6))
plt.plot(sample_wdf["speed"].values, label="Wind Speed (m/s)")
plt.plot(sample_wdf["height"].values, label="Wind Height (m)")
plt.title("Sample Weather Data for Location 1, Scenario 0")
plt.xlabel("Time (hours)")
plt.ylabel("Value")
plt.legend()
plt.show()

#plot a sample of the electricity price data for location 1, scenario 0
sample_edf = electricity_df[(electricity_df["location"] == 1) & (electricity_df["scenario_id"] == 0)]
#make daily by averaging over each day
sample_edf_daily = sample_edf.groupby("day_id")["power_price"].mean().reset_index()
plt.figure(figsize=(12,6))
plt.plot(sample_edf_daily["power_price"].values, label="Power Price (EUR/MWhe)")
plt.title("Sample Electricity Price Data for Location 1, Scenario 0")
plt.xlabel("Time (days)")
plt.ylabel("Price (EUR/MWhe)")
plt.legend()
plt.show()

#plot a scatter plot of daily wind speed vs daily power price for location 1, scenario 0

#first make daily wind speed by averaging over each day
sample_wdf_daily = sample_wdf.groupby("day_id")["speed"].mean().reset_index()
plt.figure(figsize=(8,6))
plt.scatter(sample_wdf_daily["speed"].values, sample_edf_daily["power_price"].values, alpha=0.5)
plt.title("Wind Speed vs Power Price for Location 1, Scenario 0")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Power Price (EUR/MWhe)")
xlim = (1.8, 16.6)
ylim = (-50, 353)
plt.grid(True)
plt.xlim(xlim)
plt.ylim(ylim)
#plot a trendline
z = np.polyfit(sample_wdf_daily["speed"].values, sample_edf_daily["power_price"].values, 1)
p = np.poly1d(z)
#print the equation of the line
print(f"Trendline: y = {z[0]:.2f}x + {z[1]:.2f}")
plt.plot(sample_wdf_daily["speed"].values, p(sample_wdf_daily["speed"].values), "r--")
plt.show()

# # Plot power prices for one week (7 days) for location 1, scenario 0
# start_day = 100  # day_id 1..360
# end_day = start_day + 6

# sample_edf = electricity_df[(electricity_df["location"] == 1) & (electricity_df["scenario_id"] == 0)]
# week_edf = sample_edf[(sample_edf["day_id"] >= start_day) & (sample_edf["day_id"] <= end_day)].copy()

# week_edf = week_edf.sort_values(["day_id", "hour"])

# # Continuous hour index for the week: 0..167
# week_edf["t_week"] = (week_edf["day_id"] - start_day) * 24 + week_edf["hour"]  # if hour is 0..23

# plt.figure(figsize=(12,6))
# plt.plot(week_edf["t_week"], week_edf["power_price"], label="Power price")
# plt.title(f"Power Price – Location 1, Scenario 0, Days {start_day}–{end_day}")
# plt.xlabel("Hour in week (0–167)")
# plt.ylabel("Price")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
