import argparse

import numpy as np

from data.fixed_data import data
from data.scripts import fetch_weather_data
from data.scripts import process_price_data 
from data.scripts import generate_scenarios
from data.scripts import make_scenarios


parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "--fetch_weather",
    action="store_true",
    help="Fetch new weather data"
)
group.add_argument(
    "--no_fetch_weather",
    action="store_true",
    help="Do not fetch new weather data"
)

args = parser.parse_args()

if args.fetch_weather:
    fetch_weather_data()
else:
    print("Skipping weather fetch...")

process_price_data()

rng = np.random.default_rng(seed=data.generate_scenarios_seed)
scenarios = [s for s in range(1, data.n_scenarios_to_generate + 1)]

generate_scenarios(rng, scenarios)

make_scenarios(scenarios)
