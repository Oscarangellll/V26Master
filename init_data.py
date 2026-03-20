import argparse

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

generate_scenarios()

make_scenarios()
