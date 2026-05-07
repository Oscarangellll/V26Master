import argparse

from plot_scripts.plot_stratified_comparison import plot_stratified_comparison
from plot_scripts.plot_runtime_comparison import plot_runtime_comparison
from plot_scripts.plot_oss_con_mip import plot_oss_con_mip
from plot_scripts.plot_stability import plot_stability
from plot_scripts.plot_map import plot_map
from plot_scripts.plot_real_weather_seasonality import plot_real_weather_seasonality
from plot_scripts.plot_real_weather_correlation import plot_real_weather_correlation 

PLOT_REGISTRY = {
    "stratified": plot_stratified_comparison,
    "runtime": plot_runtime_comparison,
    "oss_con_mip": plot_oss_con_mip,
    "stability": plot_stability,
    "map": plot_map,
    "real_weather_seasonality": plot_real_weather_seasonality,
    "real_weather_correlation": plot_real_weather_correlation
}

def build_parser():
    parser = argparse.ArgumentParser(
        description="Run thesis plots with shared style"
    )

    parser.add_argument(
        "plots",
        nargs="*",
        choices=list(PLOT_REGISTRY.keys()),
        help="Which plots to run (default: all)",
    )

    return parser

args = build_parser().parse_args()

# If no plots specified, run all
selected = args.plots if args.plots else list(PLOT_REGISTRY.keys())

for name in selected:
    PLOT_REGISTRY[name]()
