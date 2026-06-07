import argparse
from importlib import import_module

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["figure.constrained_layout.use"] = False


PLOT_REGISTRY = {
    "stratified": ("plot_scripts.plot_stratified_comparison", "plot_stratified_comparison"),
    "stratified_cv": ("plot_scripts.plot_stratified_comparison", "plot_stratified_cv"),
    "runtime": ("plot_scripts.plot_runtime_comparison", "plot_runtime_comparison"),
    "oss_con_mip": ("plot_scripts.plot_oss_con_mip", "plot_oss_con_mip"),
    "oss_con_mip_gap": ("plot_scripts.plot_oss_con_mip_gap", "plot_oss_con_mip_gap"),
    "stability": ("plot_scripts.plot_stability", "plot_stability"),
    "stability_cv": ("plot_scripts.plot_stability_cv", "plot_stability_cv"),
    "master_gap_vs_oos_gap": (
        "plot_scripts.plot_master_gap_vs_oos_gap_ad_hoc",
        "plot_master_gap_vs_oos_gap_ad_hoc",
    ),
    "optimality_gap": (
        "plot_scripts.plot_optimality_gap_ad_hoc",
        "plot_optimality_gap_ad_hoc",
    ),
    "map": ("plot_scripts.plot_map", "plot_map"),
    "real_weather_seasonality": (
        "plot_scripts.plot_real_weather_seasonality",
        "plot_real_weather_seasonality",
    ),
    "real_weather_correlation": (
        "plot_scripts.plot_real_weather_correlation",
        "plot_real_weather_correlation",
    ),
    "real_electricity_prices": (
        "plot_scripts.plot_real_electricity_prices",
        "plot_real_electricity_prices",
    ),
    "electricity_validation": (
        "plot_scripts.plot_electricity_validation",
        "plot_electricity_validation",
    ),
    "weather_validation": (
        "plot_scripts.plot_weather_validation",
        "plot_weather_validation",
    ),
    "price_weather_relationship": (
        "plot_scripts.plot_price_weather_relationship",
        "plot_price_weather_relationship",
    ),
    "case_studies": (
        "plot_scripts.plot_case_studies",
        "plot_case_studies",
    ),
    "distance_cases": (
        "plot_scripts.plot_distance_cases",
        "plot_distance_cases",
    ),
    "n_turbines_cases": (
        "plot_scripts.plot_n_turbines_allocations",
        "plot_n_turbines_cases",
    ),
}

PLOT_GROUPS = {
    "method": [
        "stratified",
        "stratified_cv",
        "oss_con_mip",
        "oss_con_mip_gap",
        "runtime",
        "stability",
        "stability_cv",
    ],
    "data": [
        "map",
        "real_weather_seasonality",
        "real_weather_correlation",
        "real_electricity_prices",
    ],
    "appendix": [
        "weather_validation",
        "electricity_validation",
        "price_weather_relationship",
    ],
    "case": [
        "case_studies",
        "distance_cases",
        "n_turbines_cases",
    ],
    "diagnostic": [
        "master_gap_vs_oos_gap",
        "optimality_gap",
    ],
}
PLOT_GROUPS["thesis"] = (
    PLOT_GROUPS["method"]
    + PLOT_GROUPS["data"]
    + PLOT_GROUPS["case"]
    + PLOT_GROUPS["appendix"]
)
PLOT_GROUPS["all"] = PLOT_GROUPS["thesis"] + PLOT_GROUPS["diagnostic"]


def build_parser():
    valid_targets = sorted([*PLOT_REGISTRY, *PLOT_GROUPS])
    parser = argparse.ArgumentParser(
        description=(
            "Generate thesis plots into plot_scripts/plots. Pass plot names or "
            "groups such as method, data, case, appendix, thesis, or all."
        )
    )
    parser.add_argument(
        "targets",
        nargs="*",
        default=["thesis"],
        help="Plot names or groups to run. Use --list to show valid targets.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available plot names and groups, then exit.",
    )
    parser.set_defaults(valid_targets=valid_targets)
    return parser


def load_plot(name):
    module_name, function_name = PLOT_REGISTRY[name]
    module = import_module(module_name)
    return getattr(module, function_name)


def expand_targets(targets):
    selected = []
    seen = set()

    for target in targets:
        if target in PLOT_GROUPS:
            names = PLOT_GROUPS[target]
        elif target in PLOT_REGISTRY:
            names = [target]
        else:
            valid = ", ".join(sorted([*PLOT_REGISTRY, *PLOT_GROUPS]))
            raise SystemExit(f"Unknown plot target '{target}'. Valid targets: {valid}")

        for name in names:
            if name not in seen:
                selected.append(name)
                seen.add(name)

    return selected


def print_available_targets():
    print("Plot groups:")
    for group, names in PLOT_GROUPS.items():
        print(f"  {group}: {', '.join(names)}")

    print("\nIndividual plots:")
    for name in sorted(PLOT_REGISTRY):
        print(f"  {name}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        print_available_targets()
        return

    for name in expand_targets(args.targets):
        print(f"[plot] {name}")
        load_plot(name)()


if __name__ == "__main__":
    main()
