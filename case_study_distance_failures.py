import argparse
import os
from pathlib import Path

import numpy as np

from data.fixed_data import data


DISTANCE_TURBINES = {
    "B": 100,
    "C": 100,
    "D": 100,
    "E": 100,
    "G": 100,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate only failure scenarios for the distance case study. "
            "Weather, prices, downtime costs, and pattern data are not regenerated."
        )
    )
    parser.add_argument(
        "--output-root",
        default="data/scenario_data/scenario_data_distance/uniform_100",
        help="Folder where the failures/ directory is written.",
    )
    parser.add_argument("--scenario-start", type=int, default=1)
    parser.add_argument("--scenario-end", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=data.generate_scenarios_seed)
    return parser


def _set_turbine_counts() -> None:
    for wind_farm in data.wind_farms:
        if wind_farm.name in DISTANCE_TURBINES:
            wind_farm.n_turbines = DISTANCE_TURBINES[wind_farm.name]


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    _set_turbine_counts()
    os.environ["SCENARIO_DATA_DIR"] = str(output_root)

    # Import after SCENARIO_DATA_DIR is set, since generate_scenarios.py reads it at import time.
    from data.scripts.generate_scenarios import _generate_failures

    scenarios = list(range(args.scenario_start, args.scenario_end + 1))
    rng = np.random.default_rng(args.seed)

    print(f"Generating distance failure scenarios {args.scenario_start}-{args.scenario_end}")
    print(f"Output:   {output_root / 'failures'}")
    print(f"Turbines: {DISTANCE_TURBINES}")
    _generate_failures(rng, scenarios)


if __name__ == "__main__":
    main()
