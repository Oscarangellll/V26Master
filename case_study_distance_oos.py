import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from case_study_distance_failures import DISTANCE_TURBINES
from case_study_distance_iss import DISTANCE_CASES
from case_study_oos import run as run_oos
from data.fixed_data import data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate distance ISS case-study solutions on OOS scenarios using the "
            "common uniform-100-turbine failure scenario set."
        )
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(DISTANCE_CASES),
        default=["BCD_close", "BCG_cluster_far", "BEG_spread"],
    )
    parser.add_argument(
        "--input-root",
        default="results/case_studies/distance",
        help="Root containing <case>/iss.csv.",
    )
    parser.add_argument(
        "--output-root",
        default="results/case_studies/distance",
        help="Root where <case>/coalition_oos.csv and <case>/windfarm_oos.csv are written.",
    )
    parser.add_argument(
        "--failure-dir",
        default="data/scenario_data/scenario_data_distance/uniform_100/failures",
        help="Common failure scenario directory used by all distance cases.",
    )
    parser.add_argument("--oos-start", type=int, default=501)
    parser.add_argument("--oos-end", type=int, default=1500)
    parser.add_argument("--bases", nargs="+", default=None)
    parser.add_argument("--max-multiday-vessels", type=int, default=3)
    parser.add_argument("--fix-alpha", action="store_true")
    return parser


def _set_turbine_counts() -> None:
    for wind_farm in data.wind_farms:
        if wind_farm.name in DISTANCE_TURBINES:
            wind_farm.n_turbines = DISTANCE_TURBINES[wind_farm.name]


def _oos_args(args: argparse.Namespace, case_name: str) -> SimpleNamespace:
    case_dir = Path(args.output_root) / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    return SimpleNamespace(
        input=str(Path(args.input_root) / case_name / "iss.csv"),
        case_id=case_name,
        bases=args.bases,
        max_multiday_vessels=args.max_multiday_vessels,
        oos_start=args.oos_start,
        oos_end=args.oos_end,
        num_nodes=1,
        node_id=0,
        coalitions=None,
        coalition_output=str(case_dir / "coalition_oos.csv"),
        windfarm_output=str(case_dir / "windfarm_oos.csv"),
        fix_alpha=args.fix_alpha,
    )


def main() -> None:
    args = build_parser().parse_args()
    failure_dir = Path(args.failure_dir)
    if not failure_dir.exists():
        raise FileNotFoundError(f"Missing distance failure directory: {failure_dir}")

    previous_failure_dir = os.environ.get("FAILURE_SCENARIO_DIR")
    try:
        _set_turbine_counts()
        os.environ["FAILURE_SCENARIO_DIR"] = str(failure_dir)

        for case_name in args.cases:
            input_path = Path(args.input_root) / case_name / "iss.csv"
            if not input_path.exists():
                raise FileNotFoundError(f"Missing ISS file for {case_name}: {input_path}")

            print(f"\n=== OOS {case_name} ({''.join(DISTANCE_CASES[case_name])}) ===")
            print(f"ISS input: {input_path}")
            print(f"Failures:  {failure_dir}")
            print(f"Turbines:  {DISTANCE_TURBINES}")
            run_oos(_oos_args(args, case_name))
    finally:
        if previous_failure_dir is None:
            os.environ.pop("FAILURE_SCENARIO_DIR", None)
        else:
            os.environ["FAILURE_SCENARIO_DIR"] = previous_failure_dir


if __name__ == "__main__":
    main()
