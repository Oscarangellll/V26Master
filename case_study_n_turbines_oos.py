import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from case_study_n_turbines_iss import CASE_TURBINES
from case_study_oos import run as run_oos
from data.fixed_data import data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate n-turbines ISS case-study solutions on OOS scenarios. "
            "The wrapper reuses case_study_oos.py, but sets the correct failure "
            "scenario directory and turbine counts for each variant."
        )
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(CASE_TURBINES),
        default=["BCD_low", "BCD_high", "BCD_mixed"],
    )
    parser.add_argument(
        "--input-root",
        default="results/case_studies/n_turbines",
        help="Root containing <case>/iss.csv.",
    )
    parser.add_argument(
        "--output-root",
        default="results/case_studies/n_turbines",
        help="Root where <case>/coalition_oos.csv and <case>/windfarm_oos.csv are written.",
    )
    parser.add_argument(
        "--failure-root",
        default="data/scenario_data/scenario_data_n_turbines",
        help="Root containing <case>/failures folders.",
    )
    parser.add_argument("--oos-start", type=int, default=501)
    parser.add_argument("--oos-end", type=int, default=1500)
    parser.add_argument("--bases", nargs="+", default=None)
    parser.add_argument("--max-multiday-vessels", type=int, default=3)
    parser.add_argument(
        "--coalitions",
        nargs="+",
        default=None,
        help="Only evaluate selected coalitions.",
    )
    parser.add_argument(
        "--use-iss-scenarios",
        action="store_true",
        help="Evaluate each row on its ISS scenarios instead of the OOS range.",
    )
    parser.add_argument(
        "--fix-alpha",
        action="store_true",
        help="Also fix alpha vessel-index decisions during OOS evaluation.",
    )
    return parser


def _set_turbine_counts(counts: dict[str, int]) -> None:
    for wind_farm in data.wind_farms:
        if wind_farm.name in counts:
            wind_farm.n_turbines = counts[wind_farm.name]


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
        coalitions=args.coalitions,
        use_iss_scenarios=args.use_iss_scenarios,
        coalition_output=str(case_dir / "coalition_oos.csv"),
        windfarm_output=str(case_dir / "windfarm_oos.csv"),
        fix_alpha=args.fix_alpha,
    )


def main() -> None:
    args = build_parser().parse_args()
    previous_failure_dir = os.environ.get("FAILURE_SCENARIO_DIR")

    try:
        for case_name in args.cases:
            failure_dir = Path(args.failure_root) / case_name / "failures"
            if not failure_dir.exists():
                raise FileNotFoundError(
                    f"Missing failure directory for {case_name}: {failure_dir}"
                )

            input_path = Path(args.input_root) / case_name / "iss.csv"
            if not input_path.exists():
                raise FileNotFoundError(f"Missing ISS file for {case_name}: {input_path}")

            _set_turbine_counts(CASE_TURBINES[case_name])
            os.environ["FAILURE_SCENARIO_DIR"] = str(failure_dir)

            print(f"\n=== OOS {case_name} ===")
            print(f"ISS input: {input_path}")
            print(f"Failures:  {failure_dir}")
            print(f"Turbines:  {CASE_TURBINES[case_name]}")

            run_oos(_oos_args(args, case_name))
    finally:
        if previous_failure_dir is None:
            os.environ.pop("FAILURE_SCENARIO_DIR", None)
        else:
            os.environ["FAILURE_SCENARIO_DIR"] = previous_failure_dir


if __name__ == "__main__":
    main()
