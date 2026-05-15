import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


CASE_TURBINES = {
    "BCD_low": {"B": 50, "C": 50, "D": 50},
    "BCD_high": {"B": 150, "C": 150, "D": 150},
    "BCD_mixed": {"B": 50, "C": 100, "D": 150},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run ISS case studies for BCD turbine-count variants while reading "
            "weather, prices, and other scenario data from the ordinary scenario root."
        )
    )

    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(CASE_TURBINES),
        default=["BCD_low", "BCD_high", "BCD_mixed"],
        help="Turbine-count variants to run.",
    )
    parser.add_argument(
        "--failure-root",
        default="data/scenario_data/scenario_data_n_turbines",
        help="Root containing <case>/failures folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/case_studies/n_turbines",
        help="Root folder for the generated ISS files.",
    )
    parser.add_argument(
        "--actors",
        nargs="+",
        default=["B", "C", "D"],
        help="Wind farms to include. Default is B C D.",
    )
    parser.add_argument(
        "--bases",
        nargs="+",
        default=["3", "5", "6", "7"],
        help="Candidate bases passed to main.py.",
    )
    parser.add_argument(
        "--max-multiday-vessels",
        type=int,
        default=3,
        help="Passed to main.py.",
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=15,
        help="Number of scenarios in each ISS solve.",
    )
    parser.add_argument("--scenario-start", type=int, default=1)
    parser.add_argument("--scenario-end", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument(
        "--method",
        choices=["con_mp", "mip"],
        default="con_mp",
    )
    parser.add_argument("--mip-gap", type=float, default=0.02)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument(
        "--sampling-mode",
        choices=["per-coalition", "shared"],
        default="shared",
        help="Shared keeps the same scenario sample across coalitions within each case.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing case output files instead of overwriting.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip coalitions already present in each case output file.",
    )

    return parser


def failure_dir_for_case(failure_root: Path, case_name: str) -> Path:
    return failure_root / case_name / "failures"


def output_path_for_case(output_dir: Path, case_name: str) -> Path:
    return output_dir / case_name / "iss.csv"


def write_metadata(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "case_metadata.csv"

    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "B",
                "C",
                "D",
                "failure_dir",
                "output",
                "n_scenarios",
                "scenario_start",
                "scenario_end",
                "seed",
                "sampling_mode",
            ],
        )
        writer.writeheader()
        for case_name in args.cases:
            row = {
                "case": case_name,
                **CASE_TURBINES[case_name],
                "failure_dir": failure_dir_for_case(Path(args.failure_root), case_name),
                "output": output_path_for_case(output_dir, case_name),
                "n_scenarios": args.n_scenarios,
                "scenario_start": args.scenario_start,
                "scenario_end": args.scenario_end,
                "seed": args.seed,
                "sampling_mode": args.sampling_mode,
            }
            writer.writerow(row)


def build_main_command(args: argparse.Namespace, output_path: Path) -> list[str]:
    command = [
        sys.executable,
        "main.py",
        "--actors",
        *args.actors,
        "--bases",
        *args.bases,
        "--max-multiday-vessels",
        str(args.max_multiday_vessels),
        "--num-nodes",
        "1",
        "--node-id",
        "0",
        "--n-scenarios",
        str(args.n_scenarios),
        "--scenario-start",
        str(args.scenario_start),
        "--scenario-end",
        str(args.scenario_end),
        "--seed",
        str(args.seed),
        "--method",
        args.method,
        "--mip-gap",
        str(args.mip_gap),
        "--sampling-mode",
        args.sampling_mode,
        "--output",
        str(output_path),
    ]

    if args.time_limit is not None:
        command.extend(["--time-limit", str(args.time_limit)])
    if args.append:
        command.append("--append")
    if args.skip_existing:
        command.append("--skip-existing")

    return command


def run_case(args: argparse.Namespace, case_name: str) -> None:
    failure_dir = failure_dir_for_case(Path(args.failure_root), case_name)
    if not failure_dir.exists():
        raise FileNotFoundError(
            f"Missing failure directory for {case_name}: {failure_dir}"
        )

    output_path = output_path_for_case(Path(args.output_dir), case_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["FAILURE_SCENARIO_DIR"] = str(failure_dir)

    turbines = ", ".join(
        f"{wind_farm}={count}"
        for wind_farm, count in CASE_TURBINES[case_name].items()
    )
    print(f"\n=== {case_name} ({turbines}) ===")
    print(f"Failures: {failure_dir}")
    print(f"Output:   {output_path}")

    subprocess.run(
        build_main_command(args, output_path),
        check=True,
        env=env,
    )


def main() -> None:
    args = build_parser().parse_args()
    write_metadata(args)

    for case_name in args.cases:
        run_case(args, case_name)


if __name__ == "__main__":
    main()
