import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

from case_study_distance_failures import DISTANCE_TURBINES


DISTANCE_CASES = {
    "BCD_close": ["B", "C", "D"],
    "BCG_cluster_far": ["B", "C", "G"],
    "BEG_spread": ["B", "E", "G"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run ISS case studies for selected distance structures with a common "
            "uniform-100-turbine failure scenario set."
        )
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(DISTANCE_CASES),
        default=["BCD_close", "BCG_cluster_far", "BEG_spread"],
    )
    parser.add_argument(
        "--failure-dir",
        default="data/scenario_data/scenario_data_distance/uniform_100/failures",
        help="Common failure scenario directory used by all distance cases.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/case_studies/distance",
        help="Root folder for generated ISS files.",
    )
    parser.add_argument(
        "--bases",
        nargs="+",
        default=["3", "5", "6", "7"],
        help="Candidate bases passed to main.py.",
    )
    parser.add_argument("--max-multiday-vessels", type=int, default=3)
    parser.add_argument("--n-scenarios", type=int, default=15)
    parser.add_argument("--scenario-start", type=int, default=1)
    parser.add_argument("--scenario-end", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--method", choices=["con_mp", "mip"], default="con_mp")
    parser.add_argument("--mip-gap", type=float, default=0.02)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument(
        "--sampling-mode",
        choices=["per-coalition", "shared"],
        default="shared",
    )
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


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
                "actors",
                "failure_dir",
                "output",
                "turbine_count",
                "n_scenarios",
                "scenario_start",
                "scenario_end",
                "seed",
                "sampling_mode",
            ],
        )
        writer.writeheader()
        for case_name in args.cases:
            writer.writerow(
                {
                    "case": case_name,
                    "actors": ";".join(DISTANCE_CASES[case_name]),
                    "failure_dir": args.failure_dir,
                    "output": output_path_for_case(output_dir, case_name),
                    "turbine_count": 100,
                    "n_scenarios": args.n_scenarios,
                    "scenario_start": args.scenario_start,
                    "scenario_end": args.scenario_end,
                    "seed": args.seed,
                    "sampling_mode": args.sampling_mode,
                }
            )


def build_main_command(args: argparse.Namespace, case_name: str, output_path: Path) -> list[str]:
    command = [
        sys.executable,
        "main.py",
        "--actors",
        *DISTANCE_CASES[case_name],
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
    failure_dir = Path(args.failure_dir)
    if not failure_dir.exists():
        raise FileNotFoundError(f"Missing distance failure directory: {failure_dir}")

    output_path = output_path_for_case(Path(args.output_dir), case_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["FAILURE_SCENARIO_DIR"] = str(failure_dir)

    print(f"\n=== {case_name} ({''.join(DISTANCE_CASES[case_name])}) ===")
    print(f"Failures: {failure_dir}")
    print(f"Output:   {output_path}")
    print(f"Turbines: {DISTANCE_TURBINES}")

    subprocess.run(build_main_command(args, case_name, output_path), check=True, env=env)


def main() -> None:
    args = build_parser().parse_args()
    write_metadata(args)

    for case_name in args.cases:
        run_case(args, case_name)


if __name__ == "__main__":
    main()
