import argparse

from iss import run_iss
from oos import run_oos


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ISS first, then evaluate unique ISS solutions OOS."
    )

    parser.add_argument("-c", "--case", required=True, help="Case name, e.g. 1W1B")
    parser.add_argument(
        "-m",
        "--method",
        default="mip",
        choices=["mip", "con", "con_mp"],
        help="Solution method (currently only mip is supported)",
    )
    parser.add_argument(
        "-n",
        "--n_trees",
        type=int,
        required=True,
        help="Number of ISS replications",
    )
    parser.add_argument(
        "-s",
        "--scenario_tree_sizes",
        type=int,
        nargs="+",
        required=True,
        help="Scenario tree sizes, e.g. 1 3 5",
    )

    parser.add_argument("--seed", type=int, default=99, help="Random seed")
    parser.add_argument("--iss_pool_start", type=int, default=1)
    parser.add_argument("--iss_pool_end", type=int, default=50)
    parser.add_argument("--oos_pool_start", type=int, default=51)
    parser.add_argument("--oos_pool_end", type=int, default=300)
    parser.add_argument(
        "--nested_trees",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use nested ISS trees inside each replication",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append ISS rows to existing ISS.csv",
    )
    parser.add_argument(
        "--append_oos",
        action="store_true",
        help="Append only new OOS rows and reuse previously evaluated solutions",
    )
    parser.add_argument(
        "--scenario_reduction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use k-medoids scenario reduction within each instance pool",
    )
    parser.add_argument(
        "--instance_pool_size",
        type=int,
        default=100,
        help="Number of ISS scenarios per instance",
    )

    parser.add_argument(
        "--iss_output",
        default=None,
        help="Optional ISS output path (default: results/stability/<case>/<method>/ISS.csv)",
    )
    parser.add_argument(
        "--oos_output",
        default=None,
        help="Optional OOS output path (default: results/stability/<case>/<method>/OSS.csv)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    iss_path = run_iss(args)
    run_oos(args, iss_file=iss_path)


if __name__ == "__main__":
    main()
