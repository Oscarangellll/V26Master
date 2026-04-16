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
        choices=["mip", "con_mp"],
        help="Solution method",
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
    parser.add_argument("--iss_pool_end", type=int, default=3000)
    parser.add_argument("--oos_pool_start", type=int, default=3001)
    parser.add_argument("--oos_pool_end", type=int, default=4000)
    parser.add_argument(
        "--gap_prune_threshold",
        type=float,
        default=0.10,
        help="Skip larger scenario tree sizes when an evaluated size exceeds this MIPGap threshold.",
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
        "--iss_output",
        default=None,
        help="Optional ISS output path (default: results/stability/<case>/<method>/ISS.csv)",
    )
    parser.add_argument(
        "--oos_output",
        default=None,
        help="Optional OOS output path (default: results/stability/<case>/<method>/OSS.csv)",
    )
    parser.add_argument(
        "--sampling_strategy",
        choices=["random", "stratified_bins"],
        default="random",
        help="Scenario sampling strategy.",
    )
    parser.add_argument(
        "--tail_fraction",
        type=float,
        default=0.20,
        help="y in [0, 0.5]: top y%% is kind bin, bottom y%% is harsh bin.",
    )
    parser.add_argument(
        "--prob_kind",
        type=float,
        default=0.20,
        help="Sampling probability mass for kind bin.",
    )
    parser.add_argument(
        "--prob_normal",
        type=float,
        default=0.60,
        help="Sampling probability mass for normal bin.",
    )
    parser.add_argument(
        "--prob_harsh",
        type=float,
        default=0.20,
        help="Sampling probability mass for harsh bin.",
    )
    parser.add_argument(
        "--kindness_metric",
        choices=[
            "count_location_days_over_threshold",
            "total_window_hours",
            "count_location_days_under_threshold",
            "max_bad_streak_under_threshold",
        ],
        default="count_location_days_over_threshold",
        help="Axis used to rank scenarios from kind to harsh.",
    )
    parser.add_argument(
        "--kindness_ww_threshold",
        type=float,
        default=8.0,
        help="Threshold used by threshold-based kindness metrics.",
    )
    parser.add_argument(
        "--kindness_vessel",
        default=None,
        help="Optional vessel name for kindness metric. Defaults to first vessel type in case.",
    )
    return parser


def main() -> None:   
    parser = build_parser()
    args = parser.parse_args()

    iss_path = run_iss(args)
    run_oos(args, iss_file=iss_path)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn")
    
    main()
