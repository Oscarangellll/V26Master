from plot_scripts.stability import run


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot ISS vs OOS stability curves")
    parser.add_argument("--iss-file", default="results/stability/1W1B/mip/ISS.csv")
    parser.add_argument("--oss-file", default="results/stability/1W1B/mip/OSS.csv")
    parser.add_argument("--no_show", action="store_true")
    parser.add_argument("--output-dir", default="plot_scripts/plots")
    parser.add_argument("--table-dir", default="plot_scripts/tables")
    parser.add_argument("--mode", choices=["both", "iss-oss", "cv"], default="both")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--tree-sizes", type=int, nargs="+", default=None)

    args = parser.parse_args()
    args.action = "none" if args.no_show else "show"
    run(args)
