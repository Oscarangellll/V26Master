from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from plot_scripts.style import finalize_figure, prepare_output_path, save_table


def _require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def _plot_series(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str, label: str) -> None:
    _require_columns(df, [x_col, y_col], label)
    df = df[[x_col, y_col]].dropna().sort_values(x_col)
    if df.empty:
        raise ValueError(f"{label} has no rows after dropping missing values.")
    ax.plot(df[x_col], df[y_col], marker="o", label=label)


def register_parser(subparsers: argparse._SubParsersAction, common_parser: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "runtime-comparison",
        help="Compare runtime between MIP ISS and con ISS, and compare OOS objectives.",
        parents=[common_parser],
    )
    parser.add_argument("--mip-iss-file", default="1W1B/mip/ISS.csv")
    parser.add_argument("--con-iss-file", default="1W1B/con_mp/ISS.csv")
    parser.add_argument("--mip-oos-file", default="1W1B/mip/OSS.csv")
    parser.add_argument("--con-oos-file", default="1W1B/con_mp/OSS.csv")
    parser.set_defaults(func=run)


def run(args) -> None:
    mip_iss = pd.read_csv(args.mip_iss_file)
    con_iss = pd.read_csv(args.con_iss_file)
    mip_oos = pd.read_csv(args.mip_oos_file)
    con_oos = pd.read_csv(args.con_oos_file)

    fig, (ax_runtime, ax_oos) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    _plot_series(ax_runtime, mip_iss, "tree_size", "MIP_runtime", "MIP ISS")
    _plot_series(ax_runtime, con_iss, "tree_size", "Con_total runtime", "con ISS")
    ax_runtime.set_title("Runtime vs tree size")
    ax_runtime.set_xlabel("Tree size")
    ax_runtime.set_ylabel("Runtime [s]")
    ax_runtime.legend()

    _plot_series(ax_oos, mip_oos, "tree_size", "objective", "MIP OOS")
    _plot_series(ax_oos, con_oos, "tree_size", "objective", "con OOS")
    ax_oos.set_title("OOS objective vs tree size")
    ax_oos.set_xlabel("Tree size")
    ax_oos.set_ylabel("Objective value")
    ax_oos.legend()

    runtime_summary = pd.DataFrame(
        {
            "source": ["MIP ISS", "con ISS"],
            "mean_runtime": [mip_iss["MIP_runtime"].mean(), con_iss["Con_total runtime"].mean()],
        }
    )
    save_table(runtime_summary, args.action, prepare_output_path(args.table_dir, "runtime_comparison_summary.csv"))

    finalize_figure(fig, args.action, prepare_output_path(args.output_dir, "runtime_comparison.png"))