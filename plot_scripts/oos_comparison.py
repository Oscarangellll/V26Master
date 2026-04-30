from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from plot_scripts.style import finalize_figure, prepare_output_path, save_table


def _plot_series(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str, label: str) -> None:
    missing = [column for column in [x_col, y_col] if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")
    df = df[[x_col, y_col]].dropna().sort_values(x_col)
    if df.empty:
        raise ValueError(f"{label} has no rows after dropping missing values.")
    ax.plot(df[x_col], df[y_col], marker="o", label=label)


def register_parser(subparsers: argparse._SubParsersAction, common_parser: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "oos-comparison",
        help="Compare out-of-sample objectives for MIP and con solutions.",
        parents=[common_parser],
    )
    parser.add_argument("--mip-oos-file", default="1W1B/mip/OSS.csv")
    parser.add_argument("--con-oos-file", default="1W1B/con_mp/OSS.csv")
    parser.set_defaults(func=run)


def run(args) -> None:
    mip_oos = pd.read_csv(args.mip_oos_file)
    con_oos = pd.read_csv(args.con_oos_file)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    _plot_series(ax, mip_oos, "tree_size", "objective", "MIP OOS")
    _plot_series(ax, con_oos, "tree_size", "objective", "con OOS")

    ax.set_title("Out-of-sample comparison")
    ax.set_xlabel("Tree size")
    ax.set_ylabel("Objective value")
    ax.legend()

    comparison_summary = pd.DataFrame(
        {
            "source": ["MIP OOS", "con OOS"],
            "mean_objective": [mip_oos["objective"].mean(), con_oos["objective"].mean()],
        }
    )
    save_table(comparison_summary, args.action, prepare_output_path(args.table_dir, "oos_comparison_summary.csv"))

    finalize_figure(fig, args.action, prepare_output_path(args.output_dir, "oos_comparison.png"))