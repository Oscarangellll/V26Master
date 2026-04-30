from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from plot_scripts.style import finalize_figure, prepare_output_path, save_table


def _series_tree_sizes(df: pd.DataFrame) -> pd.Series:
    if "tree_size" not in df.columns:
        return pd.Series(dtype=int)
    ts = pd.to_numeric(df["tree_size"], errors="coerce").dropna()
    return ts.astype(int)


def _plot_with_stop(ax: plt.Axes, df: pd.DataFrame, y_col: str, color: str, marker: str, label: str, stop_ts):
    if df.empty or y_col not in df.columns:
        return

    df_local = df[["tree_size", y_col]].dropna().sort_values("tree_size").copy()
    if df_local.empty:
        return

    if stop_ts is None:
        ax.plot(df_local["tree_size"], df_local[y_col], color=color, marker=marker, label=label)
        return

    solid_df = df_local[df_local["tree_size"] < stop_ts]
    stop_df = df_local[df_local["tree_size"] == stop_ts]

    if not solid_df.empty:
        ax.plot(solid_df["tree_size"], solid_df[y_col], color=color, marker=marker, label=label)
    else:
        ax.plot([], [], color=color, marker=marker, label=label)

    if not stop_df.empty:
        if not solid_df.empty:
            tail = pd.concat([solid_df.tail(1), stop_df]).sort_values("tree_size")
            ax.plot(tail["tree_size"], tail[y_col], color=color, linestyle="--", alpha=0.8)
        ax.plot(stop_df["tree_size"], stop_df[y_col], color=color, marker=marker, linestyle="none", alpha=0.8)


def _compute_iss_stats(iss: pd.DataFrame) -> pd.DataFrame:
    iss_stats = []
    for ts, group in iss.groupby("tree_size"):
        valid = group[group["objective"].notna()]
        if len(valid) == 0:
            continue

        n = len(valid)
        mean = valid["objective"].mean()
        sd = valid["objective"].std(ddof=1) if n > 1 else 0.0
        cv = sd / mean if mean != 0 else np.nan
        se = sd / np.sqrt(max(n, 1))
        crit = t.ppf(0.975, max(n - 1, 1))
        lo = mean - crit * se
        hi = mean + crit * se

        iss_stats.append(
            {
                "tree_size": ts,
                "mean": mean,
                "sd": sd,
                "cv": cv,
                "lo": lo,
                "hi": hi,
                "n": n,
                "is_pruned": len(group) > n,
            }
        )

    return pd.DataFrame(iss_stats)


def _compute_oss_stats(oss: pd.DataFrame) -> pd.DataFrame:
    valid_oss = oss[oss["objective"].notna() & (oss["objective"] > 0)].copy()
    oss_stats = []
    for ts, group in valid_oss.groupby("tree_size"):
        n = len(group)
        weighted_mean = (group["objective"] * group["count"]).sum() / group["count"].sum()
        var_weighted = ((group["objective"] - weighted_mean) ** 2 * group["count"]).sum() / group["count"].sum()
        sd_weighted = np.sqrt(var_weighted)
        cv = sd_weighted / weighted_mean if weighted_mean != 0 else np.nan
        se = sd_weighted / np.sqrt(max(n, 1))
        crit = t.ppf(0.975, max(n - 1, 1))
        lo = weighted_mean - crit * se
        hi = weighted_mean + crit * se

        oss_stats.append(
            {
                "tree_size": ts,
                "mean": weighted_mean,
                "sd": sd_weighted,
                "cv": cv,
                "lo": lo,
                "hi": hi,
                "n": n,
            }
        )

    return pd.DataFrame(oss_stats)


def register_parser(subparsers: argparse._SubParsersAction, common_parser: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "stability",
        help="Plot ISS/OOS stability curves or CV curves.",
        parents=[common_parser],
    )
    parser.add_argument("--iss-file", default="results/stability/1W1B/mip/ISS.csv")
    parser.add_argument("--oss-file", default="results/stability/1W1B/mip/OSS.csv")
    parser.add_argument(
        "--mode",
        choices=["both", "iss-oss", "cv"],
        default="both",
        help="Choose which stability plot to generate.",
    )
    parser.add_argument("--no-summary", action="store_true", help="Skip printing summary tables.")
    parser.add_argument(
        "--tree-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit tree sizes for the x-axis.",
    )
    parser.set_defaults(func=run)


def _build_outputs(args, iss_ci: pd.DataFrame, oss_ci: pd.DataFrame, stop_ts):
    x_tree_sizes = sorted(set(args.tree_sizes)) if args.tree_sizes else sorted(
        set(_series_tree_sizes(iss_ci)).union(set(_series_tree_sizes(oss_ci)))
    )

    outputs = []

    if args.mode in {"both", "iss-oss"}:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        _plot_with_stop(ax, iss_ci, "mean", color="tab:blue", marker="o", label="In-sample (ISS)", stop_ts=stop_ts)
        _plot_with_stop(ax, oss_ci, "mean", color="tab:red", marker="s", label="Out-of-sample (OOS)", stop_ts=stop_ts)
        ax.set_title("Stability: ISS vs OOS mean objectives")
        ax.set_xlabel("Tree size")
        ax.set_ylabel("Objective value")
        ax.legend()
        if x_tree_sizes:
            ax.set_xticks(x_tree_sizes)
        outputs.append((fig, "stability_iss_oss.png"))

    if args.mode in {"both", "cv"}:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        _plot_with_stop(ax, iss_ci, "cv", color="tab:blue", marker="o", label="ISS CV", stop_ts=stop_ts)
        _plot_with_stop(ax, oss_ci, "cv", color="tab:red", marker="s", label="OOS CV", stop_ts=stop_ts)
        ax.set_title("Relative variability: CV across instances")
        ax.set_xlabel("Tree size")
        ax.set_ylabel("Coefficient of variation")
        ax.legend()
        if x_tree_sizes:
            ax.set_xticks(x_tree_sizes)
        outputs.append((fig, "stability_cv.png"))

    return outputs


def run(args) -> None:
    iss = pd.read_csv(args.iss_file)
    oss = pd.read_csv(args.oss_file)

    iss_ci = _compute_iss_stats(iss)
    oss_ci = _compute_oss_stats(oss)

    stop_ts = None
    if not iss_ci.empty and "is_pruned" in iss_ci.columns:
        pruned = iss_ci[iss_ci["is_pruned"]].sort_values("tree_size")
        if not pruned.empty:
            stop_ts = int(pruned.iloc[0]["tree_size"])

    save_table(iss_ci, args.action, prepare_output_path(args.table_dir, "stability_iss_summary.csv"))
    save_table(oss_ci, args.action, prepare_output_path(args.table_dir, "stability_oos_summary.csv"))

    outputs = _build_outputs(args, iss_ci, oss_ci, stop_ts)
    for fig, filename in outputs:
        finalize_figure(fig, args.action, prepare_output_path(args.output_dir, filename))

    if not args.no_summary:
        print("\n=== Summary Statistics ===\n")
        print("ISS (In-sample):\n")
        print(iss_ci[["tree_size", "mean", "sd", "cv", "n", "is_pruned"]].to_string(index=False))
        print("\nOSS (Out-of-sample):\n")
        print(oss_ci[["tree_size", "mean", "sd", "cv", "n"]].to_string(index=False))
        if stop_ts is not None:
            print(f"\nDetected stop point from ISS pruning: tree_size={stop_ts}")