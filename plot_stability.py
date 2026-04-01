import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

parser = argparse.ArgumentParser(description="Plot ISS vs OOS stability curves")
parser.add_argument(
    "--iss_file",
    default="results/stability/1W1B/mip/ISS.csv",
    help="Path to ISS.csv",
)
parser.add_argument(
    "--oss_file",
    default="results/stability/1W1B/mip/OSS.csv",
    help="Path to OSS.csv",
)
parser.add_argument(
    "--no_show",
    action="store_true",
    help="Skip opening the interactive plot window",
)
parser.add_argument(
    "--tree_sizes",
    type=int,
    nargs="+",
    default=None,
    help="Optional explicit tree sizes for x-axis (e.g. 1 3 5 7 9 11 13 15)",
)
args = parser.parse_args()

# Read the results
iss = pd.read_csv(args.iss_file)
oss = pd.read_csv(args.oss_file)


def _series_tree_sizes(df: pd.DataFrame) -> pd.Series:
    if "tree_size" not in df.columns:
        return pd.Series(dtype=int)
    ts = pd.to_numeric(df["tree_size"], errors="coerce").dropna()
    return ts.astype(int)


def _plot_with_stop(ax, df: pd.DataFrame, y_col: str, color: str, marker: str, label: str, stop_ts):
    if df.empty or y_col not in df.columns:
        return

    df_local = df[["tree_size", y_col]].dropna().sort_values("tree_size").copy()
    if df_local.empty:
        return

    if stop_ts is None:
        ax.plot(
            df_local["tree_size"],
            df_local[y_col],
            color=color,
            marker=marker,
            linewidth=2.5,
            markersize=7,
            label=label,
        )
        return

    solid_df = df_local[df_local["tree_size"] < stop_ts]
    stop_df = df_local[df_local["tree_size"] == stop_ts]

    if not solid_df.empty:
        ax.plot(
            solid_df["tree_size"],
            solid_df[y_col],
            color=color,
            marker=marker,
            linewidth=2.5,
            markersize=7,
            label=label,
        )
    else:
        ax.plot([], [], color=color, marker=marker, linewidth=2.5, markersize=7, label=label)

    if not stop_df.empty:
        if not solid_df.empty:
            tail = pd.concat([solid_df.tail(1), stop_df]).sort_values("tree_size")
            ax.plot(
                tail["tree_size"],
                tail[y_col],
                color=color,
                linestyle="--",
                linewidth=2.5,
                alpha=0.8,
            )
        ax.plot(
            stop_df["tree_size"],
            stop_df[y_col],
            color=color,
            marker=marker,
            markersize=7,
            linestyle="none",
            alpha=0.8,
        )

# Filter out pruned/empty solutions from OOS (where mean objective is 0 or null)
oss_valid = oss[oss["objective"].notna() & (oss["objective"] > 0)].copy()

# Compute ISS statistics per tree size
iss_stats = []
for ts, g in iss.groupby("tree_size"):
    # Filter out None/NaN objectives
    g_valid = g[g["objective"].notna()]
    if len(g_valid) == 0:
        continue
    
    n = len(g_valid)
    mean = g_valid["objective"].mean()
    sd = g_valid["objective"].std(ddof=1) if n > 1 else 0.0
    cv = sd / mean if mean != 0 else np.nan
    se = sd / np.sqrt(max(n, 1))
    crit = t.ppf(0.975, max(n - 1, 1))
    lo = mean - crit * se
    hi = mean + crit * se
    
    # Flag if some instances were pruned (solutions were empty)
    n_pruned = len(g) - n
    is_pruned = n_pruned > 0
    
    iss_stats.append({
        "tree_size": ts, 
        "mean": mean, 
        "sd": sd, 
        "cv": cv, 
        "lo": lo, 
        "hi": hi, 
        "n": n, 
        "is_pruned": is_pruned
    })
iss_ci = pd.DataFrame(iss_stats)

# Compute OOS statistics per tree size (weighted by count) - skip invalid rows
oss_stats = []
for ts, g in oss_valid.groupby("tree_size"):
    n = len(g)
    weighted_mean = (g["objective"] * g["count"]).sum() / g["count"].sum()
    
    # Compute variance of weighted average
    var_weighted = ((g["objective"] - weighted_mean) ** 2 * g["count"]).sum() / g["count"].sum()
    sd_weighted = np.sqrt(var_weighted)
    cv = sd_weighted / weighted_mean if weighted_mean != 0 else np.nan
    
    se = sd_weighted / np.sqrt(max(n, 1))
    crit = t.ppf(0.975, max(n - 1, 1))
    lo = weighted_mean - crit * se
    hi = weighted_mean + crit * se
    oss_stats.append({
        "tree_size": ts, 
        "mean": weighted_mean, 
        "sd": sd_weighted, 
        "cv": cv, 
        "lo": lo, 
        "hi": hi, 
        "n": n
    })
oss_ci = pd.DataFrame(oss_stats)

# Create two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 7))

# Define x-axis tree sizes from input argument or data
if args.tree_sizes:
    x_tree_sizes = sorted(set(args.tree_sizes))
else:
    x_tree_sizes = sorted(set(_series_tree_sizes(iss)).union(set(_series_tree_sizes(oss))))

# Detect first prune point in ISS and use it as visual stop point
stop_ts = None
if not iss_ci.empty and "is_pruned" in iss_ci.columns:
    pruned = iss_ci[iss_ci["is_pruned"]].sort_values("tree_size")
    if not pruned.empty:
        stop_ts = int(pruned.iloc[0]["tree_size"])

# ========== Plot 1: ISS vs OOS curves ==========
_plot_with_stop(ax1, iss_ci, "mean", color="blue", marker="o", label="In-sample (ISS)", stop_ts=stop_ts)
_plot_with_stop(ax1, oss_ci, "mean", color="red", marker="s", label="Out-of-sample (OOS)", stop_ts=stop_ts)

ax1.set_xlabel("Tree size", fontsize=12)
ax1.set_ylabel("Objective value", fontsize=12)
ax1.set_title("Stability: ISS vs OOS mean objectives", fontsize=13, fontweight="bold")
ax1.legend(fontsize=11, loc="best")
ax1.grid(True, alpha=0.3)
if x_tree_sizes:
    ax1.set_xticks(x_tree_sizes)

# ========== Plot 2: Coefficient of Variation (CV) ==========
_plot_with_stop(ax2, iss_ci, "cv", color="blue", marker="o", label="ISS CV", stop_ts=stop_ts)
_plot_with_stop(ax2, oss_ci, "cv", color="red", marker="s", label="OOS CV", stop_ts=stop_ts)

ax2.set_xlabel("Tree size", fontsize=12)
ax2.set_ylabel("Coefficient of Variation (std/mean)", fontsize=12)
ax2.set_title("Relative variability: CV across instances", fontsize=13, fontweight="bold")
ax2.legend(fontsize=11, loc="best")
ax2.grid(True, alpha=0.3)
if x_tree_sizes:
    ax2.set_xticks(x_tree_sizes)

plt.tight_layout()
if not args.no_show:
    plt.show()

# Optional: print summary table
print("\n=== Summary Statistics ===\n")
print("ISS (In-sample) - stipled segment ends at first pruned tree size:\n")
print(iss_ci[["tree_size", "mean", "sd", "cv", "n", "is_pruned"]].to_string(index=False))
print("\nOSS (Out-of-sample) - invalid/pruned solutions excluded:\n")
print(oss_ci[["tree_size", "mean", "sd", "cv", "n"]].to_string(index=False))
if stop_ts is not None:
    print(f"\nDetected stop point from ISS pruning: tree_size={stop_ts}")
print("\n")
