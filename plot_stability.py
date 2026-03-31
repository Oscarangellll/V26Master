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
args = parser.parse_args()

# Read the results
iss = pd.read_csv(args.iss_file)
oss = pd.read_csv(args.oss_file)

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

# Find which tree_sizes had pruning in ISS
pruned_tree_sizes = set(iss_ci[iss_ci["is_pruned"]]["tree_size"].values)

# Separate OOS data by whether corresponding ISS tree_size was pruned
oss_normal = oss_ci[~oss_ci["tree_size"].isin(pruned_tree_sizes)]
oss_unreliable = oss_ci[oss_ci["tree_size"].isin(pruned_tree_sizes)]

# ========== Plot 1: ISS vs OOS curves ==========
# Separate pruned and non-pruned ISS data for plotting with different line styles
iss_normal = iss_ci[~iss_ci["is_pruned"]]
iss_pruned = iss_ci[iss_ci["is_pruned"]]

# Plot normal ISS points and line
if len(iss_normal) > 0:
    ax1.plot(iss_normal["tree_size"], iss_normal["mean"], 
             color="blue", marker="o", linewidth=2.5, markersize=7, label="In-sample (ISS)")
    # If there are pruned points, extend line with dashes
    if len(iss_pruned) > 0:
        combined_iss = pd.concat([iss_normal, iss_pruned]).sort_values("tree_size")
        ax1.plot(combined_iss["tree_size"], combined_iss["mean"], 
                color="blue", linestyle="--", linewidth=2.5, alpha=0.6)
        # Highlight pruned points
        ax1.plot(iss_pruned["tree_size"], iss_pruned["mean"], 
                color="blue", marker="o", markersize=7, linestyle="none", alpha=0.6)

# Plot normal OOS
if len(oss_normal) > 0:
    ax1.plot(oss_normal["tree_size"], oss_normal["mean"], 
             color="red", marker="s", linewidth=2.5, markersize=7, label="Out-of-sample (OOS)")
    # If there are unreliable points, extend line with dashes
    if len(oss_unreliable) > 0:
        combined_oss = pd.concat([oss_normal, oss_unreliable]).sort_values("tree_size")
        ax1.plot(combined_oss["tree_size"], combined_oss["mean"], 
                color="red", linestyle="--", linewidth=2.5, alpha=0.6)
        # Highlight unreliable points
        ax1.plot(oss_unreliable["tree_size"], oss_unreliable["mean"], 
                color="red", marker="s", markersize=7, linestyle="none", alpha=0.6)

ax1.set_xlabel("Tree size", fontsize=12)
ax1.set_ylabel("Objective value", fontsize=12)
ax1.set_title("Stability: ISS vs OOS mean objectives", fontsize=13, fontweight="bold")
ax1.legend(fontsize=11, loc="best")
ax1.grid(True, alpha=0.3)

# ========== Plot 2: Coefficient of Variation (CV) ==========
# Separate pruned and non-pruned ISS CV for plotting
if len(iss_normal) > 0:
    ax2.plot(iss_normal["tree_size"], iss_normal["cv"], 
             color="blue", marker="o", linewidth=2.5, markersize=7, label="ISS CV")
    # If there are pruned points, extend line with dashes
    if len(iss_pruned) > 0:
        combined_iss = pd.concat([iss_normal, iss_pruned]).sort_values("tree_size")
        ax2.plot(combined_iss["tree_size"], combined_iss["cv"], 
                color="blue", linestyle="--", linewidth=2.5, alpha=0.6)
        # Highlight pruned points
        ax2.plot(iss_pruned["tree_size"], iss_pruned["cv"], 
                color="blue", marker="o", markersize=7, linestyle="none", alpha=0.6)

# Plot normal OOS CV
if len(oss_normal) > 0:
    ax2.plot(oss_normal["tree_size"], oss_normal["cv"], 
             color="red", marker="s", linewidth=2.5, markersize=7, label="OOS CV")
    # If there are unreliable points, extend line with dashes
    if len(oss_unreliable) > 0:
        combined_oss = pd.concat([oss_normal, oss_unreliable]).sort_values("tree_size")
        ax2.plot(combined_oss["tree_size"], combined_oss["cv"], 
                color="red", linestyle="--", linewidth=2.5, alpha=0.6)
        # Highlight unreliable points
        ax2.plot(oss_unreliable["tree_size"], oss_unreliable["cv"], 
                color="red", marker="s", markersize=7, linestyle="none", alpha=0.6)

ax2.set_xlabel("Tree size", fontsize=12)
ax2.set_ylabel("Coefficient of Variation (std/mean)", fontsize=12)
ax2.set_title("Relative variability: CV across instances", fontsize=13, fontweight="bold")
ax2.legend(fontsize=11, loc="best")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
if not args.no_show:
    plt.show()

# Optional: print summary table
print("\n=== Summary Statistics ===\n")
print("ISS (In-sample) - stipled lines indicate tree sizes where pruning occurred:\n")
print(iss_ci[["tree_size", "mean", "sd", "cv", "n", "is_pruned"]].to_string(index=False))
print("\nOSS (Out-of-sample) - invalid/pruned solutions excluded:\n")
print(oss_ci[["tree_size", "mean", "sd", "cv", "n"]].to_string(index=False))
print("\n")
