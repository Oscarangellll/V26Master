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

# Compute ISS statistics per tree size
iss_stats = []
for ts, g in iss.groupby("tree_size"):
    n = len(g)
    mean = g["objective"].mean()
    sd = g["objective"].std(ddof=1) if n > 1 else 0.0
    cv = sd / mean if mean != 0 else np.nan
    se = sd / np.sqrt(max(n, 1))
    crit = t.ppf(0.975, max(n - 1, 1))
    lo = mean - crit * se
    hi = mean + crit * se
    iss_stats.append({"tree_size": ts, "mean": mean, "sd": sd, "cv": cv, "lo": lo, "hi": hi, "n": n})
iss_ci = pd.DataFrame(iss_stats)

# Compute OOS statistics per tree size (weighted by count)
oss_stats = []
for ts, g in oss.groupby("tree_size"):
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
    oss_stats.append({"tree_size": ts, "mean": weighted_mean, "sd": sd_weighted, "cv": cv, "lo": lo, "hi": hi, "n": n})
oss_ci = pd.DataFrame(oss_stats)

# Create two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 7))

# ========== Plot 1: ISS vs OOS curves ==========
ax1.plot(iss_ci["tree_size"], iss_ci["mean"], color="blue", marker="o", linewidth=2.5, markersize=7, label="In-sample (ISS)")
ax1.plot(oss_ci["tree_size"], oss_ci["mean"], color="red", marker="s", linewidth=2.5, markersize=7, label="Out-of-sample (OOS)")
ax1.set_xlabel("Tree size", fontsize=12)
ax1.set_ylabel("Objective value", fontsize=12)
ax1.set_title("Stability: ISS vs OOS mean objectives", fontsize=13, fontweight="bold")
ax1.legend(fontsize=11, loc="best")
ax1.grid(True, alpha=0.3)

# ========== Plot 2: Coefficient of Variation (CV) ==========
ax2.plot(iss_ci["tree_size"], iss_ci["cv"], color="blue", marker="o", linewidth=2.5, markersize=7, label="ISS CV")
ax2.plot(oss_ci["tree_size"], oss_ci["cv"], color="red", marker="s", linewidth=2.5, markersize=7, label="OOS CV")
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
print("ISS (In-sample):")
print(iss_ci[["tree_size", "mean", "sd", "cv", "n"]].to_string(index=False))
print("\nOSS (Out-of-sample):")
print(oss_ci[["tree_size", "mean", "sd", "cv", "n"]].to_string(index=False))
print("\n")
