import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from plot_scripts.config import FIGWIDTH, PLOT_DIR, colors


CASES = ["1W1B", "2W2B", "3W2B", "4W3B"]
TREE_SIZES = [1, 3, 5, 7, 10, 15, 20]
MIN_INSTANCES = 19


def iss(path):
    df = pd.read_csv(path)[["tree_size", "objective"]]
    rows = []
    for tree_size, group in df.groupby("tree_size"):
        valid = group["objective"].dropna()
        if len(valid) >= MIN_INSTANCES:
            rows.append({"tree_size": tree_size, "avg": valid.mean(), "n": len(valid)})
    return pd.DataFrame(rows).sort_values("tree_size")


def oss(path):
    df = pd.read_csv(path)[["tree_size", "count", "objective"]]
    df = df[df["objective"].notna() & (df["objective"] > 0)]
    rows = []
    for tree_size, group in df.groupby("tree_size"):
        total_count = group["count"].sum()
        if total_count >= MIN_INSTANCES:
            weighted_avg = (group["objective"] * group["count"]).sum() / total_count
            rows.append({"tree_size": tree_size, "avg": weighted_avg, "n": total_count})
    return pd.DataFrame(rows).sort_values("tree_size")


def _plot(ax, df, color, marker, label):
    if df.empty:
        ax.plot([], [], color=color, marker=marker, label=label)
        return
    ax.plot(df["tree_size"], df["avg"] / 1e6, color=color, marker=marker, label=label, linewidth=1.4)


def _axis_bounds_from_iss(case_data):
    lower_bounds = {}
    required_spans = []
    for case, (df_iss, df_oss) in case_data.items():
        iss_values = df_iss["avg"] / 1e6
        all_values = pd.concat([df_iss["avg"], df_oss["avg"]]) / 1e6

        y_bottom = math.floor(iss_values.min() * 2) / 2
        lower_bounds[case] = y_bottom
        required_spans.append(all_values.max() - y_bottom)

    common_span = math.ceil(max(required_spans) * 2) / 2
    return lower_bounds, common_span


def plot_stability():
    output_dir = Path(PLOT_DIR) / "ISS_OSS"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(FIGWIDTH / 2.54, 8 / 2.54))

    case_data = {}
    for case in CASES:
        case_data[case] = (
            iss(f"results/stability/{case}/con_mp/ISS.csv"),
            oss(f"results/stability/{case}/con_mp/OSS.csv"),
        )
    lower_bounds, common_span = _axis_bounds_from_iss(case_data)

    for ax, case in zip(axs.flat, CASES):
        df_iss, df_oss = case_data[case]

        _plot(ax, df_iss, colors.consensus_iss, "o", "ISS")
        _plot(ax, df_oss, colors.consensus_oss, "s", "OSS")

        ax.set_title(case)
        ax.set_xticks(TREE_SIZES)
        ax.set_ylim(lower_bounds[case], lower_bounds[case] + common_span)
        ax.grid(color="0.90", linewidth=0.5)

    fig.supxlabel("Tree size", y=0.04)
    fig.supylabel("AOV [MEUR]", x=0.04)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.12, right=0.98, hspace=0.45, wspace=0.32)

    fig.savefig(output_dir / "stability.svg")
    plt.close(fig)
