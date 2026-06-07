import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib.lines import Line2D

from plot_scripts.config import FIGWIDTH, PLOT_DIR, colors


CASES = ["1W1B", "2W2B", "3W2B", "4W3B"]
TREE_SIZES = [1, 3, 5, 7, 10, 15, 20]
MIN_INSTANCES = 19
MAX_MIP_TREE_SIZE_BY_CASE = {
    "3W2B": 10,
    "4W3B": 3,
}


def _runtime_average(path, column, count_path):
    df = pd.read_csv(path)[["tree_size", column]]
    count_df = pd.read_csv(count_path)[["tree_size", "count"]]
    counts = count_df.groupby("tree_size")["count"].sum().to_dict()

    rows = []
    for tree_size, group in df.groupby("tree_size"):
        valid = group[column].dropna()
        if len(valid) > 0:
            rows.append(
                {
                    "tree_size": tree_size,
                    "avg": valid.mean(),
                    "n": counts.get(tree_size, 0),
                }
            )
    return pd.DataFrame(rows).sort_values("tree_size")


def _plot_runtime(ax, df, color, marker, label):
    if df.empty:
        ax.plot([], [], color=color, marker=marker, label=label)
        return
    ax.scatter(df["tree_size"], df["avg"], color=color, marker=marker, label=label, zorder=3)
    df = df.sort_values("tree_size").reset_index(drop=True)
    for idx in range(len(df) - 1):
        segment = df.iloc[idx : idx + 2]
        linestyle = "-" if (segment["n"] >= MIN_INSTANCES).all() else "--"
        ax.plot(
            segment["tree_size"],
            segment["avg"],
            color=color,
            linestyle=linestyle,
            linewidth=1.4,
        )


def _filter_mip_tree_sizes(case, df):
    max_tree_size = MAX_MIP_TREE_SIZE_BY_CASE.get(case)
    if max_tree_size is None:
        return df
    return df[df["tree_size"] <= max_tree_size]


def plot_runtime_comparison():
    output_dir = Path(PLOT_DIR) / "ISS_OSS"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(FIGWIDTH / 2.54, 8 / 2.54))

    for ax, case in zip(axs.flat, CASES):
        mip = _runtime_average(
            f"results/stability/{case}/mip/ISS.csv",
            "MIP_runtime",
            f"results/stability/{case}/mip/OSS.csv",
        )
        con = _runtime_average(
            f"results/stability/{case}/con_mp/ISS.csv",
            "Con_total runtime",
            f"results/stability/{case}/con_mp/OSS.csv",
        )
        mip = _filter_mip_tree_sizes(case, mip)

        _plot_runtime(ax, mip, colors.direct_mip, "o", "Direct MIP")
        _plot_runtime(ax, con, colors.consensus_oss, "s", "Consensus")

        ax.set_title(case)
        ax.set_xticks(TREE_SIZES)
        ax.grid(color="0.90", linewidth=0.5)

    fig.supxlabel("Tree size", y=0.04)
    fig.supylabel("Runtime [s]", x=0.04)
    handles = [
        Line2D([0], [0], color=colors.direct_mip, marker="o", linewidth=1.4, label="Direct MIP"),
        Line2D([0], [0], color=colors.consensus_oss, marker="s", linewidth=1.4, label="Consensus"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.12, right=0.98, hspace=0.45, wspace=0.32)

    fig.savefig(output_dir / "runtime_comparison.svg")
    plt.close(fig)
