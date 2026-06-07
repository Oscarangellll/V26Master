import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib.lines import Line2D
from math import ceil

from plot_scripts.config import PLOT_DIR, colors, FIGWIDTH

LINEWIDTH = 1.4
Y_AXIS_SPAN = 3.0
Y_AXIS_OVERRIDES = {
    "2W2B": (28.0, 31.0),
    "3W2B": (38.0, 41.0),
}

def compute_weighted_avg(path):
    df = pd.read_csv(path)[["tree_size", "count", "objective"]]

    results = []

    for tree_size, group in df.groupby("tree_size"):
        total_count = group["count"].sum()

        if total_count >= 10:
            weighted_avg = (group["objective"] * group["count"]).sum() / total_count

            results.append({
                "tree_size": tree_size,
                "weighted_avg": weighted_avg,
                "count": total_count,
            })

    return pd.DataFrame(results)


def plot_with_count_style(ax, df, color, marker):
    ax.scatter(
        df["tree_size"],
        df["weighted_avg"] / 1e6,
        color=color,
        marker=marker,
        zorder=3,
    )

    df = df.sort_values("tree_size").reset_index(drop=True)
    for idx in range(len(df) - 1):
        segment = df.iloc[idx : idx + 2]
        linestyle = "-" if (segment["count"] >= 19).all() else "--"
        ax.plot(
            segment["tree_size"],
            segment["weighted_avg"] / 1e6,
            color=color,
            linestyle=linestyle,
            linewidth=LINEWIDTH,
        )


def set_common_y_span(ax, case, *dfs):
    if case in Y_AXIS_OVERRIDES:
        ax.set_ylim(*Y_AXIS_OVERRIDES[case])
        return

    values = []
    for df in dfs:
        values.extend((df["weighted_avg"] / 1e6).tolist())
    y_top = ceil(max(values) * 2) / 2
    ax.set_ylim(y_top - Y_AXIS_SPAN, y_top)


def plot_oss_con_mip():
    output_dir = Path(PLOT_DIR) / "ISS_OSS"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(FIGWIDTH / 2.54, 8 / 2.54))

    cases = ["1W1B", "2W2B", "3W2B", "4W3B"]

    for ax, case in zip(axs.flat, cases):

        df_mip = compute_weighted_avg(f"results/stability/{case}/mip/OSS.csv")
        df_con = compute_weighted_avg(f"results/stability/{case}/con_mp/OSS.csv")

        plot_with_count_style(ax, df_mip, colors.direct_mip, "o")
        plot_with_count_style(ax, df_con, colors.consensus_oss, "s")
        set_common_y_span(ax, case, df_mip, df_con)

        ax.set_title(case)
        ax.set_xticks([1, 3, 5, 7, 10, 15, 20])
        ax.grid(color="0.90", linewidth=0.5)

    fig.supxlabel("Tree size", y=0.04)
    fig.supylabel("AOV [MEUR]", x=0.04)
    handles = [
        Line2D(
            [0],
            [0],
            color=colors.direct_mip,
            marker="o",
            linewidth=1.4,
            label="Direct MIP",
        ),
        Line2D(
            [0],
            [0],
            color=colors.consensus_oss,
            marker="s",
            linewidth=1.4,
            label="Consensus",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.12, right=0.98, hspace=0.45, wspace=0.32)
    fig.savefig(output_dir / "oss_con_mip.svg")
    plt.close(fig)
