from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_scripts.config import FIGWIDTH, PLOT_DIR, colors


CASES = ["1W1B", "2W2B", "3W2B", "4W3B"]
TREE_SIZES = [1, 3, 5, 7, 10, 15, 20]
MIN_EVALUATIONS = 10
MIN_COMPLETE_EVALUATIONS = 19


def compute_weighted_avg(path):
    df = pd.read_csv(path)[["tree_size", "count", "objective"]]
    df = df[pd.to_numeric(df["objective"], errors="coerce").notna()].copy()

    rows = []
    for tree_size, group in df.groupby("tree_size"):
        total_count = group["count"].sum()
        if total_count >= MIN_EVALUATIONS:
            weighted_avg = (group["objective"] * group["count"]).sum() / total_count
            rows.append(
                {
                    "tree_size": int(tree_size),
                    "weighted_avg": weighted_avg,
                    "count": int(total_count),
                }
            )

    return pd.DataFrame(rows)


def compute_gap(case):
    base = Path("results") / "stability" / case
    mip = compute_weighted_avg(base / "mip" / "OSS.csv")
    con = compute_weighted_avg(base / "con_mp" / "OSS.csv")

    if mip.empty or con.empty:
        return pd.DataFrame()

    df = mip.merge(con, on="tree_size", suffixes=("_mip", "_con"))
    df["gap_pct"] = (
        100 * (df["weighted_avg_con"] - df["weighted_avg_mip"]) / df["weighted_avg_mip"]
    )
    return df.sort_values("tree_size")


def plot_with_count_style(ax, df, y_col, color, marker, count_cols):
    ax.scatter(
        df["tree_size"],
        df[y_col],
        color=color,
        marker=marker,
        zorder=3,
    )

    df = df.sort_values("tree_size").reset_index(drop=True)
    for idx in range(len(df) - 1):
        segment = df.iloc[idx : idx + 2]
        complete = all((segment[col] >= MIN_COMPLETE_EVALUATIONS).all() for col in count_cols)
        linestyle = "-" if complete else "--"
        ax.plot(
            segment["tree_size"],
            segment[y_col],
            color=color,
            linestyle=linestyle,
            linewidth=1.4,
        )


def plot_oss_con_mip_gap():
    output_dir = Path(PLOT_DIR) / "ISS_OSS"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(FIGWIDTH / 2.54, 8 / 2.54))

    for ax, case in zip(axs.flat, CASES):
        df = compute_gap(case)

        ax.axhline(0, color="0.35", linewidth=0.8)
        if df.empty:
            ax.text(0.5, 0.5, "No overlap", ha="center", va="center", transform=ax.transAxes)
        else:
            df["mip_baseline"] = 0.0
            plot_with_count_style(
                ax,
                df,
                "mip_baseline",
                colors.direct_mip,
                "o",
                ["count_mip"],
            )
            plot_with_count_style(
                ax,
                df,
                "gap_pct",
                colors.consensus_oss,
                "s",
                ["count_mip", "count_con"],
            )

        ax.set_title(case)
        ax.set_xticks(TREE_SIZES)
        ax.set_ylim(-1, 2)
        ax.grid(color="0.90", linewidth=0.5)

    fig.supxlabel("Scenario tree size", y=0.04)
    fig.supylabel(r"Relative difference from $AOV_{\mathrm{out}}^{\mathrm{MIP}}$ [%]", x=0.04)
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

    fig.savefig(output_dir / "oss_con_mip_gap.svg")
    plt.close(fig)


if __name__ == "__main__":
    plot_oss_con_mip_gap()
