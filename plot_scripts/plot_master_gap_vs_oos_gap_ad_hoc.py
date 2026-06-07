from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_scripts.config import FIGWIDTH, PLOT_DIR, colors


CASES = ["1W1B", "2W2B", "3W2B", "4W3B"]
TREE_SIZES = [1, 3, 5, 7, 10, 15, 20]
MIN_EVALUATIONS = 19


def _iss_master_gap(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)[["tree_size", "objective", "MIPGap"]]
    df["objective"] = pd.to_numeric(df["objective"], errors="coerce")
    df["MIPGap"] = pd.to_numeric(df["MIPGap"], errors="coerce")
    df.loc[~np.isfinite(df["MIPGap"]), "MIPGap"] = pd.NA

    rows = []
    for tree_size, group in df.groupby("tree_size"):
        valid_obj = group["objective"].dropna()
        valid_gap = group["MIPGap"].dropna()
        if len(valid_obj) >= MIN_EVALUATIONS and len(valid_gap) > 0:
            rows.append(
                {
                    "tree_size": tree_size,
                    "master_gap_pct": 100 * valid_gap.mean(),
                }
            )
    return pd.DataFrame(rows)


def _oss_gap(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)[["tree_size", "count", "MIPGap"]]
    df["MIPGap"] = pd.to_numeric(df["MIPGap"], errors="coerce")
    df.loc[~np.isfinite(df["MIPGap"]), "MIPGap"] = pd.NA

    rows = []
    for tree_size, group in df.groupby("tree_size"):
        valid = group.dropna(subset=["MIPGap"])
        total_count = valid["count"].sum()
        if total_count >= MIN_EVALUATIONS:
            weighted_gap = (valid["MIPGap"] * valid["count"]).sum() / total_count
            rows.append({"tree_size": tree_size, "oss_gap_pct": 100 * weighted_gap})
    return pd.DataFrame(rows)


def _case_gaps(case: str) -> pd.DataFrame:
    base = Path("results") / "stability" / case / "con_mp"
    df = _iss_master_gap(base / "ISS.csv").merge(_oss_gap(base / "OSS.csv"), on="tree_size")
    return df.sort_values("tree_size")


def plot_master_gap_vs_oos_gap_ad_hoc() -> None:
    output_dir = Path(PLOT_DIR) / "ISS_OSS"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(FIGWIDTH / 2.54, 8 / 2.54))

    for ax, case in zip(axs.flat, CASES):
        df = _case_gaps(case)
        ax.axhline(0, color="0.35", linewidth=0.8)
        ax.plot(
            df["tree_size"],
            df["master_gap_pct"],
            color=colors.purple,
            marker="D",
            linewidth=1.4,
        )
        ax.plot(
            df["tree_size"],
            df["oss_gap_pct"],
            color=colors.consensus_oss,
            marker="s",
            linewidth=1.4,
        )
        ax.set_title(case)
        ax.set_xticks(TREE_SIZES)
        ax.grid(color="0.90", linewidth=0.5)

    fig.supxlabel("Tree size", y=0.04)
    fig.supylabel("Gap [%]", x=0.04)
    handles = [
        Line2D(
            [0],
            [0],
            color=colors.purple,
            marker="D",
            linewidth=1.4,
            label="Restricted master MIP gap",
        ),
        Line2D(
            [0],
            [0],
            color=colors.consensus_oss,
            marker="s",
            linewidth=1.4,
            label="OSS evaluation MIP gap",
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

    out = output_dir / "master_gap_vs_oos_gap_ad_hoc.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    plot_master_gap_vs_oos_gap_ad_hoc()
