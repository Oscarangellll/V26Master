import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter

sys.path.append(str(Path(__file__).resolve().parents[1]))

from plot_scripts.config import PLOT_DIR, colors, FIGWIDTH


CASES = ["1W1B", "2W2B", "3W2B", "4W3B"]
TREE_SIZES = [1, 3, 5, 7, 10, 15, 20]
LINEWIDTH = 1.4


def compute_iss_cv(path):
    df = pd.read_csv(path)[["tree_size", "objective"]]
    results = []

    for tree_size, group in df.groupby("tree_size"):
        valid = group["objective"].dropna()
        if len(valid) >= 19:
            avg = valid.mean()
            cv = valid.std(ddof=1) / avg if avg else None
            results.append({"tree_size": tree_size, "cv": cv})

    return pd.DataFrame(results)


def compute_oss_cv(path):
    df = pd.read_csv(path)[["tree_size", "count", "objective"]]
    df = df[df["objective"].notna() & (df["objective"] > 0)]
    results = []

    for tree_size, group in df.groupby("tree_size"):
        total_count = group["count"].sum()
        if total_count < 19:
            continue

        avg = (group["objective"] * group["count"]).sum() / total_count
        variance = (((group["objective"] - avg) ** 2) * group["count"]).sum() / total_count
        cv = variance**0.5 / avg if avg else None
        results.append({"tree_size": tree_size, "cv": cv})

    return pd.DataFrame(results)


def plot_stability_cv():
    output_dir = Path(PLOT_DIR) / "ISS_OSS"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(FIGWIDTH / 2.54, 9 / 2.54))

    case_data = {}
    y_max = 0
    for case in CASES:
        df_iss = compute_iss_cv(f"results/stability/{case}/con_mp/ISS.csv")
        df_oss = compute_oss_cv(f"results/stability/{case}/con_mp/OSS.csv")
        case_data[case] = (df_iss, df_oss)
        y_max = max(y_max, df_iss["cv"].max(), df_oss["cv"].max())

    for ax, case in zip(axs.flat, CASES):
        df_iss, df_oss = case_data[case]

        ax.plot(
            df_iss["tree_size"],
            df_iss["cv"],
            color=colors.consensus_iss,
            marker="o",
            label="ISS",
            linewidth=LINEWIDTH,
        )

        ax.plot(
            df_oss["tree_size"],
            df_oss["cv"],
            color=colors.consensus_oss,
            marker="s",
            label="OSS",
            linewidth=LINEWIDTH,
        )

        ax.set_title(case)
        ax.set_xticks(TREE_SIZES)
        ax.set_ylim(0, 0.20)
        ax.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(color="0.90", linewidth=0.5)

    fig.supxlabel("Tree size", y=0.04)
    fig.supylabel("Coefficient of variation", x=0.04)
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
    fig.savefig(output_dir / "stability_cv.svg")
    fig.savefig(output_dir / "stability_cv_con_mp.svg")
    plt.close(fig)


if __name__ == "__main__":
    plot_stability_cv()
