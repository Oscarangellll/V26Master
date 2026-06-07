
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from plot_scripts.config import PLOT_DIR, colors, FIGWIDTH

FILES = {
    "Total ww hours": "results/stratified/mip_total_ww_hours_OSS.csv",
    "Max streak": "results/stratified/mip_max_streak_under_4_OSS.csv",
    "Count under 4": "results/stratified/mip_count_under_4_OSS.csv",
    "Count over 8": "results/stratified/mip_count_over_8_OSS.csv",
    "Random": "results/stratified/mip_random_OSS.csv",
}

COLOR_MAP = {
    "Total ww hours": colors.red,
    "Max streak": colors.orange,
    "Count under 4": colors.blue,
    "Count over 8": colors.purple,
    "Random": colors.direct_mip,
}

def compute_weighted_avg(path):
    df = pd.read_csv(path)[["tree_size", "count", "objective"]]
    df = df[df["objective"].notna() & (df["objective"] > 0)].copy()

    results = []

    for tree_size, group in df.groupby("tree_size"):
        total_count = group["count"].sum()

        if total_count > 0:
            weighted_avg = (group["objective"] * group["count"]).sum() / total_count

            results.append({
                "tree_size": tree_size,
                "weighted_avg": weighted_avg,
                "count": total_count,
            })

    return pd.DataFrame(results).sort_values("tree_size")


def plot_stratified_comparison():
    output_dir = Path(PLOT_DIR) / "ISS_OSS"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(FIGWIDTH / 2.54, 7 / 2.54))

    for name, path in FILES.items():
        df = compute_weighted_avg(path)
        is_random = name == "Random"

        ax.scatter(
            df["tree_size"],
            df["weighted_avg"] / 1e6,
            label=name,
            color=COLOR_MAP[name],
            marker="o",
            s=28 if is_random else 18,
            zorder=5 if is_random else 3,
        )
        for idx in range(len(df) - 1):
            segment = df.iloc[idx : idx + 2]
            linestyle = "-" if (segment["count"] >= 20).all() else "--"
            ax.plot(
                segment["tree_size"],
                segment["weighted_avg"] / 1e6,
                color=COLOR_MAP[name],
                linewidth=2.0 if is_random else 1.2,
                linestyle=linestyle,
                zorder=5 if is_random else 3,
            )

    ax.set_xlabel("Tree size")
    ax.set_ylabel("AOV [MEUR]")

    ax.set_xticks([1, 3, 5, 7, 10, 15, 20])
    ax.grid(color="0.90", linewidth=0.5)

    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.12, right=0.98)

    fig.savefig(output_dir / "stratified_comparison.svg")
    plt.close(fig)


def compute_weighted_cv(path):
    df = pd.read_csv(path)[["tree_size", "count", "objective"]]
    df = df[df["objective"].notna() & (df["objective"] > 0)].copy()

    results = []

    for tree_size, group in df.groupby("tree_size"):
        total_count = group["count"].sum()
        if total_count <= 1:
            continue

        avg = (group["objective"] * group["count"]).sum() / total_count
        variance = (((group["objective"] - avg) ** 2) * group["count"]).sum() / total_count
        cv = variance**0.5 / avg if avg else None

        results.append({
            "tree_size": tree_size,
            "cv": cv,
            "count": total_count,
        })

    return pd.DataFrame(results).sort_values("tree_size")


def plot_stratified_cv():
    output_dir = Path(PLOT_DIR) / "ISS_OSS"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(FIGWIDTH / 2.54, 7 / 2.54))

    for name, path in FILES.items():
        df = compute_weighted_cv(path)
        is_random = name == "Random"

        ax.scatter(
            df["tree_size"],
            df["cv"],
            label=name,
            color=COLOR_MAP[name],
            marker="o",
            s=28 if is_random else 18,
            zorder=5 if is_random else 3,
        )
        for idx in range(len(df) - 1):
            segment = df.iloc[idx : idx + 2]
            linestyle = "-" if (segment["count"] >= 20).all() else "--"
            ax.plot(
                segment["tree_size"],
                segment["cv"],
                color=COLOR_MAP[name],
                linewidth=2.0 if is_random else 1.2,
                linestyle=linestyle,
                zorder=5 if is_random else 3,
            )

    ax.set_xlabel("Tree size")
    ax.set_ylabel("Coefficient of variation")
    ax.set_xticks([1, 3, 5, 7, 10, 15, 20])
    ax.grid(color="0.90", linewidth=0.5)

    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.12, right=0.98)

    fig.savefig(output_dir / "stratified_cv.svg")
    plt.close(fig)


if __name__ == "__main__":
    plot_stratified_comparison()
    plot_stratified_cv()
