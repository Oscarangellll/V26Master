
import matplotlib.pyplot as plt
import pandas as pd

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
    "Random": colors.green,
}

def compute_weighted_avg(path):
    df = pd.read_csv(path)[["tree_size", "count", "objective"]]

    results = []

    for tree_size, group in df.groupby("tree_size"):
        total_count = group["count"].sum()

        if total_count == 20:
            weighted_avg = (group["objective"] * group["count"]).sum() / total_count

            results.append({
                "tree_size": tree_size,
                "weighted_avg": weighted_avg
            })

    return pd.DataFrame(results)


def plot_stratified_comparison():

    fig, ax = plt.subplots(figsize=(FIGWIDTH/2.54, 3))

    for name, path in FILES.items():
        df = compute_weighted_avg(path)

        ax.plot(df["tree_size"], df["weighted_avg"] / 1e6, label=name, color=COLOR_MAP[name])

    ax.set_xlabel("Tree size")
    ax.set_ylabel("Weighted objective [MEUR]")

    ax.set_xticks([1, 3, 5, 7, 10, 15])

    ax.legend()

    fig.savefig(PLOT_DIR + "stratified_comparison") 
    plt.show()
