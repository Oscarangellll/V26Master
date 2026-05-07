import matplotlib.pyplot as plt
import pandas as pd

from plot_scripts.config import PLOT_DIR, colors, FIGWIDTH

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


def plot_oss_con_mip():

    fig, axs = plt.subplots(2, 2, figsize=(FIGWIDTH/2.54, 3))

    cases = ["1W1B", "2W2B", "3W2B", "4W3B"]

    for ax, case in zip(axs.flat, cases):

        df_mip = compute_weighted_avg(f"results/stability/{case}/mip/OSS.csv")
        df_con = compute_weighted_avg(f"results/stability/{case}/con_mp/OSS.csv")

        ax.plot(
            df_mip["tree_size"],
            df_mip["weighted_avg"] / 1e6,
            color="blue",
            marker="o",
        )

        ax.plot(
            df_con["tree_size"],
            df_con["weighted_avg"] / 1e6,
            color="red",
            marker="s",
        )

        ax.set_title(case)
        ax.set_xticks([1, 3, 5, 7, 10, 15, 20])

    fig.savefig(PLOT_DIR + "oss_con_mip")
    plt.show()
