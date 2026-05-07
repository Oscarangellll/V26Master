
import matplotlib.pyplot as plt
import pandas as pd

from plot_scripts.config import PLOT_DIR, colors, FIGWIDTH

def iss(path):
    df = pd.read_csv(path)[["tree_size", "objective"]]
    
    results = []

    for tree_size, group in df.groupby("tree_size"):

        if len(group) == 20:
            avg = group["objective"].mean()

            results.append({
                "tree_size": tree_size,
                "avg": avg
            })

    return pd.DataFrame(results)

def oss(path): 
    df = pd.read_csv(path)[["tree_size", "count", "objective"]]

    results = []

    for tree_size, group in df.groupby("tree_size"):
        total_count = group["count"].sum()

        if total_count == 20:
            weighted_avg = (group["objective"] * group["count"]).sum() / total_count

            results.append({
                "tree_size": tree_size,
                "avg": weighted_avg
            })

    return pd.DataFrame(results)

def plot_stability():

    fig, axs = plt.subplots(2, 2, figsize=(FIGWIDTH / 2.54, 3))

    cases = ["1W1B", "2W2B", "3W2B", "4W3B"]

    for ax, case in zip(axs.flat, cases):

        df_iss = iss(f"results/stability/{case}/con_mp/ISS.csv")
        df_oss = oss(f"results/stability/{case}/con_mp/OSS.csv")

        ax.plot(
            df_iss["tree_size"],
            df_iss["avg"] / 1e6,
            color="red",
            marker="s",
        )

        ax.plot(
            df_oss["tree_size"],
            df_oss["avg"] / 1e6,
            color="red",
            marker="s",
        )

        ax.set_title(case)
        ax.set_xticks([1, 3, 5, 7, 10, 15, 20])


    fig.savefig(PLOT_DIR + "stability")
    plt.show()
