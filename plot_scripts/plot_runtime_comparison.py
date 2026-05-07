
import matplotlib.pyplot as plt
import pandas as pd

from plot_scripts.config import PLOT_DIR, colors, FIGWIDTH

MIP_FILES = {
    "1W1B mip": "results/stability/1W1B/mip/ISS.csv",
    "2W2B mip": "results/stability/2W2B/mip/ISS.csv",
    "3W2B mip": "results/stability/3W2B/mip/ISS.csv",
    "4W3B mip": "results/stability/4W3B/mip/ISS.csv",
}

CON_FILES = {
    "1W1B con": "results/stability/1W1B/con_mp/ISS.csv",
    "2W2B con": "results/stability/2W2B/con_mp/ISS.csv",
    "3W2B con": "results/stability/3W2B/con_mp/ISS.csv",
    "4W3B con": "results/stability/4W3B/con_mp/ISS.csv",
}

def compute_runtime_avg(path, method):
    if method == "mip":
        colname = "MIP_runtime" 
    if method == "con":
        colname = "Con_total runtime"
    df = pd.read_csv(path)[["tree_size", colname]]
    
    results = []

    for tree_size, group in df.groupby("tree_size"):

        if len(group) == 20:
            avg = group[colname].mean()

            results.append({
                "tree_size": tree_size,
                "avg": avg
            })

    return pd.DataFrame(results)


def plot_runtime_comparison():

    fig, ax = plt.subplots(figsize=(FIGWIDTH/2.54, 3))

    for name, path in MIP_FILES.items():
        df = compute_runtime_avg(path, "mip")
        ax.plot(df["tree_size"], df["avg"], label=name, color="blue")

    for name, path in CON_FILES.items():
        df = compute_runtime_avg(path, "con")
        ax.plot(df["tree_size"], df["avg"], label=name, color="red")

    ax.set_xlabel("Tree size")
    ax.set_ylabel("Runtime [s]")

    ax.set_xticks([1, 3, 5, 7, 10, 15, 20])

    ax.legend()
    
    fig.savefig(PLOT_DIR + "runtime_comparison")
    plt.show()
