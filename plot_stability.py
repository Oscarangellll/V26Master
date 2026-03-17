import pandas as pd
import matplotlib.pyplot as plt

# Read the results
iss = pd.read_csv("results/stability/1W1B/mip/ISS.csv")
oss = pd.read_csv("results/stability/1W1B/mip/OSS.csv")

# Compute average objective per tree size
iss_avg = iss.groupby("tree_size")["objective"].mean().reset_index()
oss_avg = (
    oss.assign(weighted_obj=lambda df: df["objective"] * df["count"])
       .groupby("tree_size", as_index=False)
       .apply(lambda g: g["weighted_obj"].sum() / g["count"].sum())
       .rename(columns={None: "objective"})
)

# Create scatter plot
plt.figure(figsize=(8,5))
plt.scatter(iss_avg["tree_size"], iss_avg["objective"], label="In-sample", color="blue", alpha=0.6)
plt.scatter(oss_avg["tree_size"], oss_avg["objective"], label="Out-of-sample", color="red", alpha=0.6)

# Optional: connect points with lines
plt.plot(iss_avg["tree_size"], iss_avg["objective"], color="blue", linestyle="--", alpha=0.5)
plt.plot(oss_avg["tree_size"], oss_avg["objective"], color="red", linestyle="--", alpha=0.5)

plt.xlabel("Tree size")
plt.ylabel("Average objective value")
plt.title("In-sample vs Out-of-sample objectives")
plt.legend()
plt.grid(True)
plt.show()
