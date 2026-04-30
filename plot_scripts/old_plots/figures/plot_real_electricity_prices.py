import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.fixed_data import data

def plot_real_electricity_prices():
    isos = {wind_farm.iso for wind_farm in data.wind_farms}
    
    df = pd.read_csv("data/price/original.csv")
    df = df[df["ISO3 Code"].isin(isos)]
    
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date") 

    df = df[df.index.year >= 2019]
    
    fig, ax = plt.subplots(figsize=(15/2.54,2.5))

    for iso3, group in df.groupby("ISO3 Code"):
        ax.plot(group.index, group["Price (EUR/MWhe)"], label=iso3) 

    ax.legend(loc="upper left")

    fig.savefig("figures/plots/real_electricity_prices.svg")
    plt.show()
