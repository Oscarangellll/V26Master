
import matplotlib.pyplot as plt
import numpy as np

from data.fixed_data import data


def plot_power_curve(): 
    x = np.linspace(0, 35, 200)

    y = data.power_curve(x)
    
    fig, ax = plt.subplots(figsize=(15/2.54,2))
    
    ax.plot(x, y)

    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Power [MW]")
    ax.margins(y=0.1)

    fig.savefig("figures/plots/power_curve.svg")

