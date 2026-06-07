import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.fixed_data import data
from pathlib import Path

from plot_scripts.config import PLOT_DIR, colors, FIGWIDTH

def make_gdf(items):
    df = pd.DataFrame(
        [(item.name, item.lat, item.lon) for item in items],
        columns=["label", "lat", "lon"]
    )

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )

gdf_wf = make_gdf(data.wind_farms)
gdf_bases = make_gdf(data.bases)

world = gpd.read_file(
    "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"
)
def get_bounds(gdf_wf, gdf_bases, pad_x=0.08, pad_y=0.08):
    combined = pd.concat([gdf_wf, gdf_bases])

    minx, miny, maxx, maxy = combined.total_bounds

    dx = maxx - minx
    dy = maxy - miny

    minx -= pad_x * dx
    maxx += pad_x * dx

    miny -= pad_y * dy
    maxy += pad_y * dy

    return minx, miny, maxx, maxy

def plot_map():
    minx, miny, maxx, maxy = get_bounds(
        gdf_wf,
        gdf_bases
    )
    map_width = maxx - minx
    map_height = maxy - miny

    mean_lat = (miny + maxy) / 2

    fig_width = FIGWIDTH / 2.54

    fig_height = fig_width * map_height / (
        map_width * np.cos(np.deg2rad(mean_lat))
    )

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height)
    )

    ax.set_facecolor("#dceef7")
    world.plot(
        ax=ax,
        color="#d9ead3",
        edgecolor="#666666",
        linewidth=0.6,
        #rasterized=True
    )

    gdf_wf.plot(
        ax=ax,
        marker="o",
        color=colors.blue,
        markersize=35,
        label="Wind farms"
    )

    gdf_bases.plot(
        ax=ax,
        marker="^",
        color=colors.red,
        markersize=55,
        label="Bases"
    )
    
    # Zoom to North Sea region
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    ax.legend()
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Wind farms and bases")

    output_dir = Path(PLOT_DIR) / "weather_validation_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "map.svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
