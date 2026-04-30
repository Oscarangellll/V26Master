import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pandas as pd

# -----------------------------
# Data
# -----------------------------

wind_farms = [
    ("A", 55.30, 7.80),
    ("B", 54.64, 7.94),
    ("C", 54.08, 8.13),
    ("D", 53.93, 7.22),
    ("E", 53.64, 5.03),
    ("F", 52.55, 4.22),
    ("G", 53.30, 1.37),
    ("H", 53.99, 0.48),
]

bases = [
    ("B2", 55.48, 8.34),
    ("B3", 54.68, 8.74),
    ("B4", 53.87, 8.63),
    ("B5", 53.63, 7.14),
    ("B6", 52.88, 4.74),
    ("B7", 52.72, 1.59),
    ("B8", 54.35, -0.47),
]


def make_gdf(data):
    rows = []
    for label, lat, lon in data:
        rows.append({
            "label": label,
            "lat": lat,
            "lon": lon,
            "geometry": Point(lon, lat)
        })
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


gdf_wf = make_gdf(wind_farms)
gdf_bases = make_gdf(bases)
# -----------------------------
# Load map
# -----------------------------

world = gpd.read_file(
    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
)

# -----------------------------
# Plot
# -----------------------------

fig, ax = plt.subplots(figsize=(12, 8))

ax.set_facecolor("#dceef7")

world.plot(
    ax=ax,
    color="#d9ead3",
    edgecolor="#666666",
    linewidth=0.6
)

# Wind farms
gdf_wf.plot(
    ax=ax,
    color="darkred",
    edgecolor="white",
    linewidth=0.8,
    markersize=80,
    marker="s",
    label="Wind farms"
)

# Bases
gdf_bases.plot(
    ax=ax,
    color="navy",
    edgecolor="white",
    linewidth=0.8,
    markersize=140,
    marker="^",
    label="Bases"
)

# Zoom to North Sea region
ax.set_xlim(-3, 11.5)
ax.set_ylim(51.5, 57)

# Styling
ax.set_title("North Sea Wind Farms and Bases", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend(loc="lower left")

plt.tight_layout()

# Save SVG if desired
plt.savefig("north_sea_locations.svg", format="svg", bbox_inches="tight")

plt.show()