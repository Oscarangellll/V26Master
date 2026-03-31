
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from data.fixed_data import data

fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())

ax.coastlines()
ax.add_feature(cfeature.BORDERS)

for w in data.wind_farms:
    ax.scatter(w.lon, w.lat, marker='s', color='blue', s=40, transform=ccrs.PlateCarree())
    ax.text(w.lon + 0.1, w.lat + 0.1, w.name, fontsize=8, transform=ccrs.PlateCarree())

for b in data.bases:
    ax.scatter(b.lon, b.lat, marker='o', color='red', s=40, transform=ccrs.PlateCarree())
    ax.text(b.lon + 0.1, b.lat + 0.1, b.name, fontsize=8, transform=ccrs.PlateCarree())

plt.show()
