import os
import zipfile

import pandas as pd
import numpy as np
import cdsapi

from data.fixed_data import data
from data.hashing import *

dataset = "reanalysis-era5-single-levels-timeseries"
client = cdsapi.Client()

from_year = data.weather_from_year
to_year = data.weather_to_year

complete_time_index = pd.date_range(
    start=f"{from_year}-01-01 00:00:00",
    end=f"{to_year}-12-31 23:00:00",
    freq="h"
)

df_full = []

for wl in data.weather_locations:
    
    filehash = hash_weather_location(wl, from_year, to_year)
    filepath = f"data/weather/{filehash}.zip"

    if not os.path.exists(filepath):
        request = {
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "significant_height_of_combined_wind_waves_and_swell"
            ],
            "location": {"longitude": wl.lon, "latitude": wl.lat},
            "date": [f"{from_year}-01-01/{to_year}-12-31"],
            "data_format": "csv"
        }

        results = client.retrieve(dataset, request)
        results.download(target=filepath)
    else:
        print(f"{filepath} already exists.")
    
    with zipfile.ZipFile(filepath, "r") as z:
        csv_files = z.namelist()
        
        if len(csv_files) == 1:
            df = pd.read_csv(
                z.open(csv_files[0]), 
                index_col="valid_time", 
                parse_dates=True
            ).drop(columns=["latitude", "longitude"]) 
        else:
            dfs = [
                pd.read_csv(
                    z.open(csv_file), 
                    index_col="valid_time", 
                    parse_dates=True
                ).drop(columns=["latitude", "longitude"]) 
                for csv_file in csv_files
            ]

            df = pd.concat(dfs, axis=1, join="inner")
    
        df["speed"] = np.sqrt(df["u10"]**2 + df["v10"]**2)
    
        df = df.rename(columns={"swh": "height"})
        df = df[["speed", "height"]]

        df = df.reindex(complete_time_index)
        
        assert not df.isna().any().any(), (
            f"Data for location {wl.id} contains missing values."
        )
        assert not df.le(0).any().any(), (
            f"Data for location {wl.id} contains non-positive values."
        )
        
        df.index.name = "time"
        df["weather_location_id"] = wl.id 
        df_full.append(df)

df_full = pd.concat(df_full)

df_full_filehash = hash_all_weather_locations(data.weather_locations, from_year, to_year)
df_full_filepath = f"data/weather/{df_full_filehash}.csv"

df_full.to_csv(df_full_filepath, float_format="%.4f")




