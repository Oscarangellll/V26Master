import os
import zipfile

import pandas as pd
import numpy as np
import cdsapi

from data import FixedData

dataset = "reanalysis-era5-single-levels-timeseries"
client = cdsapi.Client()

weather_locations = [l for l in FixedData().weather_locations]

from_year, to_year = 2015, 2025

complete_time_index = pd.date_range(
    start=f"{from_year}-01-01 00:00:00",
    end=f"{to_year}-12-31 23:00:00",
    freq="h"
)

df_full = []

for l in weather_locations:
    
    filename = f"{l.weather_location_id}_{from_year}_{to_year}.zip"
    filepath = f"data/weather/{filename}"

    if not os.path.exists(filepath):
        request = {
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "significant_height_of_combined_wind_waves_and_swell"
            ],
            "location": {"longitude": l.lon, "latitude": l.lat},
            "date": [f"{from_year}-01-01/{to_year}-12-31"],
            "data_format": "csv"
        }

        results = client.retrieve(dataset, request)
        results.download(target=filepath)
    else:
        print(f"{filename} already exists.")
    
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
        print(df)
        assert not df.isna().any().any(), (
            f"Data for weather_location {l.weather_location_id} contains missing values."
        )
        assert not df.le(0).any().any(), (
            f"Data for weather_location {l.weather_location_id} contains non-positive values."
        )
        
        df.index.name = "time"
        df["weather_location_id"] = l.weather_location_id 
        df_full.append(df)

df_full = pd.concat(df_full)
df_full.to_csv(f"data/weather/{from_year}_{to_year}.csv", float_format="%.4f")
