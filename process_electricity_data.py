import pandas as pd

from data import FixedData

df = pd.read_csv("data/electricity/original.csv", parse_dates=["Date"])

iso_codes = {w.ISO3 for w in FixedData().wind_farms}

df = df[
    df["Date"].dt.year.isin([2023, 2024, 2025]) &
    df["ISO3 Code"].isin(iso_codes)
].drop(columns=["Country"])

df = df.rename(
    columns={
        "Date": "date", 
        "ISO3 Code": "ISO3", 
        "Price (EUR/MWhe)": "price"
    }
)

df.to_csv("data/electricity/processed.csv", index=False)
