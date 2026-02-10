import pandas as pd

locations = pd.read_csv("data/locations.csv", index_col="ISO3")

df = pd.read_csv("data/electricity/original.csv", parse_dates=["Date"])

df = df[
    df["Date"].dt.year.isin([2023, 2024, 2025]) &
    df["ISO3 Code"].isin(locations.index)
]

df["locationID"] = df["ISO3 Code"].map(locations["locationID"])

df = df[["Date", "locationID", "Price (EUR/MWhe)"]].rename(columns={"Date": "date", "Price (EUR/MWhe)": "price"})

df.to_csv("data/electricity/processed.csv", index=False)
