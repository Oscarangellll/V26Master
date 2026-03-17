import pandas as pd

from data.fixed_data import data

df = pd.read_csv("data/price/original.csv", parse_dates=["Date"])

isos = {w.iso for w in data.wind_farms}

df = df[
    df["Date"].dt.year.between(data.price_from_year, data.price_to_year) &
    df["ISO3 Code"].isin(isos)
].drop(columns=["Country"])

df = df.rename(
    columns={
        "Date": "date", 
        "ISO3 Code": "ISO3", 
        "Price (EUR/MWhe)": "price"
    }
)

df.to_csv("data/price/price.csv", index=False)
