import pandas as pd

from data.fixed_data import data
from data.hashing import *

df = pd.read_csv("data/electricity/original.csv", parse_dates=["Date"])

iso_codes = {w.iso for w in data.wind_farms}
from_year = data.electricity_price_from_year
to_year = data.electricity_price_to_year

df = df[
    df["Date"].dt.year.between(from_year, to_year) &
    df["ISO3 Code"].isin(iso_codes)
].drop(columns=["Country"])

df = df.rename(
    columns={
        "Date": "date", 
        "ISO3 Code": "ISO3", 
        "Price (EUR/MWhe)": "price"
    }
)

filehash = hash_electricity_prices(iso_codes, from_year, to_year)
filepath = f"data/electricity/{filehash}.csv"
df.to_csv(filepath, index=False)
