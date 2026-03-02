import pandas as pd

from data.fixed_data import data

df = pd.read_csv("data/electricity/original.csv", parse_dates=["Date"])

df = df[
    df["Date"].dt.year.between(data.price_from_year, data.price_to_year) &
    df["ISO3 Code"].isin(data.iso_codes)
].drop(columns=["Country"])

df = df.rename(
    columns={
        "Date": "date", 
        "ISO3 Code": "ISO3", 
        "Price (EUR/MWhe)": "price"
    }
)

data_hash = data.price_data_hash()
data_path = f"data/electricity/{data_hash}.csv"
df.to_csv(data_path, index=False)
