import os
from pathlib import Path
import pandas as pd

from data.fixed_data import data

def process_price_data():

    data_dir = Path(os.environ.get("DATA_DIR", "data"))

    df = pd.read_csv(data_dir / "price/original.csv", parse_dates=["Date"])
    
    isos = {w.iso for w in data.wind_farms}
    
    df = df[
        df["Date"].dt.year.between(data.price_from_year, data.price_to_year) &
        df["ISO3 Code"].isin(isos)
    ].drop(columns=["Country"])

    df = df.rename(
        columns={
            "ISO3 Code": "ISO3", 
            "Price (EUR/MWhe)": "price"
        }
    )
    df = df.set_index("Date")
    df.index.name = "date"

    df.to_parquet(data_dir / "price/price.parquet")
    #df.to_csv("data/price/price.csv")
