import pandas_datareader.data as web
import datetime as dt
import pandas as pd

start = dt.datetime(2020, 1, 1)
end = dt.datetime.now()

fred_codes = {
    "DGS10": "10Y_Yield",
    "DGS2": "2Y_Yield",
    "BAMLH0A0HYM2": "HY_Spread"
}

print("Fetching FRED Data...")
try:
    macro = web.DataReader(list(fred_codes.keys()), "fred", start, end)
    macro = macro.rename(columns=fred_codes)
    print("\nData Head:")
    print(macro.head())
    print("\nData Tail:")
    print(macro.tail())
    print("\nColumns:", macro.columns)
    print("\nMissing Values:\n", macro.isna().sum())
except Exception as e:
    print(f"Error fetching FRED data: {e}")
