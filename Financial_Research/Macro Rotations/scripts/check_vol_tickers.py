import yfinance as yf

tickers = [
    "^VIX",   # SP500 Vol
    "^VXN",   # Nasdaq Vol
    "^RVX",   # Russell 2000 Vol
    "^GVZ",   # Gold Vol
    "^VXTLT", # TLT Vol (Check availability)
    "^EVZ",   # Euro Vol
    "^JYV",   # Yen Vol
    "TYV"     # Treasury Yield Vol (CBOE) - alternative check
]

print("Checking Volatility Tickers...")
data = yf.download(tickers, period="5d", progress=False)["Close"]

print("\nDownloaded Data Columns:")
print(data.columns.tolist())

for t in tickers:
    if t in data.columns:
        last_price = data[t].iloc[-1]
        print(f"{t}: Available (Last: {last_price:.2f})")
    else:
        print(f"{t}: NOT FOUND or No Data")
