"""
Data downloading and processing logic.
Includes caching to avoid redundant API calls.
"""
import datetime as dt
import os
import yfinance as yf
import pandas as pd
from typing import Dict, Tuple, Optional

try:
    from pandas_datareader import data as pdr
except ImportError:
    raise ImportError("Please install pandas_datareader: pip install pandas_datareader")

CACHE_DIR = "data/cache"

def download_data(config: Dict, use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Downloads Price (YF) and Macro (FRED) data with caching."""
    
    # Setup Cache
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        
    end = config["end_date"] if config["end_date"] else dt.date.today()
    start = config["start_date"]
    current_date_str = dt.date.today().strftime("%Y-%m-%d")
    
    # Cache File Names (Versioned by date to ensure freshness)
    prices_file = f"{CACHE_DIR}/prices_{current_date_str}.pkl"
    macro_file = f"{CACHE_DIR}/macro_{current_date_str}.pkl"
    
    prices = pd.DataFrame()
    macro = pd.DataFrame()
    
    # Try Loading from Cache
    if use_cache and os.path.exists(prices_file) and os.path.exists(macro_file):
        print("   Loading data from local cache...")
        prices = pd.read_pickle(prices_file)
        macro = pd.read_pickle(macro_file)
        
        # Still fetch fundamentals (Snapshot)
        fundamentals = {}
        try:
            spy_ticker = yf.Ticker("SPY")
            info = spy_ticker.info
            fundamentals["SPY_PE"] = info.get("trailingPE", None)
            print(f"   SPY Trailing PE: {fundamentals['SPY_PE']} (Live)")
        except Exception:
            pass
            
        return prices, macro, fundamentals
    
    # --- 1. Fetching Asset Prices (Yahoo Finance) ---
    print("1. Fetching Asset Prices (Yahoo Finance)...")
    tickers = list(set(config["tickers"])) # Dedupe
    
    try:
        # Progress=False to keep stdout clean
        prices = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)["Close"]
        
        # Handle single ticker edge case
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
            
    except Exception as e:
        print(f"YF Download Error: {e}")
        
    # --- 2. Fetching Macro Plumbing (FRED) ---
    print("2. Fetching Macro Plumbing (FRED)...")
    try:
        fred_map = config["fred_codes"]
        # PDR FRED reader
        macro = pdr.DataReader(list(fred_map.keys()), "fred", start, end)
        macro = macro.rename(columns=fred_map)
        
        # --- Pre-Calculate YoY on Native Frequencies (More Accurate) ---
        if "CPI" in macro.columns:
            # CPI is Monthly. YoY = 12 periods.
            # dropna() ensures we operate on the dense series
            cpi_series = macro["CPI"].dropna()
            macro["CPI_YoY"] = cpi_series.pct_change(12)
            
        if "M2_Money" in macro.columns:
            # M2SL is Monthly.
            m2_series = macro["M2_Money"].dropna()
            macro["M2_YoY"] = m2_series.pct_change(12)
            
        if "Fed_Assets" in macro.columns:
            # WALCL is Weekly (Wednesdays). YoY = 52 periods.
            fed_series = macro["Fed_Assets"].dropna()
            macro["Fed_Assets_YoY"] = fed_series.pct_change(52)
            
        if "M2_Velocity" in macro.columns:
             # M2V is Quarterly. YoY = 4 periods.
             vel_series = macro["M2_Velocity"].dropna()
             macro["M2_Velocity_YoY"] = vel_series.pct_change(4)
    except Exception as e:
        print(f"FRED Download Error: {e}")
        macro = pd.DataFrame()

    # --- 3. Post-Processing ---
    # Forward fill to align timelines (macro reporting lags)
    prices = prices.ffill()
    macro = macro.ffill().reindex(prices.index).ffill()
    
    # --- 4. Fetch Fundamentals (Snapshot) ---
    print("4. Fetching Fundamentals (Snapshot)...")
    fundamentals = {}
    try:
        spy_ticker = yf.Ticker("SPY")
        info = spy_ticker.info
        fundamentals["SPY_PE"] = info.get("trailingPE", None)
        print(f"   SPY Trailing PE: {fundamentals['SPY_PE']}")
    except Exception as e:
        print(f"   Fundamentals Error: {e}")
    
    # Save to Cache
    if use_cache:
        print(f"   Saving data to cache ({CACHE_DIR})...")
        prices.to_pickle(prices_file)
        macro.to_pickle(macro_file)
        # Note: We are not caching fundamentals to keep them fresh
    
    return prices, macro, fundamentals
