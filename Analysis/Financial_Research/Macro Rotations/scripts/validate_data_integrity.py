
import pandas as pd
import datetime as dt
from core.config import CONFIG
from data.loader import download_data

def validate_data():
    print("--- DATA INTEGRITY CHECK ---\n")
    
    # Force fresh download to be sure
    # We can't easily rm cache from here without os, but download_data uses cache by default.
    # We assume the current cache is what we want to test (or we can pass use_cache=False).
    # Let's test the CACHED data first as that's what the report uses.
    
    print("Loading Data...")
    prices, macro = download_data(CONFIG, use_cache=True)
    
    today = pd.Timestamp.today()
    one_week_ago = today - pd.Timedelta(days=7)
    
    print(f"Analysis Date: {today.date()}\n")
    
    # 1. PRICE DATA VALIDATION
    print(">>> CHECKING ASSET PRICES (Yahoo Finance)")
    print(f"Total Tickers Configured: {len(CONFIG['tickers'])}")
    print(f"Columns Downloaded: {len(prices.columns)}")
    
    missing_tickers = set(CONFIG['tickers']) - set(prices.columns)
    if missing_tickers:
        print(f"!!! MISSING TICKERS: {missing_tickers}")
    else:
        print("All tickers present.")
        
    # Check Freshness
    last_date = prices.index[-1]
    print(f"Latest Price Date: {last_date.date()}")
    
    if last_date < one_week_ago:
        print("!!! CRITICAL: Price data is STALE (> 7 days old)!")
    else:
        print("Freshness: OK")
        
    # Check for excessive NaNs in recent history (last 5 rows)
    recent = prices.iloc[-5:]
    nan_counts = recent.isna().sum()
    problematic = nan_counts[nan_counts > 0]
    if not problematic.empty:
        print(f"!!! Warning: Recent NaNs found in: {problematic.index.tolist()}")
    else:
        print("Recent Data Quality: OK (No NaNs in last 5 days)")
        
    print("-" * 30)

    # 2. MACRO DATA VALIDATION
    print(">>> CHECKING MACRO PLUMBING (FRED)")
    print(f"Total Series Configured: {len(CONFIG['fred_codes'])}")
    # Note: macro columns are renamed, so we check against values of fred_codes
    expected_macro = set(CONFIG['fred_codes'].values())
    
    # Add calculated columns to expected list to avoid confusion, or just check base keys
    present_macro = set(macro.columns)
    
    missing_macro = []
    for code, name in CONFIG['fred_codes'].items():
        if name not in present_macro:
            # Maybe it's one of the ones we renamed? values() covers it.
            missing_macro.append(name)
            
    if missing_macro:
        print(f"!!! MISSING MACRO SERIES: {missing_macro}")
    else:
        print("All macro series present.")
        
    # Check Freshness (Macro updates vary, so we check last valid index)
    last_macro_date = macro.index[-1]
    print(f"Latest Macro Date (Reindexed): {last_macro_date.date()}")
    
    # Sanity Checks on Key Indicators
    print("\n>>> SANITY CHECKS (Latest Values)")
    
    checks = {
        "10Y_Yield": (1.0, 6.0, "%"), # Expect between 1% and 6%
        "VIX": (10.0, 80.0, "pts"),   # Expect between 10 and 80
        "CPI_YoY": (0.00, 0.10, "% (decimal)"), # Expect 0% to 10%
        "M2_YoY": (-0.05, 0.20, "% (decimal)"), # Expect -5% to 20%
        "Unemployment": (3.0, 10.0, "%"), # Expect 3% to 10% (FRED is %, usually)
    }
    
    for col, (min_v, max_v, unit) in checks.items():
        if col in macro.columns:
            val = macro[col].dropna().iloc[-1]
            status = "OK" if min_v <= val <= max_v else "SUSPICIOUS"
            print(f"   {col}: {val:.4f} {unit} [{status}]")
        else:
            print(f"   {col}: MISSING")

    # SPY check
    if "SPY" in prices.columns:
        spy_val = prices["SPY"].iloc[-1]
        print(f"   SPY Price: {spy_val:.2f} [OK if > 400]")
        
    print("\nValidation Complete.")

if __name__ == "__main__":
    validate_data()
