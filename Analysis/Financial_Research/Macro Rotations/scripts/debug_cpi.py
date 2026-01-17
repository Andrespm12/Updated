
import pandas as pd
from core.config import CONFIG
from data.loader import download_data
from analytics.macro_models import build_analytics
import sys

# Override config to ensure we get printouts
def debug_cpi():
    print("--- DEBUGGING CPI DATA ---")
    prices, macro = download_data(CONFIG, use_cache=True)
    
    if "CPI" not in macro.columns:
        print("ERROR: 'CPI' column not found in macro data!")
        print("Columns available:", macro.columns)
        return

    cpi_raw = macro["CPI"].dropna()
    print(f"\nRaw CPI Data (Last 5 days):")
    print(cpi_raw.tail())
    
    # Check frequency
    print(f"\nCPI Index Frequency inferred: {pd.infer_freq(cpi_raw.index)}")
    
    # Check for pre-calculated column
    if "CPI_YoY" in macro.columns:
        print("\nPre-calculated 'CPI_YoY' found in macro dataframe.")
        cpi_yoy = macro["CPI_YoY"]
    else:
        print("\nWARNING: 'CPI_YoY' not found. Calculating manually (252 lag)...")
        cpi_yoy = macro["CPI"].pct_change(252)
    
    print(f"\nCPI YoY (Last 5 days):")
    print(cpi_yoy.tail() * 100) # Show as percentage
    
    latest_val = cpi_yoy.iloc[-1]
    print(f"\nLATEST CPI YoY VALUE: {latest_val:.2%}")
    
    if latest_val < 0.02:
        print("CONFIRMED: Value is BELOW 2%.")
    else:
        print("Value is ABOVE 2%.")

if __name__ == "__main__":
    debug_cpi()
