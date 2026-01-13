"""
Configuration settings for the Macro Rotations Dashboard.
"""
from typing import Dict

CONFIG: Dict = {
    "start_date": "2018-01-01",  # Good history for post-2020 regimes
    "end_date": None,            # None = Today
    
    # --- ASSET TICKERS (Original + Capital Flows Additions) ---
    "tickers": [
        "SPY", "QQQ", "IWM", "EEM", "VGK", "EWJ", "SCZ", # Equities
        "TLT", "IEF", "SHY", "LQD", "HYG", "BNDX", # Fixed Income
        "GLD", "SLV", "DBC", "USO", # Commodities
        "UUP", "FXE", "FXY", "JPY=X", "DX-Y.NYB", # Currencies (Added JPY=X, DXY)
        "VNQ", "XLRE", # Real Estate
        "XLE", "XLF", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLC", # Sectors
        "MTUM", "USMV", "QUAL", "VLUE", "SIZE", # Factors
        "NOBL", "RSP", "BIL", # Other
        "SMH", "KRE", "XBI", "IGV", "XHB", "XRT", # Sub-sectors
        "URA", "REMX", "COPX", "GDX", # Thematic
        "BTC-USD", # Crypto
        # Restore Missing Tickers for Rotations
        "RPV", "VONG", "SPHB", "SPLV",
        # Labor Basket
        "MAN", "RHI", "KELYA",
        # Global
        "ACWX", "VEA",
        # Commodities
        "CPER",
        # Volatility
        "^MOVE", "^VIX3M", "^VIX", "^VXN", "^GVZ", "^SKEW",
        # Data needed for FX Playbook
        # Data needed for FX Playbook
        "EURUSD=X", "CNY=X", 
        # Missing Foreign Yields (if available in Yahoo/FRED, wait, these are macro usually)
        # We'll just keep the original list for now.
    ],
    
    # --- FRED CODES (The Plumbing) ---
    "fred_codes": {
        "DGS10": "10Y_Yield",       # Nominal 10Y
        "DGS2": "2Y_Yield",         # Nominal 2Y (Fed Policy Proxy)
        "T10YIE": "10Y_Breakeven",  # Inflation Expectations
        "BAMLH0A0HYM2": "HY_Spread",# Credit Risk [cite: 3041]
        "NFCI": "Fin_Conditions",   # Financial Conditions (The Plumbing)
        "VIXCLS": "VIX",            # Fear Premium [cite: 3965]
        
        # Expanded Macro Plumbing
        "M2SL": "M2_Money",         # Liquidity Impulse
        "WALCL": "Fed_Assets",      # QT/QE Monitor
        "WTREGEN": "TGA",           # Treasury General Account (Alpha Factor)
        "RRPONTSYD": "RRP",         # Reverse Repo (Alpha Factor)
        "CPIAUCSL": "CPI",          # Inflation
        "UNRATE": "Unemployment",   # Economic Health
        "DGS3MO": "3M_Yield",       # 3-Month Yield (for Recession Prob)
        "T10Y3M": "Spread_10Y3M",   # 10Y-3M Spread
        "RECPROUSM156N": "Recession_Prob", # Smoothed Recession Prob
        "WSHOMCB": "Fed_Custody",   # Securities Held in Custody for Foreign Accounts (Weekly)
        "IRLTLT01JPM156N": "Japan_10Y", # Long-Term Govt Bond Yield: Japan (Monthly)
        
        # Overnight Rates (Plumbing)
        "SOFR": "SOFR",             # Secured Overnight Financing Rate
        "DFF": "Fed_Funds",         # Federal Funds Effective Rate
        "M2V": "M2_Velocity",       # Velocity of M2 Money Stock
        
        # ADDED FOR GLOBAL MACRO
        "IRLTLT01DEM156N": "Germany_10Y",
        "IRLTLT01GBM156N": "UK_10Y",
        
        # Inflation Swaps / Expectations
        "T5YIE": "5Y_Breakeven",   # 5-Year Breakeven Inflation Rate
        "T5YIFR": "5Y5Y_Forward"   # 5-Year, 5-Year Forward Inflation Expectation Rate
    },
    
    # --- SETTINGS ---
    "ma_short": 50,
    "ma_long": 200,
    "regime_threshold": 1.0, # % threshold for Macro Score
}
