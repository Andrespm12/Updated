"""
Institutional Alpha Factors: Net Liquidity, Volatility Structure, and Tail Risk.
"""
import pandas as pd
import numpy as np
from typing import Dict

def calculate_net_liquidity(macro: pd.DataFrame) -> Dict:
    """
    Calculates Net Liquidity = Fed Assets - TGA - RRP.
    Normalizes all units to Trillions of USD.
    """
    print("   Calculating Institutional Net Liquidity...")
    res = {"series": pd.Series(dtype=float), "latest": 0.0, "z_score": 0.0, "status": "N/A"}
    
    # Check availability
    if not all(col in macro.columns for col in ["Fed_Assets", "TGA", "RRP"]):
        print("   -> Missing components (Fed_Assets, TGA, or RRP). Skipping.")
        return res
        
    trends = pd.DataFrame(index=macro.index)
    
    # Normalize to Trillions
    # Fed Assets (WALCL) is usually in Millions. -> / 1,000,000 = Trillions
    # TGA (WTREGEN) is usually in Billions.      -> / 1,000 = Trillions
    # RRP (RRPONTSYD) is usually in Billions.    -> / 1,000 = Trillions
    
    trends["Fed_Assets_T"] = macro["Fed_Assets"] / 1_000_000
    trends["TGA_T"] = macro["TGA"] / 1_000
    trends["RRP_T"] = macro["RRP"] / 1_000
    
    # Net Liquidity Formula (The Howell/Dale Equation)
    trends["Net_Liquidity"] = trends["Fed_Assets_T"] - trends["TGA_T"] - trends["RRP_T"]
    
    # Clean up (ffill handles mismatches in reporting days)
    nl = trends["Net_Liquidity"].ffill().dropna()
    
    if nl.empty: return res
    
    # Calculate Metrics
    latest = nl.iloc[-1]
    
    # Z-Score (6-month lookback for regime change)
    window = 126
    if len(nl) > window:
        mean = nl.rolling(window).mean().iloc[-1]
        std = nl.rolling(window).std().iloc[-1]
        z_score = (latest - mean) / std if std > 0 else 0
    else:
        z_score = 0
        
    if z_score > 1.0: status = "EXPANDING (Bullish)"
    elif z_score < -1.0: status = "CONTRACTING (Bearish)"
    else: status = "NEUTRAL"
    
    res = {
        "series": nl, 
        "latest": latest, 
        "z_score": z_score, 
        "status": status,
        "components": trends[["Fed_Assets_T", "TGA_T", "RRP_T"]].iloc[-1].to_dict()
    }
    return res

def calculate_vol_term_structure(prices: pd.DataFrame) -> Dict:
    """
    Analyzes VIX Term Structure (VIX / VIX3M).
    > 1.0 = Backwardation (Crash Signal).
    """
    print("   Calculating Volatility Term Structure...")
    res = {"ratio": pd.Series(dtype=float), "latest": 0.0, "signal": "N/A"}
    
    if "^VIX" in prices.columns and "^VIX3M" in prices.columns:
        vix = prices["^VIX"]
        vix3m = prices["^VIX3M"]
        
        ratio = (vix / vix3m).dropna()
        latest = ratio.iloc[-1]
        
        if latest > 1.05: signal = "CRASH RISK (Backwardation)"
        elif latest > 1.0: signal = "CAUTION (Inversion)"
        else: signal = "NORMAL (Contango)"
        
        res = {"ratio": ratio, "latest": latest, "signal": signal}
    else:
        print("   -> Missing VIX or VIX3M data.")
        
    return res

def calculate_tail_risk(prices: pd.DataFrame) -> Dict:
    """
    Analyzes CBOE SKEW Index for Tail Risk.
    > 135 = High Demand for Crash Protection (Bearish Divergence).
    < 115 = Complacency.
    """
    print("   Calculating Tail Risk (SKEW)...")
    res = {"series": pd.Series(dtype=float), "latest": 0.0, "signal": "N/A"}
    
    if "^SKEW" in prices.columns:
        skew = prices["^SKEW"].dropna()
        latest = skew.iloc[-1]
        
        if latest > 135: signal = "HIGH RISK (Whales Hedging)"
        elif latest < 115: signal = "COMPLACENCY (Rug Pull Risk)"
        else: signal = "NORMAL"
        
        res = {"series": skew, "latest": latest, "signal": signal}
    else:
        print("   -> Missing ^SKEW data.")
        
    return res
