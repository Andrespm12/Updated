"""
Sector Rotation and Seasonality Analytics.
"""
import pandas as pd
import datetime as dt
import calendar
from typing import Dict

def calculate_rrg_metrics(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Relative Strength (vs SPY) and Momentum for Sectors.
    Returns DataFrame with [Ticker, RS, Momentum, Quadrant]
    """
    sectors = ["XLI", "XLB", "XLU", "XLF", "XLK", "XLE", "XLV", "XLC", "XLRE", "XLY", "XLP"]
    
    if "SPY" not in prices.columns:
        return pd.DataFrame()
        
    rrg_data = []
    spy = prices["SPY"]
    
    for sec in sectors:
        if sec in prices.columns:
            p = prices[sec]
            
            # Relative Strength (Ratio)
            rs_raw = p / spy
            
            # Normalize RS: (Current / 200D MA) - 1
            # This centers it around 0. >0 means trading above trend vs SPY.
            rs_ma = rs_raw.rolling(200).mean()
            if rs_ma.iloc[-1] == 0: continue # Avoid div by zero
            rs_norm = (rs_raw.iloc[-1] / rs_ma.iloc[-1]) - 1
            
            # Momentum of RS: Rate of Change of the Ratio (e.g., 20 Days)
            # Are we gaining or losing ground vs SPY fast?
            rs_mom = rs_raw.pct_change(20).iloc[-1]
            
            # Quadrant
            if rs_norm > 0 and rs_mom > 0: quad = "Leading"
            elif rs_norm > 0 and rs_mom < 0: quad = "Weakening"
            elif rs_norm < 0 and rs_mom < 0: quad = "Lagging"
            else: quad = "Improving"
            
            rrg_data.append({
                "Ticker": sec,
                "RS": rs_norm,
                "Momentum": rs_mom,
                "Quadrant": quad
            })
            
    return pd.DataFrame(rrg_data)

def predict_sector_rotation(prices: pd.DataFrame) -> Dict:
    """
    Predicts next likely Sector Leader using a Transition Matrix.
    """
    print("   Calculating Sector Rotation Probabilities...")
    sectors = ["XLI", "XLB", "XLU", "XLF", "XLK", "XLE", "XLV", "XLC", "XLY", "XLP"]
    res = {"current_leader": "N/A", "next_likely": "N/A", "prob": 0.0}
    
    # Check data
    valid_sectors = [s for s in sectors if s in prices.columns]
    if not valid_sectors: return res
    
    # Monthly Returns
    monthly_prices = prices[valid_sectors].resample('ME').last()
    if monthly_prices.empty: return res
    monthly_rets = monthly_prices.pct_change().dropna()
    
    if monthly_rets.empty: return res
    
    # Identify Leader each month
    leaders = monthly_rets.idxmax(axis=1)
    
    # Build Transition Matrix
    # Count transitions: Leader(t) -> Leader(t+1)
    transitions = {}
    
    for prev, curr in zip(leaders[:-1], leaders[1:]):
        if prev not in transitions: transitions[prev] = {}
        if curr not in transitions[prev]: transitions[prev][curr] = 0
        transitions[prev][curr] += 1
        
    # Current Leader Logic (Same as original)
    last_date = monthly_rets.index[-1]
    today = pd.Timestamp.today()
    
    # If last_date month == today month, it's the current partial month.
    if last_date.month == today.month and last_date.year == today.year:
        if len(leaders) > 1:
            curr_leader = leaders.iloc[-2]
        else:
            curr_leader = leaders.iloc[-1]
    else:
        curr_leader = leaders.iloc[-1]
    
    # Predict Next
    if curr_leader in transitions:
        counts = transitions[curr_leader]
        total = sum(counts.values())
        
        # Find max
        best_next = max(counts, key=counts.get)
        prob = counts[best_next] / total
        
        res = {
            "current_leader": curr_leader,
            "next_likely": best_next,
            "prob": prob
        }
    else:
        res = {"current_leader": curr_leader, "next_likely": "Unknown (New Regime)", "prob": 0.0}
        
    return res

def calculate_seasonality(prices: pd.DataFrame) -> Dict:
    """
    Calculates Seasonality Stats for Current and Next Month (SPY).
    """
    print("   Calculating Seasonality...")
    res = {"curr_month": "N/A", "curr_stats": "", "next_month": "N/A", "next_stats": ""}
    
    if "SPY" not in prices.columns: return res
    
    spy = prices["SPY"]
    rets = spy.pct_change().dropna()
    
    # Create DF with Month
    df_seas = pd.DataFrame({"Ret": rets})
    df_seas["Month"] = df_seas.index.month
    
    # Current Month
    today = dt.date.today()
    curr_m = today.month
    next_m = (curr_m % 12) + 1
    
    curr_name = calendar.month_name[curr_m]
    next_name = calendar.month_name[next_m]
    
    def get_stats(m_idx):
        # We want monthly returns, not daily avg
        try:
             spy_monthly = spy.resample('ME').last().pct_change().dropna()
        except:
             spy_monthly = spy.resample('M').last().pct_change().dropna()

        df_m = pd.DataFrame({"Ret": spy_monthly})
        df_m["Month"] = df_m.index.month
        
        subset = df_m[df_m["Month"] == m_idx]["Ret"]
        
        if subset.empty: return "N/A"
        
        count = len(subset)
        win_rate = (subset > 0).mean()
        avg_ret = subset.mean()
        
        return f"Win Rate: {win_rate:.0%} | Avg: {avg_ret:+.1%} (n={count} yrs)"
        
    res = {
        "curr_month": curr_name,
        "curr_stats": get_stats(curr_m),
        "next_month": next_name,
        "next_stats": get_stats(next_m)
    }
    return res
