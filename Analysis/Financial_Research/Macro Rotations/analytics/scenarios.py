"""
Scenario Analysis & Contingency Planning Module.
Calculates probabilities of market moves and generates tactical contingencies.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from analytics.quant import simulate_gbm

def calculate_move_probabilities(series: pd.Series, days: int = 60, n_sims: int = 2000) -> pd.DataFrame:
    """
    Calculates probability of hitting key % targets (+/- 5%, 10%, etc.)
    using Monte Carlo simulations calibrated to recent volatility.
    
    Returns a DataFrame with Upside and Downside probabilities.
    """
    # 1. Calibrate to recent history (vol regime)
    rets = series.pct_change().dropna()
    
    # Use exponential weighting for vol to capture current regime? 
    # Or just calc standard vol. Let's use standard for robustness, maybe shorter window.
    # Using last 6 months (126 days) for calibration
    recent_rets = rets.iloc[-126:] 
    
    mu = recent_rets.mean() * 252
    sigma = recent_rets.std() * np.sqrt(252)
    S0 = series.iloc[-1]
    
    # 2. Simulate Paths
    T = days / 252.0
    dt_step = 1 / 252.0
    
    # Using GBM for standard probabilities
    _, paths = simulate_gbm(S0, mu, sigma, T, dt_step, n_sims)
    
    # 3. Analyze Final Prices
    final_prices = paths[:, -1]
    final_rets = (final_prices / S0) - 1
    
    # 4. Define Targets
    targets = [0.02, 0.05, 0.10, 0.15] # 2%, 5%, 10%, 15%
    
    results = {}
    
    for t in targets:
        # Upside: Prob > +t
        prob_up = (final_rets > t).mean()
        # Downside: Prob < -t
        prob_down = (final_rets < -t).mean()
        
        results[f"{t*100:.0f}%"] = {
            "Upside Prob": prob_up,
            "Downside Risk": prob_down
        }
        
    df_probs = pd.DataFrame(results).T
    df_probs["Ratio (Up/Down)"] = df_probs["Upside Prob"] / (df_probs["Downside Risk"] + 1e-6)
    
    return df_probs

def generate_contingencies(current_price: float, regime: str, vol_score: float, skew: float) -> pd.DataFrame:
    """
    Generates a 'Contingency Playbook' based on market status.
    Returns a DataFrame of scenarios and actions.
    """
    scenarios = []
    
    # Scenario 1: Crash / High Vol
    if vol_score > 0.8 or "High Vol" in regime:
         scenarios.append({
             "Scenario": "CRITICAL: High Volatility Regime",
             "Condition": "VIX Spike / Crash Mode",
             "Action": "Target Delta Neutral. Buy 30-60 DTE Puts (~1 Put per 100 shares).",
             "Key Level": f"< {current_price * 0.90:.2f} (-10%)"
         })
    elif vol_score < 0.2:
        scenarios.append({
             "Scenario": "Complacency / Low Vol",
             "Condition": "Low VIX / Grinding Higher",
             "Action": "Long Call Spreads. Watch for localized vol spikes.",
             "Key Level": f"> {current_price * 1.05:.2f} (+5%)"
        })
    else:
        scenarios.append({
             "Scenario": "Normal Volatility",
             "Condition": "Standard Oscillations",
             "Action": "Sell Mean Reversion. Market Neutral.",
             "Key Level": "Range Bound"
        })
        
    # Scenario 2: Skew / Tail Risk
    if skew < -1.0:
        scenarios.append({
            "Scenario": "Heavy Downside Skew (Fear)",
            "Condition": "Puts expensive vs Calls",
            "Action": "Sell Puts to finance Calls (Risk Reversal).",
            "Key Level": "N/A"
        })
    elif skew > 0.5:
        scenarios.append({
            "Scenario": "Positive Skew (Euphoria?)",
            "Condition": "Calls expensive vs Puts",
            "Action": "Take Profits. Buy Cheap Protection.",
            "Key Level": "N/A"
        })
        
    return pd.DataFrame(scenarios)
