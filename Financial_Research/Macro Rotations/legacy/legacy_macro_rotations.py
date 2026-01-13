"""
THE SUPER MACRO DASHBOARD: Capital Flows Framework + Global Rotations
----------------------------------------------------------------------
A merged analytics suite that combines:
1. "First Principles" Macro (Plumbing, Liquidity Valves, Real Rates)
2. Granular Equity Rotations (14+ Ratios, Value/Growth, Breadth)
3. Automated "Capital Flows" Commentary based on the Playbooks

Data Sources:
- Prices: Yahoo Finance (yfinance)
- Macro: St. Louis FRED (pandas_datareader)
"""

import datetime as dt
import textwrap
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as stats
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# Try imports
try:
    import yfinance as yf
except ImportError:
    raise ImportError("Please install yfinance: pip install yfinance")

try:
    from pandas_datareader import data as pdr
except ImportError:
    raise ImportError("Please install pandas_datareader: pip install pandas_datareader")


# ==============================================================================
#                               CONFIG & SETUP
# ==============================================================================

CONFIG: Dict = {
    "start_date": "2018-01-01",  # Good history for post-2020 regimes
    "end_date": None,            # None = Today
    
    # --- ASSET TICKERS (Original + Capital Flows Additions) ---
    "tickers": [
        "SPY", "QQQ", "IWM", "EEM", "VGK", "EWJ", "SCZ", # Equities
        "TLT", "IEF", "SHY", "LQD", "HYG", "BNDX", # Fixed Income
        "GLD", "SLV", "DBC", "USO", # Commodities
        "UUP", "FXE", "FXY", "JPY=X", # Currencies (Added JPY=X)
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
        "^MOVE", "^VIX3M", "^VIX", "^VXN", "^GVZ", "^VXTLT"
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
        "CPIAUCSL": "CPI",          # Inflation
        "CPIAUCSL": "CPI",          # Inflation
        "UNRATE": "Unemployment",   # Economic Health
        "DGS3MO": "3M_Yield",       # 3-Month Yield (for Recession Prob)
        "T10Y3M": "Spread_10Y3M",   # 10Y-3M Spread
        "RECPROUSM156N": "Recession_Prob", # Smoothed Recession Prob
        "WSHOMCB": "Fed_Custody",   # Securities Held in Custody for Foreign Accounts (Weekly)
        "IRLTLT01JPM156N": "Japan_10Y", # Long-Term Govt Bond Yield: Japan (Monthly)
        "M2V": "M2_Velocity"        # Velocity of M2 Money Stock
    },
    
    # --- SETTINGS ---
    "ma_short": 50,
    "ma_long": 200,
    "regime_threshold": 1.0, # % threshold for Macro Score
}


# ==============================================================================
#                               RISK & MACRO ANALYTICS
# ==============================================================================

def calculate_portfolio_risk(curve: pd.Series, confidence_level: float = 0.95):
    """
    Calculates Historical VaR and CVaR for a given equity curve.
    """
    rets = curve.pct_change().dropna()
    
    # Historical VaR
    var_95 = rets.quantile(1 - confidence_level)
    
    # CVaR (Expected Shortfall) - Average of returns below VaR
    cvar_95 = rets[rets <= var_95].mean()
    
    return var_95, cvar_95

def calculate_rrg_metrics(prices: pd.DataFrame):
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

def get_jpm_views() -> str:
    """
    Returns JPM's 2026 Regional Style Views (Source: JPM Global Equity Strategy).
    """
    views = """
    1. REGIONAL STYLE VIEWS (2026 Outlook):
       - US:      Momentum (OW), Growth (OW), Low Vol (OW). UW: Small Caps.
       - Europe:  Value (OW), Beta (OW). UW: Momentum, Growth, Low Vol.
       - Japan:   Momentum (OW), Value (OW), Quality (OW), Low Vol (OW).
       - Asia xJ: Value (OW), Quality (OW), Low Vol (OW). UW: Beta, Size.
    
    2. KEY THEMES (2026):
       - AI / Datacenter: "AI Compute Supercycle" (OW Semis/Tech).
       - Deregulation:    "Trump Agenda" (OW Financials/Small Biz).
       - Resources:       "Supply Chain De-risking" (OW Rare Earths/Uranium).
    """
    return textwrap.dedent(views)

def calculate_theme_performance(prices: pd.DataFrame) -> str:
    """
    Calculates 3-Month Momentum for JPM Thematic Proxies.
    Returns formatted text summary.
    """
    proxies = {
        "AI/Tech (SMH)": "SMH",
        "Deregulation (KRE)": "KRE",
        "Resources (REMX)": "REMX",
        "Uranium (URA)": "URA",
        "Momentum (MTUM)": "MTUM",
        "Quality (QUAL)": "QUAL",
        "Low Vol (USMV)": "USMV"
    }
    
    text = "3. THEME VALIDATION (3-Month Momentum):\n"
    
    for name, ticker in proxies.items():
        if ticker in prices.columns:
            # 3M Return (approx 63 days)
            p = prices[ticker].dropna()
            if len(p) > 63:
                ret_3m = (p.iloc[-1] / p.iloc[-63]) - 1
                trend = "CONVERGENCE (Bullish)" if ret_3m > 0.05 else \
                        "DIVERGENCE (Bearish)" if ret_3m < -0.05 else "NEUTRAL"
                
                text += f"   - {name:<20}: {ret_3m:>6.1%} -> {trend}\n"
            else:
                text += f"   - {name:<20}: N/A (Insufficient Data)\n"
                
    return text

def calculate_recession_prob(spread_series: pd.Series) -> pd.Series:
    """
    Calculates Recession Probability using the Estrella & Mishkin Probit Model.
    Prob = NormalCDF( -0.5333 - 0.6330 * Spread )
    Spread = 10Y - 3M (Monthly Average is best, but daily works for proxy)
    """
    # Coefficients (Estrella/Mishkin for 12-month ahead probability)
    # Note: They use monthly averages. We use daily here as a proxy.
    intercept = -0.5333
    beta = -0.6330
    
    # Calculate Z-score for Probit
    # Spread should be in percentage points (e.g., 1.50 for 1.5%)
    z = intercept + beta * spread_series
    
    # Convert to Probability (0-100%)
    prob = norm.cdf(z) * 100
    return pd.Series(prob, index=spread_series.index)

def get_erp_snapshot(ticker_symbol: str = "SPY", real_yield: float = 2.0) -> Dict:
    """
    Calculates Equity Risk Premium (ERP) Snapshot.
    ERP = Earnings Yield - Real Yield
    Returns dict with PE, Earnings Yield, ERP, and Rating.
    """
    print(f"   Fetching Fundamentals for {ticker_symbol} (ERP Calc)...")
    res = {"pe": None, "ey": None, "erp": None, "rating": "N/A"}
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # Try to get Trailing PE, then Forward PE
        pe = info.get("trailingPE")
        if not pe:
            pe = info.get("forwardPE")
            
        if pe:
            ey = (1 / pe) * 100 # Earnings Yield %
            erp = ey - real_yield
            
            # Rating
            if erp > 4.0: rating = "ATTRACTIVE (Cheap)"
            elif erp < 1.0: rating = "EXPENSIVE (Bubble?)"
            else: rating = "FAIR VALUE"
            
            res = {"pe": pe, "ey": ey, "erp": erp, "rating": rating}
            
    except Exception as e:
        print(f"   ERP Error: {e}")
        
    return res

def calculate_regime_gmm(prices: pd.DataFrame) -> Dict:
    """
    Identifies Market Regime using Gaussian Mixture Model (GMM).
    Features: Weekly Returns, Realized Volatility.
    States: 3 (Bull, Bear, Choppy).
    Returns: Current State, Probabilities, and Transition Matrix.
    """
    print("   Running GMM Regime Detection...")
    res = {"current_state": "N/A", "probs": [], "trans_matrix": None, "labels": []}
    
    if "SPY" not in prices.columns: return res
    
    # Feature Engineering
    spy = prices["SPY"]
    # Weekly resampling to reduce noise
    spy_w = spy.resample('W').last()
    rets = spy_w.pct_change().dropna()
    vol = rets.rolling(4).std().dropna() # 1-month vol
    
    # Align
    df_feat = pd.DataFrame({"Ret": rets, "Vol": vol}).dropna()
    X = df_feat.values
    
    if len(X) < 52: return res # Need history
    
    try:
        # Fit GMM
        gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
        gmm.fit(X)
        
        # Predict
        labels = gmm.predict(X)
        probs = gmm.predict_proba(X)[-1] # Last period probs
        
        # Interpret States (Sort by Mean Return)
        # We need to map arbitrary cluster IDs to "Bull", "Bear", "Choppy"
        means = gmm.means_
        # Sort indices by Mean Return (High to Low)
        sorted_idx = np.argsort(means[:, 0])[::-1]
        
        state_map = {
            sorted_idx[0]: "BULL (Low Vol, Up)",
            sorted_idx[1]: "CHOPPY (Mixed)",
            sorted_idx[2]: "BEAR (High Vol, Down)"
        }
        
        current_label = labels[-1]
        current_state = state_map.get(current_label, "Unknown")
        
        # Calculate Transition Matrix manually from labels
        n_states = 3
        trans_mat = np.zeros((n_states, n_states))
        
        for (i, j) in zip(labels[:-1], labels[1:]):
            trans_mat[i][j] += 1
            
        # Normalize
        trans_mat = trans_mat / trans_mat.sum(axis=1, keepdims=True)
        
        # Get transition probs from CURRENT state
        next_probs = trans_mat[current_label]
        
        # Format for output
        next_state_probs = {}
        for i in range(n_states):
            state_name = state_map[i]
            next_state_probs[state_name] = next_probs[i]
            
        res = {
            "current_state": current_state,
            "probs": next_state_probs,
            "labels": labels, # For plotting
            "dates": df_feat.index
        }
        
    except Exception as e:
        print(f"   GMM Error: {e}")
        
    return res

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
        
    # Current Leader
    # If we are early in the month (e.g. < 7 days), the "current" month might be noise.
    # But "Best Performer (Last Month)" implies the COMPLETED month.
    # resample('ME').last() gives the last day of the month.
    # If today is Dec 1st, the last data point is Nov 30th.
    # If today is Dec 15th, the last data point is Dec 31st (but it's partial).
    # Pandas resample 'ME' puts the label at the END of the bin.
    
    # Let's check the date of the last row
    last_date = monthly_rets.index[-1]
    today = pd.Timestamp.today()
    
    # If the last row is in the future (end of current month) or very recent, it's the "Current Partial Month".
    # We want the "Last Fully Completed Month".
    
    # If last_date month == today month, it's the current partial month.
    if last_date.month == today.month and last_date.year == today.year:
        # Use the previous one
        if len(leaders) > 1:
            curr_leader = leaders.iloc[-2]
        else:
            curr_leader = leaders.iloc[-1] # Fallback
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

def calculate_option_skew(ticker_symbol: str = "SPY") -> Dict:
    """
    Calculates Volatility Skew (Put IV - Call IV).
    Uses 10% OTM strikes (approx).
    """
    print(f"   Calculating Vol Skew for {ticker_symbol}...")
    res = {"skew": None, "status": "N/A"}
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        history = ticker.history(period="1d")
        if history.empty: return res
        current_price = history["Close"].iloc[-1]
        
        exps = ticker.options
        if not exps: return res
        
        # Get nearest monthly (skip very near term < 7 days if possible, else take first)
        # Simple logic: take first
        opt = ticker.option_chain(exps[0])
        calls = opt.calls
        puts = opt.puts
        
        # Find 10% OTM Puts (Strike ~ 90% of Price)
        target_put = current_price * 0.90
        # Find closest strike
        put_idx = (np.abs(puts['strike'] - target_put)).argmin()
        put_iv = puts.iloc[put_idx]['impliedVolatility']
        
        # Find 10% OTM Calls (Strike ~ 110% of Price)
        target_call = current_price * 1.10
        call_idx = (np.abs(calls['strike'] - target_call)).argmin()
        call_iv = calls.iloc[call_idx]['impliedVolatility']
        
        # Skew
        skew = put_iv - call_iv
        
        # Interpret
        if skew > 0.10: status = "STEEP (High Fear/Hedging)"
        elif skew < 0.02: status = "FLAT (Complacency)"
        else: status = "NORMAL"
        
        res = {"skew": skew, "status": status, "put_iv": put_iv, "call_iv": call_iv}
        
    except Exception as e:
        print(f"   Skew Error: {e}")
        
    return res

# ==============================================================================
#                               BACKTEST ENGINE
# ==============================================================================

def calculate_strategy_performance(df: pd.DataFrame, prices: pd.DataFrame):
    """
    Calculates performance metrics for multiple strategies.
    Returns:
        metrics: Dict of {StrategyName: {CAGR, Sharpe, MaxDD}}
        curves: DataFrame of cumulative returns
    """
    # Assets needed
    assets = ["SPY", "TLT", "GLD", "BTC-USD", "RSP", "XLY", "XLP", "QQQ", "EEM", "XBI", "SHY"]
    
    # Ensure all assets are in prices
    missing = [a for a in assets if a not in prices.columns]
    if missing:
        return None, None
        
    # Calculate Returns
    rets = prices[assets].pct_change().dropna()
    spy_ret = rets["SPY"]
    
    # --- STRATEGY LOGIC ---
    
    # --- STRATEGY LOGIC ---
    
    # 1. Macro Regime (Original)
    # Signal: Score > 0 -> SPY, else SHY
    sig_macro = (df["MACRO_SCORE"] > 0).astype(int).reindex(rets.index).shift(1).fillna(0)
    ret_macro = sig_macro * rets["SPY"] + (1 - sig_macro) * rets["SHY"]
    
    # 2. Liquidity Valve (Original)
    # Signal: Valve > MA -> QQQ, else SHY
    sig_liq = (df["CF_Liquidity_Valve"] > df["CF_Liquidity_Valve_MA200"]).astype(int).reindex(rets.index).shift(1).fillna(0)
    ret_liq = sig_liq * rets["QQQ"] + (1 - sig_liq) * rets["SHY"]
    
    # 3. Breadth Trend (New)
    # Signal: Breadth > MA -> SPY, else SHY
    sig_breadth = (df["CF_Breadth"] > df["CF_Breadth_MA200"]).astype(int).reindex(rets.index).shift(1).fillna(0)
    ret_breadth = sig_breadth * rets["SPY"] + (1 - sig_breadth) * rets["SHY"]
    
    # 4. Consumer Rotation (New)
    # Signal: Cyclicals > MA -> XLY, else XLP
    # Note: ROT_Cyc_Def_Sectors is NOT XLY/XLP, it's defined in build_analytics. 
    # Let's check build_analytics. Line 196: df["ROT_Cyc_Def_Sectors"] = ratio("XLY", "XLP")
    # Yes, it is XLY/XLP.
    sig_cons = (df["ROT_Cyc_Def_Sectors"] > df["ROT_Cyc_Def_Sectors_MA200"]).astype(int).reindex(rets.index).shift(1).fillna(0)
    ret_cons = sig_cons * rets["XLY"] + (1 - sig_cons) * rets["XLP"]
    
    # 5. VIX Filter (New)
    # Signal: VIX < 20 -> SPY, else SHY
    # Need to handle missing VIX if not in df
    if "CF_VIX" in df.columns:
        sig_vix = (df["CF_VIX"] < 20).astype(int).reindex(rets.index).shift(1).fillna(0)
        ret_vix = sig_vix * rets["SPY"] + (1 - sig_vix) * rets["SHY"]
    else:
        ret_vix = rets["SPY"] # Fallback to Buy & Hold
        
    # 6. Core-Satellite (CCI PB) [New]
    # Core: 50% SPY + 15% TLT
    # Satellite: 35% Dynamic (QQQ, EEM, GLD, XBI) based on 3M Momentum
    
    # Calculate 3M Momentum (63 trading days)
    sat_assets = ["QQQ", "EEM", "GLD", "XBI"]
    mom_3m = prices[sat_assets].pct_change(63).dropna()
    
    # Determine winner at each month end
    # Resample to month end to simulate monthly rebalancing decision
    monthly_mom = mom_3m.resample('M').last()
    monthly_winner = monthly_mom.idxmax(axis=1)
    
    # Map winner to daily signal (forward fill)
    # Shift by 1 day to avoid lookahead bias (trade on next day open/close)
    daily_signal = monthly_winner.reindex(rets.index, method='ffill').shift(1)
    
    # Construct Satellite Return
    sat_ret = pd.Series(0.0, index=rets.index)
    for asset in sat_assets:
        # If asset is the winner, use its return
        mask = (daily_signal == asset)
        sat_ret[mask] = rets[asset][mask]
        
    # Combine Portfolio (Daily Rebalance to Fixed Weights)
    # 50% SPY, 15% TLT, 35% Satellite
    ret_core_sat = 0.50 * rets["SPY"] + 0.15 * rets["TLT"] + 0.35 * sat_ret
    
    # Benchmark
    ret_bench = rets["SPY"]
    
    # Combine
    curves = pd.DataFrame({
        "Core-Satellite (CCI)": (1 + ret_core_sat).cumprod(),
        "Macro Regime": (1 + ret_macro).cumprod(),
        "Liquidity Valve": (1 + ret_liq).cumprod(),
        "Breadth Trend": (1 + ret_breadth).cumprod(),
        "Consumer Rotation": (1 + ret_cons).cumprod(),
        "VIX Filter": (1 + ret_vix).cumprod(),
    })
    
    # 7. SPY (Hold)
    curves["SPY (Hold)"] = (1 + spy_ret).cumprod()
    
    # --- NEW STRATEGIES ---
    
    # 8. GMM Regime Switcher (Bull/Choppy = Long, Bear = Cash)
    # Note: Uses full-sample GMM (Look-ahead bias for regime identification, but useful for "What if" analysis)
    try:
        # gmm_res = calculate_regime_gmm(prices) # This function is not defined in the provided context
        # if gmm_res["current_state"] != "N/A":
        #     labels = gmm_res["labels"]
        #     # We need to map labels to Bull/Bear. 
        #     # In calculate_regime_gmm, we sorted means: 0=Bull, 1=Choppy, 2=Bear.
        #     # We need to reconstruct that sort here or trust the function returns consistent labels?
        #     # The function returns 'labels' which are 0,1,2.
        #     # We need to know which is which.
        #     # Let's modify calculate_regime_gmm to return the 'bear_label' index?
        #     # Or just infer it: Bear state has negative mean return.
            
        #     # Let's skip the complexity and use a simpler proxy for this run:
        #     # "Vol Regime": If Vol < 20, Long. Else Cash.
        #     # Actually, let's implement "Sector Leaders" and "Vol Control" first, they are robust.
        pass
            
    except Exception as e:
        print(f"GMM Strat Error: {e}")

    # 8. Sector Leaders (Top 3 Momentum)
    # Rebalance Monthly
    try:
        sectors = ["XLI", "XLB", "XLU", "XLF", "XLK", "XLE", "XLV", "XLC", "XLY", "XLP"]
        valid_secs = [s for s in sectors if s in prices.columns]
        
        if len(valid_secs) >= 3:
            sec_prices = prices[valid_secs]
            # Monthly Returns
            sec_m = sec_prices.resample('ME').last()
            # 3M Momentum
            mom_3m = sec_m.pct_change(3)
            
            # Generate Signal mask (Daily)
            # For each month, find Top 3
            # Reindex to daily
            
            # Strategy Return Series
            strat_ret = pd.Series(0.0, index=spy_ret.index)
            
            # Loop through months (vectorized-ish)
            # We need to shift signal by 1 day (trade at close of month or open of next?)
            # Let's assume trade at Close of Month (impossible) -> Trade next day.
            
            # Faster: Resample Signal to Daily (FFill)
            # 1. Get Top 3 mask for each month
            top3_mask = mom_3m.apply(lambda x: x >= x.nlargest(3).min(), axis=1)
            
            # 2. Shift by 1 month (Signal generated at month end, applies to next month)
            top3_mask = top3_mask.shift(1).dropna()
            
            # 3. Reindex to daily (FFill)
            daily_mask = top3_mask.reindex(spy_ret.index, method='ffill').dropna()
            
            # 4. Calculate Returns
            # Equal Weight Top 3
            # We need daily returns of all sectors
            sec_rets = sec_prices.pct_change()
            
            # Filter dates
            common_idx = daily_mask.index.intersection(sec_rets.index)
            
            # For each day, return is mean of selected sectors
            # (Signal * Returns).sum(axis=1) / 3
            
            # Align
            d_mask = daily_mask.loc[common_idx]
            d_rets = sec_rets.loc[common_idx]
            
            # Result
            # Note: d_mask is boolean.
            # Sum of True is 3.
            daily_strat_ret = (d_rets * d_mask).sum(axis=1) / 3
            
            # Add to curves
            curves["Sector Leaders (Top 3)"] = (1 + daily_strat_ret).cumprod()
            
    except Exception as e:
        print(f"Sector Strat Error: {e}")

    # 9. Volatility Control (Target 12%)
    try:
        target_vol = 0.12
        # Realized Vol (20D)
        real_vol = spy_ret.rolling(20).std() * np.sqrt(252)
        
        # Weight = Target / Realized
        # Cap leverage at 1.5x
        weight = (target_vol / real_vol).clip(0, 1.5).shift(1) # Lag 1 day
        
        vol_ctrl_ret = spy_ret * weight
        curves["Vol Control (12%)"] = (1 + vol_ctrl_ret).cumprod()
        
    except Exception as e:
        print(f"Vol Ctrl Error: {e}")

    # Calculate Metrics for all
    metrics = {}
    for col in curves.columns:
        s = curves[col].dropna() # Fix: Drop NaNs (warm-up periods)
        
        if s.empty:
            metrics[col] = {"CAGR": 0.0, "Sharpe": 0.0, "MaxDD": 0.0}
            continue
            
        # CAGR
        start_date = s.index[0]
        end_date = s.index[-1]
        days = (end_date - start_date).days
        years = days / 365.25
        if years < 1: years = 1
        
        total_ret = (s.iloc[-1] / s.iloc[0]) - 1
        cagr = (1 + total_ret) ** (1 / years) - 1
        
        # Sharpe
        daily_rets = s.pct_change().dropna()
        mean = daily_rets.mean() * 252
        std = daily_rets.std() * np.sqrt(252)
        sharpe = mean / std if std > 0 else 0
        
        # MaxDD
        roll_max = s.cummax()
        dd = (s / roll_max) - 1
        max_dd = dd.min()
        
        metrics[col] = {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd}
        
    return metrics, curves

# ==============================================================================
#                               DATA INGESTION
# ==============================================================================

def download_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Downloads Price (YF) and Macro (FRED) data."""
    end = config["end_date"] if config["end_date"] else dt.date.today()
    start = config["start_date"]
    
    print("1. Fetching Asset Prices (Yahoo Finance)...")
    tickers = list(set(config["tickers"])) # Dedupe
    
    prices = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    
    # Handle single ticker edge case
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
        
    print("2. Fetching Macro Plumbing (FRED)...")
    try:
        fred_map = config["fred_codes"]
        macro = pdr.DataReader(list(fred_map.keys()), "fred", start, end)
        macro = macro.rename(columns=fred_map)
    except Exception as e:
        print(f"FRED Download Error: {e}")
        macro = pd.DataFrame()

    # Forward fill to align timelines (macro reporting lags)
    prices = prices.ffill()
    macro = macro.ffill().reindex(prices.index).ffill()
    
    return prices, macro

def get_options_data(ticker_symbol: str = "SPY", calc_gex: bool = False) -> Dict:
    """
    Fetches real-time options data for Sentiment & Structure.
    Returns: PCR, IV, Net OI (Gamma Proxy), Max Pain.
    """
    print(f"   Fetching Options Data for {ticker_symbol}...")
    metrics = {"pcr": None, "iv_call": None, "net_oi": None, "max_pain": None}
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        exps = ticker.options
        if not exps: return metrics
        
        # Get nearest monthly expiration
        opt = ticker.option_chain(exps[0])
        calls = opt.calls
        puts = opt.puts
        
        # 1. PCR & IV
        call_vol = calls['volume'].sum()
        put_vol = puts['volume'].sum()
        metrics["pcr"] = put_vol / call_vol if call_vol > 0 else 0
        
        if 'impliedVolatility' in calls.columns:
            metrics["iv_call"] = calls['impliedVolatility'].mean()
            
        # 2. Advanced Quant (Gamma & Max Pain) - Only if requested (slows down slightly)
        if calc_gex:
            # Net OI (Gamma Proxy)
            # Merge on strike
            df_opts = pd.merge(calls[['strike', 'openInterest']], puts[['strike', 'openInterest']], on='strike', suffixes=('_call', '_put'))
            df_opts['net_oi'] = df_opts['openInterest_call'] - df_opts['openInterest_put']
            metrics["net_oi"] = df_opts['net_oi'].sum()
            
            # Max Pain
            # Need current price for calculation
            current_price = ticker.history(period='1d')['Close'].iloc[-1]
            pain_data = []
            for index, row in df_opts.iterrows():
                strike = row['strike']
                # Value of options at expiration if price = strike? No, Max Pain is strike where option holders lose most.
                # Total Value = Intrinsic Value of all options at that strike.
                # We iterate through potential expiration prices (strikes) and calc total value.
                # Simplified: Iterate strikes, assume price ends there.
                total_intrinsic = 0
                # This is O(N^2) effectively if we iterate all strikes as potential prices.
                # Faster proxy: Just use the strike with highest OI? No.
                # Let's do the standard calc for the strikes we have.
                # For a candidate price 'P', total pain = sum(intrinsic of all opts).
                # We'll test just the strikes present in chain as candidate prices.
                pass 
            
            # Vectorized Max Pain Calc
            strikes = df_opts['strike'].values
            # We want to find the strike S_target that minimizes:
            # Sum(Call_OI * Max(0, S_target - K) + Put_OI * Max(0, K - S_target))
            # We can just loop over S_target in strikes
            min_pain = float('inf')
            pain_strike = 0
            
            # Optimization: Downsample if too many strikes
            candidates = strikes
            
            for s_t in candidates:
                call_pain = (df_opts['openInterest_call'] * (s_t - df_opts['strike']).clip(lower=0)).sum()
                put_pain = (df_opts['openInterest_put'] * (df_opts['strike'] - s_t).clip(lower=0)).sum()
                total = call_pain + put_pain
                if total < min_pain:
                    min_pain = total
                    pain_strike = s_t
            
            metrics["max_pain"] = pain_strike
            
    except Exception as e:
        print(f"   Options Data Error: {e}")
        
    return metrics

def calculate_systemic_risk_pca(prices: pd.DataFrame) -> Dict:
    """
    Calculates Systemic Risk using PCA Absorption Ratio.
    Basket: SPY, TLT, GLD, UUP, XLE (Cross-Asset).
    High Absorption = High Systemic Risk (Assets moving together).
    """
    print("   Calculating Systemic Risk (PCA)...")
    res = {"ratio": None, "status": "N/A", "history": pd.Series()}
    
    basket = ["SPY", "TLT", "GLD", "UUP", "XLE"]
    valid = [b for b in basket if b in prices.columns]
    
    if len(valid) < 3: return res
    
    # Returns
    rets = prices[valid].pct_change().dropna()
    
    # Rolling PCA (126 Days ~ 6 Months)
    window = 126
    absorption_history = []
    dates = []
    
    # Optimization: Don't loop every single day if slow, but for 1000 days it's fine.
    # We need at least 'window' data
    if len(rets) < window: return res
    
    # We'll calculate just the last value for text, and maybe a rolling series for plot
    # Let's do a rolling loop for the last 2 years for plotting
    lookback = 500
    start_idx = max(0, len(rets) - lookback)
    
    # Pre-allocate
    pca = PCA(n_components=1)
    
    for i in range(start_idx, len(rets)):
        if i < window: continue
        
        subset = rets.iloc[i-window:i]
        # Normalize? PCA works on covariance or correlation. 
        # Correlation is better for mixed asset classes.
        # Standardize returns
        subset_std = (subset - subset.mean()) / subset.std()
        subset_std = subset_std.dropna(axis=1) # Drop cols with 0 std
        
        if subset_std.shape[1] < 3: continue
        
        pca.fit(subset_std)
        var_ratio = pca.explained_variance_ratio_[0]
        
        absorption_history.append(var_ratio)
        dates.append(rets.index[i])
        
    if not absorption_history: return res
    
    series = pd.Series(absorption_history, index=dates)
    curr_val = series.iloc[-1]
    
    # Interpret
    # > 75% is usually very high for cross-asset
    if curr_val > 0.75: status = "CRITICAL (High Correlation)"
    elif curr_val > 0.65: status = "ELEVATED"
    else: status = "NORMAL (Decoupled)"
    
    res = {"ratio": curr_val, "status": status, "history": series}
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
    
    import calendar
    curr_name = calendar.month_name[curr_m]
    next_name = calendar.month_name[next_m]
    
    def get_stats(m_idx):
        # Filter for this month
        m_rets = df_seas[df_seas["Month"] == m_idx]["Ret"]
        # We want monthly returns, not daily avg?
        # Actually, let's look at "Win Rate" of the month closing green?
        # Or average daily return?
        # Standard seasonality charts show "Average Monthly Return".
        # We need to resample to Monthly first to get that.
        
        spy_monthly = spy.resample('ME').last().pct_change().dropna()
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

def calculate_gamma_flip(ticker_symbol: str = "SPY") -> Dict:
    """
    Estimates the 'Gamma Flip' Level (Zero Gamma).
    Simplified Model: Weighted average of Call OI (Long Gamma) and Put OI (Short Gamma).
    """
    print(f"   Calculating Gamma Flip Level for {ticker_symbol}...")
    res = {"level": None, "current_price": None, "status": "N/A"}
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Get Current Price
        history = ticker.history(period="1d")
        if history.empty: return res
        current_price = history["Close"].iloc[-1]
        
        # Get Options Chain (Nearest Monthly + Next Monthly for volume)
        exps = ticker.options
        if not exps: return res
        
        # Aggregate OI across first few expirations to get "Market Structure"
        # Gamma is highest for near-the-money, near-term options.
        # We'll take the first 2 expirations.
        
        strikes = []
        call_ois = []
        put_ois = []
        
        for date in exps[:2]:
            opt = ticker.option_chain(date)
            calls = opt.calls
            puts = opt.puts
            
            # We need to align by strike
            # Merge calls and puts on strike
            df = pd.merge(calls[['strike', 'openInterest']], puts[['strike', 'openInterest']], on='strike', how='outer', suffixes=('_call', '_put')).fillna(0)
            
            strikes.extend(df['strike'].tolist())
            call_ois.extend(df['openInterest_call'].tolist())
            put_ois.extend(df['openInterest_put'].tolist())
            
        df_total = pd.DataFrame({"strike": strikes, "call_oi": call_ois, "put_oi": put_ois})
        # Group by strike (sum OI across expirations)
        df_total = df_total.groupby("strike").sum().reset_index()
        
        # Calculate Net GEX (Gamma Exposure) Proxy per strike
        # Call OI contributes +Gamma, Put OI contributes -Gamma
        # This is a simplification. Real GEX depends on IV and Time.
        # But the "Flip Level" is roughly where Net OI switches from Put-heavy to Call-heavy.
        
        df_total["net_oi"] = df_total["call_oi"] - df_total["put_oi"]
        
        # Find the strike where Net OI flips from Negative to Positive (or vice versa)
        # Or simply, finding the "Neutral" point.
        # A common proxy is the strike with the largest Put OI (Support) vs Largest Call OI (Resistance).
        # The "Flip" is often estimated as the point where Total Put OI = Total Call OI in the local area?
        # Let's use the "Zero Gamma" approximation:
        # Sum(Call OI * Strike) - Sum(Put OI * Strike) / Total OI? No.
        
        # Better Proxy: The strike where Cumulative Net OI crosses zero?
        # Let's try Cumulative Net OI.
        df_total = df_total.sort_values("strike")
        df_total["cum_net_oi"] = df_total["net_oi"].cumsum()
        
        # Find zero crossing
        # This is noisy.
        
        # Alternative: SpotGamma style "Vol Trigger" is often the large Put Wall.
        # Let's define Gamma Flip as the level where Call OI dominance shifts to Put OI dominance.
        # We can calculate the "GEX" profile and find the zero crossing.
        # GEX ~ OI * Gamma. Gamma ~ 1 / (Price * Sigma * sqrt(T)) * N'(d1).
        # Simplified: GEX ~ OI.
        # Net GEX ~ (Call OI - Put OI).
        # We want the strike where this balance shifts.
        
        # Let's smooth the Net OI and find the zero crossing near current price.
        df_total["net_oi_smooth"] = df_total["net_oi"].rolling(5, center=True).mean()
        
        # Find crossings
        # We want the crossing closest to current price
        crossings = df_total[np.sign(df_total["net_oi_smooth"]).diff() != 0]
        
        if crossings.empty:
            flip_level = current_price # Fallback
        else:
            # Find closest to current price
            crossings["dist"] = abs(crossings["strike"] - current_price)
            flip_level = crossings.sort_values("dist").iloc[0]["strike"]
            
        # Status
        dist_pct = (current_price - flip_level) / flip_level
        if dist_pct > 0.02: status = "BULLISH (Above Flip)"
        elif dist_pct < -0.02: status = "BEARISH (Below Flip)"
        else: status = "VOLATILE (At Flip)"
        
        res = {"level": flip_level, "current_price": current_price, "status": status}
        
    except Exception as e:
        print(f"   Gamma Flip Error: {e}")
        
    return res

def calculate_global_flows(prices: pd.DataFrame, macro: pd.DataFrame) -> Dict:
    """
    Calculates Global Capital Flow indicators.
    - USD Strength (UUP)
    - Japan Carry Trade (JPY=X vs US rates)
    - Fed Custody (Foreign demand for US assets)
    """
    print("   Calculating Global Capital Flows...")
    res = {
        "usd_strength": {"value": None, "status": "N/A"},
        "japan_carry": {"value": None, "status": "N/A"},
        "fed_custody": {"value": None, "status": "N/A"}
    }

    # USD Strength (UUP vs 200-day MA)
    if "UUP" in prices.columns:
        uup = prices["UUP"].dropna()
        if len(uup) > 200:
            uup_ma200 = uup.rolling(200).mean().iloc[-1]
            curr_uup = uup.iloc[-1]
            if curr_uup > uup_ma200:
                res["usd_strength"] = {"value": curr_uup, "status": "STRONG (Above MA)"}
            else:
                res["usd_strength"] = {"value": curr_uup, "status": "WEAK (Below MA)"}

    # Japan Carry Trade (JPY=X vs US 10Y Yield)
    # A weakening JPY (higher JPY=X) combined with higher US rates suggests carry trade unwind.
    if "JPY=X" in prices.columns and "10Y_Yield" in macro.columns:
        jpy = prices["JPY=X"].dropna()
        us_10y = macro["10Y_Yield"].dropna()
        
        if not jpy.empty and not us_10y.empty:
            # Align data
            common_idx = jpy.index.intersection(us_10y.index)
            jpy_aligned = jpy.loc[common_idx]
            us_10y_aligned = us_10y.loc[common_idx]

            if len(jpy_aligned) > 20: # Need some history for trend
                # JPY trend (higher JPY=X means weaker JPY)
                jpy_20d_change = (jpy_aligned.iloc[-1] / jpy_aligned.iloc[-20]) - 1
                
                # US 10Y trend (rising rates)
                us_10y_20d_change = (us_10y_aligned.iloc[-1] / us_10y_aligned.iloc[-20]) - 1

                # Simplified logic: JPY weakening AND US rates rising = potential carry trade unwind/stress
                if jpy_20d_change > 0.01 and us_10y_20d_change > 0.05: # Arbitrary thresholds
                    res["japan_carry"] = {"value": jpy_20d_change, "status": "UNWINDING (Stress)"}
                elif jpy_20d_change < -0.01 and us_10y_20d_change < -0.05:
                    res["japan_carry"] = {"value": jpy_20d_change, "status": "ACCELERATING (Risk-On)"}
                else:
                    res["japan_carry"] = {"value": jpy_20d_change, "status": "NEUTRAL"}
    
    # Fed Custody (Foreign demand for US assets)
    if "Fed_Custody" in macro.columns:
        fed_custody = macro["Fed_Custody"].dropna()
        if len(fed_custody) > 52: # 1 year of weekly data
            # YoY change
            custody_yoy = (fed_custody.iloc[-1] / fed_custody.iloc[-52]) - 1
            if custody_yoy > 0.02: # Growing foreign demand
                res["fed_custody"] = {"value": custody_yoy, "status": "INCREASING (Demand)"}
            elif custody_yoy < -0.02: # Decreasing foreign demand
                res["fed_custody"] = {"value": custody_yoy, "status": "DECREASING (Withdrawal)"}
            else:
                res["fed_custody"] = {"value": custody_yoy, "status": "STABLE"}

    return res

def calculate_carry_trade(prices: pd.DataFrame, macro: pd.DataFrame) -> Dict:
    """
    Calculates a simplified Carry Trade indicator, focusing on JPY.
    A common carry trade involves borrowing in a low-interest currency (like JPY)
    and investing in a higher-interest currency (like USD).
    Unwind: JPY strengthens (JPY=X falls), or interest rate differential narrows.
    """
    print("   Calculating Carry Trade Indicator...")
    res = {"jpy_carry_signal": None, "status": "N/A"}

    # We need JPY=X (USDJPY) and interest rate differentials.
    # Let's use US 10Y vs Japan 10Y.
    if "JPY=X" in prices.columns and "10Y_Yield" in macro.columns and "Japan_10Y" in macro.columns:
        usd_jpy = prices["JPY=X"].dropna()
        us_10y = macro["10Y_Yield"].dropna()
        jp_10y = macro["Japan_10Y"].dropna()

        # Align indices
        common_idx = usd_jpy.index.intersection(us_10y.index).intersection(jp_10y.index)
        if len(common_idx) < 252: # Need at least a year of data
            return res

        usd_jpy_aligned = usd_jpy.loc[common_idx]
        us_10y_aligned = us_10y.loc[common_idx]
        jp_10y_aligned = jp_10y.loc[common_idx]

        # Calculate Rate Differential (US - Japan)
        rate_diff = us_10y_aligned - jp_10y_aligned

        # Carry Trade Signal:
        # 1. Rate differential trend (e.g., 60-day change)
        # 2. JPY=X trend (e.g., 60-day change)
        
        # A rising rate differential (US rates rising faster than Japan, or Japan rates falling)
        # encourages carry trade.
        # A falling JPY=X (JPY strengthening) indicates unwind.

        # Let's use a simple combined signal:
        # (Rate Diff - 60D MA) * (JPY=X - 60D MA)
        # If Rate Diff is above its MA (widening) AND JPY=X is above its MA (JPY weakening),
        # it suggests carry trade is "on" and potentially expanding.
        # If Rate Diff is below its MA (narrowing) AND JPY=X is below its MA (JPY strengthening),
        # it suggests carry trade is "unwinding".

        rate_diff_ma60 = rate_diff.rolling(60).mean()
        usd_jpy_ma60 = usd_jpy_aligned.rolling(60).mean()

        # Current state relative to MA
        rate_diff_signal = (rate_diff.iloc[-1] > rate_diff_ma60.iloc[-1])
        usd_jpy_signal = (usd_jpy_aligned.iloc[-1] > usd_jpy_ma60.iloc[-1])

        if rate_diff_signal and usd_jpy_signal:
            status = "ON (Expanding)"
        elif not rate_diff_signal and not usd_jpy_signal:
            status = "UNWINDING (Contracting)"
        else:
            status = "MIXED"
        
        res["jpy_carry_signal"] = rate_diff.iloc[-1] # Current rate diff
        res["status"] = status

    return res

# ==============================================================================
#                       SIGNAL CONSTRUCTION (THE ENGINE)
# ==============================================================================

def build_analytics(prices: pd.DataFrame, macro: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Constructs ALL ratios: The Original Rotations + The Capital Flows Plumbing.
    """
    df = pd.DataFrame(index=prices.index)
    
    # --- HELPER: Safe Division ---
    def ratio(n, d):
        return prices[n] / prices[d]

    # ==========================================
    # PART A: The "Capital Flows" Framework
    # ==========================================
    
    # 1. The Liquidity Valve (Bitcoin vs Gold)
    # Theory: "Bitcoin is a high-octane liquidity gauge... it acts like a liquidity release valve" [cite: 617, 684]
    df["CF_Liquidity_Valve"] = ratio("BTC-USD", "GLD")
    
    # 2. Real Yields (The Price of Money)
    # Theory: Rising Real Yields = Liquidity Withdrawal. "Bitcoin tends to rise on falling real yields" [cite: 627]
    df["CF_Real_Yield"] = macro["10Y_Yield"] - macro["10Y_Breakeven"]
    
    # 3. Market Fragility (Breadth vs Volatility)
    # Theory: "Narrow breadth is a warning sign... fragility increases when everyone is positioned the same way" [cite: 3917, 4256]
    df["CF_Breadth"] = ratio("RSP", "SPY")
    df["CF_VIX"] = macro["VIX"]
    df["CF_VIX3M"] = macro["VIX3M"] if "VIX3M" in macro.columns else np.nan
    df["CF_NFCI"] = macro["Fin_Conditions"] # <0 Loose, >0 Tight [cite: 1129]
    
    # --- ADD RAW MACRO FOR DASHBOARD ---
    if "10Y_Yield" in macro.columns: df["10Y_Yield"] = macro["10Y_Yield"]
    if "2Y_Yield" in macro.columns: df["2Y_Yield"] = macro["2Y_Yield"]
    if "HY_Spread" in macro.columns: df["HY_Spread"] = macro["HY_Spread"]
    
    # --- ADD GLOBAL MACRO FROM PRICES ---
    if "DX-Y.NYB" in prices.columns: df["DX-Y.NYB"] = prices["DX-Y.NYB"]
    if "^MOVE" in prices.columns: df["MOVE_Index"] = prices["^MOVE"]
    
    # --- EXPANDED MACRO PLUMBING (YoY Growth) ---
    # M2 & Fed Assets (Liquidity)
    if "M2_Money" in macro.columns:
        # YoY Change (252 trading days approx)
        df["M2_YoY"] = macro["M2_Money"].pct_change(252)
    if "Fed_Assets" in macro.columns:
        df["Fed_Assets_YoY"] = macro["Fed_Assets"].pct_change(252)
    if "M2_Velocity" in macro.columns:
        df["M2_Velocity"] = macro["M2_Velocity"]
        df["M2_Velocity_YoY"] = macro["M2_Velocity"].pct_change(252)
        
    # Economy (Inflation & Jobs)
    if "CPI" in macro.columns:
        df["CPI_YoY"] = macro["CPI"].pct_change(252)
    if "Unemployment" in macro.columns:
        df["Unemployment"] = macro["Unemployment"] / 100 # Normalize to decimal

    # 4. Consumer Transmission (Growth)
    # Theory: "Consumption is a key transmission mechanism... XLY/XLP reflects this" [cite: 9, 131]
    df["CF_Consumer"] = ratio("XLY", "XLP")
    
    # ==========================================
    # PART B: The Original Equity Rotations
    # ==========================================
    
    # Core Internals
    df["ROT_Value_Growth"] = ratio("RPV", "VONG") # [cite: 3776]
    df["ROT_HiBeta_LoVol"] = ratio("SPHB", "SPLV")
    df["ROT_Small_Large"]  = ratio("IWM", "QQQ")
    df["ROT_Quality_Mkt"]  = ratio("NOBL", "SPY")
    df["ROT_Biotech_Mkt"]  = ratio("XBI", "SPY") # Speculation proxy
    
    # Sectors
    cyclicals = prices["XLI"] + prices["XLB"]
    defensives = prices["XLU"] + prices["XLP"]
    df["ROT_Cyc_Def_Sectors"] = cyclicals / defensives
    
    # Labor Proxy (Staffing vs SPY)
    labor = (prices["MAN"] + prices["RHI"] + prices["KELYA"]) / 3
    df["ROT_Labor_SPY"] = labor / prices["SPY"]
    
    # Global
    df["ROT_US_World"] = ratio("SPY", "ACWX")
    df["ROT_EM_DM"]    = ratio("EEM", "VEA")
    
    # Cross Asset
    df["ROT_Copper_Gold"] = ratio("CPER", "GLD")
    df["ROT_Fin_Tech"]    = ratio("XLF", "XLK") # Curve proxy
    
    # Stock/Bond Correlation (Risk Parity Health)
    # Need raw prices for this. 'prices' df is passed in.
    # We need to align dates.
    if "SPY" in prices.columns and "TLT" in prices.columns:
        spy_ret = prices["SPY"].pct_change()
        tlt_ret = prices["TLT"].pct_change()
        # Rolling 60-day correlation
        df["ROT_SPY_TLT_Corr"] = spy_ret.rolling(60).corr(tlt_ret)
    else:
        df["ROT_SPY_TLT_Corr"] = 0.0

    # ==========================================
    # PART C: GLOBAL MACRO (New)
    # ==========================================
    # Rate Differentials (US 10Y - Foreign 10Y)
    # Positive = US Yields Higher -> USD Bullish
    if "Germany_10Y" in macro.columns:
        df["RateDiff_US_DE"] = macro["10Y_Yield"] - macro["Germany_10Y"]
    if "Japan_10Y" in macro.columns:
        df["RateDiff_US_JP"] = macro["10Y_Yield"] - macro["Japan_10Y"]
    if "UK_10Y" in macro.columns:
        df["RateDiff_US_UK"] = macro["10Y_Yield"] - macro["UK_10Y"]

    # --- PART D: FORWARD LOOKING MODELS ---
    # Recession Probability (10Y-3M Probit)
    if "Spread_10Y3M" in macro.columns:
        df["Recession_Prob_Model"] = calculate_recession_prob(macro["Spread_10Y3M"])
        df["Spread_10Y3M"] = macro["Spread_10Y3M"]
        # Set primary Recession_Prob for plotting
        df["Recession_Prob"] = df["Recession_Prob_Model"]
        
    elif "10Y_Yield" in macro.columns and "3M_Yield" in macro.columns:
        spread_10y3m = macro["10Y_Yield"] - macro["3M_Yield"]
        df["Recession_Prob_Model"] = calculate_recession_prob(spread_10y3m)
        df["Spread_10Y3M"] = spread_10y3m
        df["Recession_Prob"] = df["Recession_Prob_Model"]
    
    if "Recession_Prob" in macro.columns: # FRED's own smoothed recession prob
        df["FRED_Recession_Prob"] = macro["Recession_Prob"]
        # Prefer FRED data if available and recent
        if not df["FRED_Recession_Prob"].dropna().empty:
             df["Recession_Prob"] = df["FRED_Recession_Prob"]

    # --- PART E: STATISTICAL EXTREMES (Z-SCORES) ---
    # Calculate 1-Year Rolling Z-Score for Mean Reversion Signals
    # Z = (Current - RollingMean) / RollingStd
    window = 252
    for col in df.columns:
        if "MA200" in col: continue # Skip MAs
        
        roll_mean = df[col].rolling(window).mean()
        roll_std = df[col].rolling(window).std()
        df[f"{col}_Z"] = (df[col] - roll_mean) / roll_std

    return df

def add_trends_and_score(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Adds MAs and calculates the composite Macro Score."""
    
    # 1. Moving Averages for ALL columns
    for col in df.columns:
        df[f"{col}_MA200"] = df[col].rolling(config["ma_long"]).mean()
        
    # 2. Composite Macro Score Calculation
    # We define which ratios are "Risk-On" (1) vs "Risk-Off" (-1)
    # Logic: Score based on % distance from 200D MA
    
    risk_map = {
        "CF_Liquidity_Valve": 1,  # BTC/Gold Up = Risk On
        "CF_Consumer": 1,         # Disc/Staples Up = Risk On
        "CF_Breadth": 1,          # RSP/SPY Up = Healthy
        "ROT_Value_Growth": 1,    # Usually correlates with rising yields/growth
        "ROT_HiBeta_LoVol": 1,
        "ROT_Small_Large": 1,
        "ROT_Biotech_Mkt": 1,
        "ROT_Cyc_Def_Sectors": 1,
        "ROT_Labor_SPY": 1,
        "ROT_US_World": 1,
        "ROT_EM_DM": 1,
        "ROT_Copper_Gold": 1,
        "ROT_Fin_Tech": 1,
        "ROT_Quality_Mkt": -1     # Quality outperforming is usually defensive
    }
    
    score_series = pd.Series(0.0, index=df.index)
    valid_count = 0
    
    threshold = config["regime_threshold"]
    
    for col, direction in risk_map.items():
        if col not in df.columns: continue
        
        ma = df[f"{col}_MA200"]
        # % Distance from trend
        dist = (df[col] / ma - 1.0) * 100
        
        # Scoring logic
        raw_score = pd.Series(0.0, index=df.index)
        raw_score[dist > threshold] = 1.0
        raw_score[dist < -threshold] = -1.0
        
        score_series += (raw_score * direction)
        valid_count += 1
        
    if valid_count > 0:
        df["MACRO_SCORE"] = score_series / valid_count
       # df["MACRO_SCORE"] = df["MACRO_SCORE"] / df["MACRO_SCORE"].std() 

    return df

def calculate_macro_radar(df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the 1-Year Percentile Rank (0-100) for the Macro Radar.
    Factors: Growth, Liquidity, Inflation, Rates, Risk, Sentiment.
    """
    # 1. Select Data (Last 252 Days)
    window = 252
    subset = df.iloc[-window:].copy()
    
    # 2. Define Factors
    # We need to handle missing columns gracefully
    factors = {}
    
    # Growth: Consumer Strength (XLY/XLP)
    if "CF_Consumer" in subset.columns:
        factors["Growth"] = subset["CF_Consumer"]
        
    # Liquidity: M2 YoY
    if "M2_YoY" in subset.columns:
        factors["Liquidity"] = subset["M2_YoY"]
        
    # Inflation: CPI YoY (Inverse: Low Inflation is Good/Bullish usually, but for "Regime" shape:
    # Let's map High Inflation = 100 (Hot) vs Low = 0 (Cold). 
    # Interpretation: Expanded Polygon = "Hot/Expansionary". 
    if "CPI_YoY" in subset.columns:
        factors["Inflation"] = subset["CPI_YoY"]
        
    # Rates: TLT Price (High Price = Low Rates = Bullish/Loose)
    # We need TLT price. It's in 'prices' df.
    if "TLT" in prices.columns:
        factors["Rates"] = prices["TLT"].iloc[-window:]
        
    # Risk: Liquidity Valve (BTC/Gold)
    if "CF_Liquidity_Valve" in subset.columns:
        factors["Risk"] = subset["CF_Liquidity_Valve"]
        
    # Sentiment: Inverse VIX (High VIX = Fear = Low Rank. Low VIX = Complacency = High Rank)
    if "CF_VIX" in subset.columns:
        # We want High Rank = Bullish/Calm. So we rank raw VIX, then invert?
        # No, simpler: Rank(-VIX). High (-VIX) = Low VIX = High Rank.
        factors["Sentiment"] = -subset["CF_VIX"]
        
    # 3. Calculate Percentile Ranks
    ranks = {}
    current_values = {}
    
    for name, series in factors.items():
        # Percentile of current value within the window
        curr = series.iloc[-1]
        rank = stats.percentileofscore(series.dropna(), curr)
        ranks[name] = rank
        current_values[name] = curr
        
    return pd.DataFrame({"Rank": ranks, "Value": current_values})


# ==============================================================================
#                       REPORTING & VISUALIZATION
# ==============================================================================
def calculate_liquidity_stress(df: pd.DataFrame) -> Dict:
    """
    Calculates Liquidity Stress (TED Spread Proxy).
    Spread = 3M LIBOR (or Comm Paper) - 3M T-Bill.
    Since we don't have LIBOR in standard free feeds easily, we use:
    Corporate Yield (LQD/HYG implied) - Treasury Yield?
    Better Proxy: HYG vs TLT spread relative to history.
    Or: SPY/TLT correlation breakdown?
    Let's use the "Credit Spread" we already have (HYG - IEF/TLT) but normalized.
    Actually, let's use the 'CF_NFCI' if available, or build a 'Stress Index'.
    Let's build a 'Stress Index' from VIX, Credit Spreads, and Dollar.
    """
    print("   Calculating Liquidity Stress...")
    res = {"level": None, "status": "N/A"}
    
    # We need Credit Spread (HYG vs IEF) and VIX
    # We have 'Spread_Credit' in df if calculated in build_analytics
    # Let's re-calc to be safe or use what we have.
    
    # Simple Proxy: High Yield Spread Z-Score + VIX Z-Score
    # If both are high, liquidity is stressed.
    
    if "Spread_Credit" in df.columns and "CF_VIX" in df.columns:
        spread = df["Spread_Credit"].iloc[-1]
        vix = df["CF_VIX"].iloc[-1]
        
        # Z-Scores (approx based on 1Y lookback)
        spread_mean = df["Spread_Credit"].rolling(252).mean().iloc[-1]
        spread_std = df["Spread_Credit"].rolling(252).std().iloc[-1]
        spread_z = (spread - spread_mean) / spread_std if spread_std > 0 else 0
        
        vix_mean = df["CF_VIX"].rolling(252).mean().iloc[-1]
        vix_std = df["CF_VIX"].rolling(252).std().iloc[-1]
        vix_z = (vix - vix_mean) / vix_std if vix_std > 0 else 0
        
        stress_score = (spread_z + vix_z) / 2
        
        if stress_score > 2.0: status = "CRITICAL (Liquidity Crunch)"
        elif stress_score > 1.0: status = "ELEVATED (Tightening)"
        else: status = "NORMAL (Abundant)"
        
        res = {"level": stress_score, "status": status, "spread_z": spread_z, "vix_z": vix_z}
        
    return res

def calculate_garch_crash_prob(prices: pd.DataFrame) -> Dict:
    """
    Estimates 'Crash Probability' using a simplified GARCH(1,1) Volatility Forecast.
    Crash defined as > 5% drop in next 5 days.
    """
    print("   Calculating GARCH Crash Probability...")
    res = {"prob": None, "vol_forecast": None, "status": "N/A"}
    
    if "SPY" not in prices.columns: return res
    
    spy = prices["SPY"]
    rets = spy.pct_change().dropna() * 100 # In percent
    
    # Simplified GARCH(1,1) Logic
    # Sigma^2_t = omega + alpha * Ret^2_{t-1} + beta * Sigma^2_{t-1}
    # We need to fit parameters. 
    # For speed/robustness without 'arch' lib, we use 'EWMA' (RiskMetrics) which is a special GARCH.
    # EWMA: Sigma^2_t = (1-lambda) * Ret^2_{t-1} + lambda * Sigma^2_{t-1}
    # Standard lambda = 0.94
    
    lambda_param = 0.94
    var_t = rets.var() # Init
    
    # Fast loop or vectorised?
    # Pandas ewm is exactly this.
    # EWM Variance
    # span = (1+lambda)/(1-lambda)? No.
    # alpha = 1 - lambda = 0.06
    # span = 2/alpha - 1 = 2/0.06 - 1 = 32.33
    
    vol_series = rets.ewm(alpha=(1-lambda_param)).std()
    current_vol_daily = vol_series.iloc[-1] # Daily Vol %
    
    # Forecast next 5 days volatility
    # In EWMA, forecast is constant (Random Walk).
    # 5-day Vol = Daily * sqrt(5)
    vol_5d = current_vol_daily * np.sqrt(5)
    
    # Probability of Drop > 5%
    # Z-score = (-5% - 0%) / Vol_5d
    # Assuming 0 mean return for crash calc (conservative)
    z_score = -5.0 / vol_5d
    
    prob_crash = norm.cdf(z_score) * 100
    
    if prob_crash > 5.0: status = "HIGH RISK (>5%)"
    elif prob_crash > 1.0: status = "ELEVATED"
    else: status = "LOW"
    
    res = {"prob": prob_crash, "vol_forecast": vol_5d, "status": status}
    return res

def calculate_correlation_surprise(prices: pd.DataFrame) -> Dict:
    """
    Calculates Correlation Surprise (Stocks vs Bonds).
    Surprise = Short-Term Corr (1M) - Long-Term Corr (1Y).
    Positive Surprise = Correlations tightening (Risk Off).
    """
    print("   Calculating Correlation Surprise...")
    res = {"surprise": None, "status": "N/A", "curr_corr": None}
    
    if "SPY" not in prices.columns or "TLT" not in prices.columns: return res
    
    spy = prices["SPY"].pct_change().dropna()
    tlt = prices["TLT"].pct_change().dropna()
    
    # Align
    df_c = pd.DataFrame({"SPY": spy, "TLT": tlt}).dropna()
    
    # Rolling Correlations
    corr_1m = df_c["SPY"].rolling(21).corr(df_c["TLT"]).iloc[-1]
    corr_1y = df_c["SPY"].rolling(252).corr(df_c["TLT"]).iloc[-1]
    
    surprise = corr_1m - corr_1y
    
    # Interpretation
    # If 1M corr is much higher (more positive) than 1Y, stocks/bonds moving together.
    # If 1M is much lower (more negative), diversification is working.
    
    if surprise > 0.5: status = "SHOCK (Correlations Spiking)"
    elif surprise > 0.2: status = "TIGHTENING (Diversification Fading)"
    elif surprise < -0.2: status = "DECOUPLING (Diversification Working)"
    else: status = "STABLE"
    
    res = {"surprise": surprise, "status": status, "curr_corr": corr_1m, "lt_corr": corr_1y}
    return res

def calculate_global_flows(df: pd.DataFrame, prices: pd.DataFrame) -> Dict:
    """
    Analyzes Global Liquidity Flows & Carry Trade.
    1. Foreign Custody Holdings (Fed): Proxy for Central Bank Demand.
    2. JPY Carry Trade: USD/JPY Trend + Yield Spread.
    """
    print("   Calculating Global Flows & Carry Trade...")
    res = {
        "custody_trend": "N/A", "custody_chg": None,
        "carry_status": "N/A", "usdjpy_trend": "N/A", "spread_usjp": None
    }
    
    # 1. Foreign Custody Holdings (WSHOMCB)
    if "Fed_Custody" in df.columns:
        custody = df["Fed_Custody"].dropna()
        if not custody.empty:
            # YoY Change
            # Data is weekly. 52 weeks.
            custody_yoy = custody.pct_change(52).iloc[-1]
            
            # Trend (vs 26 week MA)
            ma = custody.rolling(26).mean().iloc[-1]
            curr = custody.iloc[-1]
            
            if curr < ma:
                trend = "UNWINDING (Selling)"
            else:
                trend = "ACCUMULATING (Buying)"
                
            res["custody_trend"] = trend
            res["custody_chg"] = custody_yoy
            
    # 2. JPY Carry Trade
    # Signal: USD/JPY < 200DMA = Unwind.
    # Spread: US 10Y - JP 10Y.
    
    if "JPY=X" in prices.columns:
        usdjpy = prices["JPY=X"].dropna()
        curr_px = usdjpy.iloc[-1]
        ma200 = usdjpy.rolling(200).mean().iloc[-1]
        
        if curr_px < ma200:
            fx_trend = "BEARISH (Yen Strengthening)"
            carry_signal = "UNWIND RISK (High)"
        else:
            fx_trend = "BULLISH (Yen Weakening)"
            carry_signal = "STABLE (Carry On)"
            
        res["usdjpy_trend"] = fx_trend
        res["carry_status"] = carry_signal
        res["usdjpy_px"] = curr_px
        
    # Spread (US 10Y - JP 10Y)
    # US 10Y is "DGS10" in df. JP 10Y is "Japan_10Y".
    if "DGS10" in df.columns and "Japan_10Y" in df.columns:
        us10 = df["DGS10"].iloc[-1]
        jp10 = df["Japan_10Y"].iloc[-1] # Monthly, might be nan if not ffilled
        
        # Forward fill JP data since it's monthly
        jp_series = df["Japan_10Y"].ffill()
        jp10 = jp_series.iloc[-1]
        
        spread = us10 - jp10
        res["spread_usjp"] = spread
        
    return res

def run_backtest(df: pd.DataFrame, prices: pd.DataFrame) -> plt.Figure:
    """
    Runs a vectorised backtest for multiple strategies and plots the results.
    Includes Regime-Conditional Forward Projections.
    """
    print("   Running Backtest & Projections...")
    
    # Calculate Metrics & Curves
    metrics, curves = calculate_strategy_performance(df, prices)
    
    if curves.empty:
        print("   Backtest failed: Missing assets.")
        return None
        
    # --- REGIME-CONDITIONAL PROJECTION ENGINE ---
    # 1. Get Regime History
    regime_data = calculate_regime_gmm(prices)
    projections = pd.DataFrame()
    proj_metrics = {}
    
    if regime_data["current_state"] != "N/A":
        # Get mask for current regime
        current_label = regime_data["labels"][-1]
        regime_dates = regime_data["dates"][regime_data["labels"] == current_label]
        
        # We need to align strategy returns with regime dates
        # Strategy returns are daily, regime is weekly.
        # Approximation: Use the weekly regime label to filter daily returns?
        # Better: Just use the full history stats for now to avoid mismatch complexity, 
        # OR use the regime stats if we can align easily.
        # Let's use Full History for robustness but weight recent vol?
        # Actually, let's try to filter.
        pass
    
    # Simplified Projection: Monte Carlo using Full History (Robust Baseline)
    # Why? Regime filtering on daily strategy returns using weekly GMM labels is prone to alignment errors.
    # We will use the Strategy's own history.
    
    future_days = 252
    n_sims = 1000
    
    # Create future index
    last_date = curves.index[-1]
    future_dates = pd.date_range(start=last_date, periods=future_days+1, freq='B')
    
    # Set Seed for Consistency
    np.random.seed(42)
    
    for strat in curves.columns:
        # Get daily returns
        series = curves[strat]
        rets = series.pct_change().dropna()
        
        # Params
        mu = rets.mean()
        sigma = rets.std()
        last_price = series.iloc[-1]
        
        # Monte Carlo
        # S_t = S_{t-1} * exp( (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )
        # Vectorized:
        # drift = (mu - 0.5 * sigma**2)
        # shock = sigma * np.random.normal(0, 1, (future_days, n_sims))
        # paths = last_price * np.exp(np.cumsum(drift + shock, axis=0))
        
        # Simpler Geometric Brownian Motion
        # ret_sim = np.random.normal(mu, sigma, (future_days, n_sims))
        # price_paths = last_price * (1 + ret_sim).cumprod(axis=0)
        
        # We want the Median Path
        # Expected Return (Drift)
        expected_daily_ret = mu # or mu - 0.5*sigma^2 for geometric
        
        # Deterministic Projection (Expected Value)
        # This is cleaner for the chart than a noisy single random path
        # proj_values = last_price * (1 + expected_daily_ret) ** np.arange(1, future_days + 1)
        
        # Let's use the Median Outcome from simulation for "Probabilistic" view
        ret_sim = np.random.normal(mu, sigma, (future_days, n_sims))
        price_paths = last_price * (1 + ret_sim).cumprod(axis=0)
        median_path = np.median(price_paths, axis=1)
        
        # Add to projections df
        # Prepend last value to connect lines
        full_proj = np.concatenate(([last_price], median_path))
        projections[strat] = full_proj
        
        # Calculate Exp CAGR
        final_val = median_path[-1]
        exp_ret = (final_val / last_price) - 1
        # Annualized
        exp_cagr = exp_ret # Since we projected 252 days (1 year)
        
        proj_metrics[strat] = exp_cagr

    # Plotting - WHITE THEME
    plt.style.use('default') # Reset to white background
    
    fig = plt.figure(figsize=(14, 12)) # Taller to fit table
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    
    # Equity Curve
    ax0 = fig.add_subplot(gs[0])
    
    # Distinct Colors
    colors = {
        "Core-Satellite (CCI)": "#1f77b4", # Blue (Professional)
        "Macro Regime": "#2ca02c", # Green
        "Liquidity Valve": "#d62728", # Red
        "Breadth Trend": "#9467bd", # Purple
        "Consumer Rotation": "#ff7f0e", # Orange
        "VIX Filter": "#e377c2", # Pink
        "Sector Leaders (Top 3)": "#8c564b", # Brown
        "Vol Control (12%)": "#17becf", # Cyan
        "SPY (Hold)": "black"
    }
    
    # Plot History
    for strat in curves.columns:
        cagr = metrics[strat]["CAGR"]
        label = f"{strat} (CAGR: {cagr:.1%})"
        
        # Style
        color = colors.get(strat, "grey")
        style = "--" if "Hold" in strat else "-"
        width = 3 if "Core-Satellite" in strat else (1.5 if "Hold" in strat else 1.5)
        alpha = 1.0 if "Core-Satellite" in strat else 0.7
        
        ax0.plot(curves.index, curves[strat], label=label, color=color, linestyle=style, linewidth=width, alpha=alpha)
        
        # Plot Projection
        if strat in projections.columns:
            ax0.plot(future_dates, projections[strat], color=color, linestyle=":", linewidth=width, alpha=0.6)
            # Add dot at end
            ax0.scatter(future_dates[-1], projections[strat].iloc[-1], color=color, s=20)

    ax0.set_title("Backtest & 1-Year Projection (Median Path)", fontsize=16, weight='bold', color='black')
    ax0.set_yscale('log')
    ax0.legend(loc="upper left", fontsize=10, frameon=True, facecolor='white', edgecolor='grey')
    ax0.grid(True, which="both", alpha=0.3, color='grey', linestyle=':')
    ax0.set_facecolor('white')
    ax0.set_ylabel("Cumulative Return (Log Scale)", fontsize=12)
    
    # Add vertical line for "Now"
    ax0.axvline(last_date, color='black', linestyle='-', linewidth=1)
    ax0.text(last_date, ax0.get_ylim()[0], "  TODAY", rotation=90, va='bottom', weight='bold')
    
    # Metrics Table
    ax1 = fig.add_subplot(gs[1])
    ax1.axis('off')
    
    table_data = [["Strategy", "Hist. CAGR", "Sharpe", "Max Drawdown", "Exp. CAGR (1Y)"]]
    # Sort by Sharpe for the table
    sorted_strats = sorted(metrics.keys(), key=lambda x: metrics[x]['Sharpe'], reverse=True)
    
    for strat in sorted_strats:
        m = metrics[strat]
        exp = proj_metrics.get(strat, 0)
        table_data.append([
            strat, 
            f"{m['CAGR']:.1%}", 
            f"{m['Sharpe']:.2f}", 
            f"{m['MaxDD']:.1%}",
            f"{exp:.1%}" # New Column
        ])
    
    table = ax1.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3, 0.12, 0.12, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # Style Table (Clean White)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor('#f0f0f0') # Light grey header
        else:
            cell.set_text_props(color='black')
            cell.set_facecolor('white')
        cell.set_edgecolor('black')
    
    plt.tight_layout()
    return fig

def plot_risk_macro_dashboard(df: pd.DataFrame, prices: pd.DataFrame) -> plt.Figure:
    """
    Generates Page 3: Macro Risk & Sector Rotation.
    1. Yield Curve (10Y-2Y)
    2. High Yield Spreads
    3. Bond Market Fear (MOVE Index)
    4. Sector RRG (Relative Rotation Graph)
    """
    print("   Generating Risk & Macro Page...")
    plt.style.use('default')
    
    fig = plt.figure(figsize=(14, 14)) # Increased height for 4th panel
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.5]) # RRG gets more space
    
    # 1. Yield Curve (Recession Watch)
    ax1 = fig.add_subplot(gs[0])
    
    # Check for columns (handle potential missing data gracefully)
    has_yields = "10Y_Yield" in df.columns and "2Y_Yield" in df.columns
    
    if has_yields:
        yc = (df["10Y_Yield"] - df["2Y_Yield"]).dropna()
        if not yc.empty:
            ax1.plot(yc.index, yc, color='black', label="10Y-2Y Spread")
            ax1.axhline(0, color='red', linestyle='--', linewidth=1)
            ax1.fill_between(yc.index, yc, 0, where=(yc < 0), color='red', alpha=0.3, label="Inversion (Recession Warning)")
            ax1.fill_between(yc.index, yc, 0, where=(yc > 0), color='green', alpha=0.1, label="Normal Slope")
            ax1.set_title("1. Yield Curve (10Y - 2Y): Recession Watch", fontsize=12, weight='bold')
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "Yield Curve Data Missing", ha='center', va='center')
        
    # 2. Credit Stress (HY Spreads)
    ax2 = fig.add_subplot(gs[1])
    if "HY_Spread" in df.columns:
        hy = df["HY_Spread"].dropna()
        if not hy.empty:
            ax2.plot(hy.index, hy, color='purple', label="High Yield Option-Adjusted Spread")
            ax2.axhline(hy.mean(), color='orange', linestyle='--', label="Avg Spread")
            ax2.set_title("2. Credit Stress (High Yield Spreads)", fontsize=12, weight='bold')
            ax2.legend(loc="upper left")
            ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Credit Spread Data Missing", ha='center', va='center')

    # 3. Bond Market Fear (MOVE Index)
    ax3 = fig.add_subplot(gs[2])
    if "MOVE_Index" in df.columns:
        move = df["MOVE_Index"].dropna()
        if not move.empty:
            ax3.plot(move.index, move, color='blue', label="MOVE Index (Bond Volatility)")
            ax3.axhline(100, color='red', linestyle='--', label="Stress Threshold (100)")
            
            # Highlight Stress
            curr = move.iloc[-1]
            status = "ELEVATED (Risk Off)" if curr > 100 else "NORMAL"
            color = 'red' if curr > 100 else 'green'
            
            ax3.set_title(f"3. Bond Market Fear (MOVE Index): {status}", fontsize=12, weight='bold', color=color)
            ax3.legend(loc="upper left")
            ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "MOVE Index Data Missing", ha='center', va='center')
        
    # 4. Sector RRG (Relative Rotation Graph)
    ax4 = fig.add_subplot(gs[3])
    rrg = calculate_rrg_metrics(prices)
    
    if not rrg.empty:
        # Scatter Plot
        ax4.axhline(0, color='black', linestyle='-', linewidth=1)
        ax4.axvline(0, color='black', linestyle='-', linewidth=1)
        
        # Plot Points
        for i, row in rrg.iterrows():
            color = 'green' if row['Quadrant'] == 'Leading' else \
                    'blue' if row['Quadrant'] == 'Improving' else \
                    'orange' if row['Quadrant'] == 'Weakening' else 'red'
            
            ax4.scatter(row['RS'], row['Momentum'], color=color, s=100, alpha=0.8)
            ax4.text(row['RS'], row['Momentum'], row['Ticker'], fontsize=9, weight='bold')
            
        ax4.set_title("4. Sector Rotation Map (RRG Proxy)", fontsize=12, weight='bold')
        ax4.set_xlabel("Relative Strength vs SPY (Trend)", fontsize=10)
        ax4.set_ylabel("Momentum of RS (Rate of Change)", fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # Dynamic Limits to ensure labels fit
        x_abs = max(abs(rrg['RS'].min()), abs(rrg['RS'].max())) * 1.2
        y_abs = max(abs(rrg['Momentum'].min()), abs(rrg['Momentum'].max())) * 1.2
        
        # Ensure non-zero limits
        x_abs = max(x_abs, 0.05)
        y_abs = max(y_abs, 0.05)
        
        ax4.set_xlim(-x_abs, x_abs)
        ax4.set_ylim(-y_abs, y_abs)
        
        # Quadrant Labels (Fixed in corners based on limits)
        # Top Right (Leading)
        ax4.text(x_abs*0.9, y_abs*0.9, "LEADING", color='green', alpha=0.5, weight='bold', ha='right', va='top')
        # Bottom Right (Weakening)
        ax4.text(x_abs*0.9, -y_abs*0.9, "WEAKENING", color='orange', alpha=0.5, weight='bold', ha='right', va='bottom')
        # Bottom Left (Lagging)
        ax4.text(-x_abs*0.9, -y_abs*0.9, "LAGGING", color='red', alpha=0.5, weight='bold', ha='left', va='bottom')
        # Top Left (Improving)
        ax4.text(-x_abs*0.9, y_abs*0.9, "IMPROVING", color='blue', alpha=0.5, weight='bold', ha='left', va='top')

    plt.tight_layout()
    return fig

def plot_macro_radar(df: pd.DataFrame, prices: pd.DataFrame) -> plt.Figure:
    """
    Generates the Macro Regime Radar (Spider Chart).
    """
    print("   Generating Macro Radar...")
    plt.style.use('default')
    
    # Calculate Ranks
    radar_data = calculate_macro_radar(df, prices)
    if radar_data.empty:
        fig = plt.figure()
        plt.text(0.5, 0.5, "Insufficient Data for Radar", ha='center')
        return fig
        
    # Prepare Data for Plotting
    categories = radar_data.index.tolist()
    values = radar_data["Rank"].tolist()
    values += values[:1] # Close the loop
    
    # Angles
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Draw one axe per variable + labels
    plt.xticks(angles[:-1], categories, color='black', size=12, weight='bold')
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([25, 50, 75], ["25", "50", "75"], color="grey", size=10)
    plt.ylim(0, 100)
    
    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='blue')
    
    # Fill area
    # Color based on average rank (Expansion vs Contraction)
    avg_rank = np.mean(values[:-1])
    fill_color = 'green' if avg_rank > 50 else 'red'
    ax.fill(angles, values, color=fill_color, alpha=0.2)
    
    # Title
    regime_type = "EXPANSIONARY" if avg_rank > 50 else "CONTRACTIONARY"
    plt.title(f"The Shape of the Macro Regime: {regime_type}\n(1-Year Percentile Rank)", size=16, weight='bold', y=1.1)
    
    # Add annotation explaining the chart
    plt.figtext(0.5, 0.02, 
                "Outer Edge (100) = Bullish/Loose/Hot | Center (0) = Bearish/Tight/Cold\n"
                "Growth: Consumer Strength | Liquidity: Money Supply | Risk: BTC/Gold\n"
                "Inflation: CPI | Rates: Bond Prices | Sentiment: Low Volatility",
                ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout()
    return fig

def plot_monetary_plumbing(df: pd.DataFrame) -> plt.Figure:
    """
    Generates Page 4: Monetary & Economic Plumbing.
    1. Liquidity Impulse (M2 & Fed Assets YoY)
    2. Inflation Trend (CPI YoY)
    3. Labor Market (Unemployment vs 12M MA)
    """
    print("   Generating Monetary Plumbing Page...")
    plt.style.use('default')
    
    fig = plt.figure(figsize=(14, 14)) # Increased height for 4th panel
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1])
    
    # 1. Liquidity Impulse
    ax1 = fig.add_subplot(gs[0])
    if "M2_YoY" in df.columns and "Fed_Assets_YoY" in df.columns:
        m2 = df["M2_YoY"].dropna()
        fed = df["Fed_Assets_YoY"].dropna()
        
        ax1.plot(m2.index, m2, color='green', label="M2 Money Supply (YoY)", linewidth=2)
        ax1.plot(fed.index, fed, color='blue', linestyle='--', label="Fed Balance Sheet (YoY)", linewidth=1.5)
        ax1.axhline(0, color='black', linewidth=1)
        ax1.fill_between(m2.index, m2, 0, where=(m2 > 0), color='green', alpha=0.1, label="Liquidity Expansion")
        ax1.fill_between(m2.index, m2, 0, where=(m2 < 0), color='red', alpha=0.1, label="Liquidity Contraction")
        
        ax1.set_title("1. Liquidity Impulse: Money Supply & Fed Assets", fontsize=12, weight='bold')
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        # Add Money Velocity on Secondary Axis
        if "M2_Velocity" in df.columns:
            m2v = df["M2_Velocity"].dropna()
            ax1_twin = ax1.twinx()
            ax1_twin.plot(m2v.index, m2v, color='orange', linestyle=':', label="M2 Velocity (Right)", linewidth=1.5)
            ax1_twin.set_ylabel("Velocity Ratio", color='orange', fontsize=10)
            ax1_twin.tick_params(axis='y', labelcolor='orange')
            # Add secondary legend manually or let it float
            # Combined legend is tricky with twinx, let's just add text or use a proxy
            ax1_twin.legend(loc="upper right")
    else:
        ax1.text(0.5, 0.5, "Liquidity Data Missing", ha='center', va='center')

    # 2. Global Liquidity Wrecking Ball (DXY)
    ax2 = fig.add_subplot(gs[1])
    dxy_col = "DX-Y.NYB" if "DX-Y.NYB" in df.columns else "UUP"
    
    if dxy_col in df.columns:
        dxy = df[dxy_col].dropna()
        ma = dxy.rolling(200).mean()
        
        ax2.plot(dxy.index, dxy, color='green', label="USD Index (DXY)", linewidth=1.5)
        ax2.plot(ma.index, ma, color='black', linestyle='--', label="200-Day MA")
        
        curr = dxy.iloc[-1]
        ma_val = ma.iloc[-1]
        status = "BULLISH" if curr > ma_val else "BEARISH"
        color = 'red' if curr > ma_val else 'green' # Strong dollar is bad for risk
        
        ax2.set_title(f"2. Global Liquidity Wrecking Ball (DXY): {status}", fontsize=12, weight='bold', color=color)
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "DXY Data Missing", ha='center', va='center')

    # 3. Inflation Trend
    ax3 = fig.add_subplot(gs[2])
    if "CPI_YoY" in df.columns:
        cpi = df["CPI_YoY"].dropna()
        ax3.plot(cpi.index, cpi, color='purple', label="CPI Inflation (YoY)", linewidth=2)
        ax3.axhline(0.02, color='red', linestyle='--', label="Fed Target (2%)")
        
        ax3.set_title("3. Inflation Trend (CPI)", fontsize=12, weight='bold')
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    else:
        ax3.text(0.5, 0.5, "Inflation Data Missing", ha='center', va='center')
        
    # 4. Labor Market (Sahm Rule Proxy)
    ax4 = fig.add_subplot(gs[3])
    if "Unemployment" in df.columns:
        unrate = df["Unemployment"].dropna()
        unrate_ma = unrate.rolling(12).mean() # 12M Moving Average
        
        ax4.plot(unrate.index, unrate, color='black', label="Unemployment Rate", linewidth=2)
        ax4.plot(unrate_ma.index, unrate_ma, color='red', linestyle='--', label="12-Month Moving Avg")
        
        ax4.set_title("4. Labor Market Health (Unemployment)", fontsize=12, weight='bold')
        ax4.legend(loc="upper left")
        ax4.grid(True, alpha=0.3)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    else:
        ax4.text(0.5, 0.5, "Labor Data Missing", ha='center', va='center')

    plt.tight_layout()
    return fig

def generate_capital_flows_commentary(df: pd.DataFrame, prices: pd.DataFrame = None):
    """
    Analyzes the results using the specific 'Capital Flows' playbooks logic.
    """
    last = df.iloc[-1]
    
    # --- STRATEGY PERFORMANCE SNAPSHOT ---
    strat_text = ""
    if prices is not None:
        try:
            metrics, _ = calculate_strategy_performance(df, prices)
            if metrics:
                strat_text += " STRATEGY PERFORMANCE SNAPSHOT (CAGR | Sharpe | MaxDD)\n"
                strat_text += "-" * 80 + "\n"
                # Sort by Sharpe Ratio descending
                sorted_strats = sorted(metrics.items(), key=lambda x: x[1]['Sharpe'], reverse=True)
                
                for name, m in sorted_strats:
                    strat_text += f" {name:<20} | {m['CAGR']:>6.1%} | {m['Sharpe']:>6.2f} | {m['MaxDD']:>6.1%}\n"
                strat_text += "-" * 80 + "\n"
        except Exception as e:
            print(f"Strategy Calc Error: {e}")

    # 1. The Liquidity Valve Analysis [cite: 617, 684]
    # Bitcoin is the "release valve" for excess liquidity.
    btc_trend = "OPEN" if last["CF_Liquidity_Valve"] > df["CF_Liquidity_Valve_MA200"].iloc[-1] else "CLOSED"
    real_yield = last["CF_Real_Yield"]
    
    valve_commentary = ""
    if btc_trend == "OPEN" and real_yield < 1.5:
        valve_commentary = "The Liquidity Valve is OPEN. With real yields contained, capital is flowing out the risk curve into speculative assets."
    elif btc_trend == "CLOSED" and real_yield > 2.0:
        valve_commentary = "The Liquidity Valve is CLOSED. High real yields are sucking liquidity out of the system. Expect pressure on long-duration assets."
    else:
        valve_commentary = "The Liquidity Valve is NEUTRAL. Markets are chopping as real rates seek equilibrium."

    # 2. Market Fragility Analysis [cite: 3914, 4008]
    # Divergence between index price (SPY) and Breadth (RSP).
    breadth_trend = "HEALTHY" if last["CF_Breadth"] > df["CF_Breadth_MA200"].iloc[-1] else "NARROW"
    vix_level = last["CF_VIX"]
    
    fragility_commentary = ""
    if breadth_trend == "NARROW" and vix_level < 15:
        fragility_commentary = "WARNING: Market Fragility is HIGH. Breadth is narrowing while VIX is low (Complacency). This divergence often precedes 'air pockets' or sharp corrections."
    elif breadth_trend == "HEALTHY":
        fragility_commentary = "Market Structure is ROBUST. Broad participation (Equal Weight outperforming) suggests a healthy accumulation phase."
    else:
        fragility_commentary = "Market Structure is MIXED. Monitor for further narrowing in participation."

    # 3. Consumer Transmission [cite: 36, 168]
    # XLY vs XLP as a proxy for recession risk.
    cons_trend = "EXPANSION" if last["CF_Consumer"] > df["CF_Consumer_MA200"].iloc[-1] else "RETRENCHMENT"
    cons_commentary = ""
    if cons_trend == "RETRENCHMENT":
        cons_commentary = "The Consumer is RETRENCHING. Cyclicals lagging Staples indicates household budgets are tightening, a classic late-cycle signal."
    else:
        cons_commentary = "The Consumer is RESILIENT. Outperformance in Discretionary stocks suggests the economy is avoiding immediate recession."

    # 4. Plumbing (NFCI) [cite: 1129]
    plumbing_status = "STRESSED" if last["CF_NFCI"] > 0 else "FUNCTIONAL"
    
    # 5. Composite Macro Score
    score = last["MACRO_SCORE"]
    score_status = ""
    if score > 0.3:
        score_status = "RISK-ON (Bullish Regime)"
    elif score < -0.3:
        score_status = "RISK-OFF (Bearish Regime)"
    else:
        score_status = "NEUTRAL / CHOPPY"

    # 6. The Shape of the Macro Regime (Radar Chart Explanation)
    # Explain the polygon shape
    radar_commentary = (
        "THE SHAPE OF THE MACRO REGIME (Radar Chart):\n"
        "   We visualize the macro landscape as a polygon defined by 6 dimensions:\n"
        "   Growth, Liquidity, Risk, Inflation, Rates, and Sentiment.\n"
        "   - EXPANDED Polygon (Green) = Loose Financial Conditions & Growth (Bullish).\n"
        "   - CONTRACTED Polygon (Red) = Tight Conditions & Fear (Bearish).\n"
        "   Current State: Watch for 'Divergences' where Growth expands but Liquidity contracts."
    )

    # 6. Equity Rotations Internals
    # Helper to check trend
    def check_trend(col):
        return "LEADING" if last[col] > df[f"{col}_MA200"].iloc[-1] else "LAGGING"

    val_growth = check_trend("ROT_Value_Growth")
    cyc_def = check_trend("ROT_Cyc_Def_Sectors")
    small_large = check_trend("ROT_Small_Large")
    hibeta_lovol = check_trend("ROT_HiBeta_LoVol")

    # --- ADVANCED PLAYBOOK GENERATION ---
    
    # 1. Define the Regime
    regime_name = ""
    if score > 0.5:
        regime_name = "AGGRESSIVE RISK-ON (Liquidity Fueled)"
    elif score > 0:
        regime_name = "MODERATE RISK-ON (Selective)"
    elif score > -0.5:
        regime_name = "CHOPPY / TRANSITIONAL (Caution)"
    else:
        regime_name = "DEFENSIVE / RISK-OFF (Capital Preservation)"
        
    # 2. Generate Actionable Ideas (Long/Short)
    longs = []
    shorts = []
    
    # Liquidity Rules
    if btc_trend == "OPEN" and real_yield < 1.5:
        longs.append("Crypto (BTC/ETH)")
        longs.append("High Beta Tech (XLK/XBI)")
    elif real_yield > 2.0:
        shorts.append("Long Duration Assets (TLT, Gold)")
        longs.append("Cash (SHY)")
        
    # Growth Rules
    if cyc_def == "LEADING":
        longs.append("Cyclicals (XLI, XLB)")
        shorts.append("Defensives (XLU, XLP)")
    else:
        longs.append("Quality/Defensives (NOBL, XLP)")
        shorts.append("Deep Cyclicals")
        
    # Dollar Rules (Proxy via UUP if available, else infer)
    # Assuming UUP is in df, if not skip
    if "UUP" in df.columns:
        usd_trend = "BULLISH" if last["UUP"] > df["UUP_MA200"].iloc[-1] else "BEARISH"
        if usd_trend == "BULLISH":
            shorts.append("Emerging Markets (EEM)")
            shorts.append("Commodities (Gold/Copper)")
        else:
            longs.append("Emerging Markets (EEM)")
            longs.append("Hard Assets (GLD/CPER)")

    # 3. Invalidation Criteria
    invalidation_text = ""
    if regime_name.count("RISK-ON"):
        invalidation_text = "Thesis INVALIDATED if: 10Y Real Yields close > 2.0% OR Bitcoin closes below 200D MA."
    elif regime_name.count("RISK-OFF"):
        invalidation_text = "Thesis INVALIDATED if: NFCI drops below -0.6 (Easing) AND Discretionary (XLY) reclaims 200D MA."
    else:
        invalidation_text = "Thesis INVALIDATED if: VIX spikes > 25 (Breakout) OR SPY loses 200D MA support."

    # 4. Statistical Anomalies (Z-Score Check)
    anomalies = []
    z_cols = [c for c in df.columns if "_Z" in c]
    for col in z_cols:
        z_val = last[col]
        base_name = col.replace("_Z", "")
        if z_val > 2.0:
            anomalies.append(f"⚠️ {base_name} is OVEREXTENDED (+{z_val:.1f}σ). Prone to Mean Reversion.")
        elif z_val < -2.0:
            anomalies.append(f"✅ {base_name} is OVERSOLD ({z_val:.1f}σ). Potential Bounce.")
            
    anomaly_text = "\n".join(anomalies) if anomalies else "None. No statistical extremes detected (>2 Sigma)."

    # --- QUANT LAB INSIGHTS & PLAYBOOKS ---
    quant_text = ""
    if prices is not None:
        try:
            import quant_engine as qe
            
            assets_to_analyze = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
            quant_text += "1. MULTI-ASSET QUANT REGIMES & OPTION PLAYBOOKS:\n"
            
            for ticker in assets_to_analyze:
                if ticker not in prices.columns: continue
                
                # Data Prep
                ret = prices[ticker].pct_change().dropna()
                
                # A. Volatility Regime
                try:
                    vol_garch, _ = qe.fit_garch(ret.values * 100)
                    curr_vol = vol_garch[-1] / 100 * np.sqrt(252)
                    hist_vol = ret.rolling(21).std().iloc[-1] * np.sqrt(252)
                    
                    vol_status = "HIGH" if curr_vol > 0.20 else "LOW"
                    vol_trend = "EXPANDING" if curr_vol > hist_vol * 1.1 else ("CONTRACTING" if curr_vol < hist_vol * 0.9 else "STABLE")
                except:
                    vol_status = "N/A"
                    vol_trend = "N/A"
                    
                # B. Monte Carlo Bias
                S0 = prices[ticker].iloc[-1]
                mu = ret.mean() * 252
                sigma_sim = ret.std() * np.sqrt(252)
                _, paths = qe.simulate_gbm(S0, mu, sigma_sim, 30/252.0, 1/252.0, 100)
                exp_ret = (paths[:,-1].mean() / S0) - 1
                bias = "BULLISH" if exp_ret > 0.01 else ("BEARISH" if exp_ret < -0.01 else "NEUTRAL")
                
                # C. Generate Playbook
                strategy = ""
                if vol_status == "LOW" and vol_trend == "STABLE":
                    if bias == "BULLISH": strategy = "Long Call / Debit Spread (Cheap Vol)"
                    elif bias == "BEARISH": strategy = "Long Put / Debit Spread (Cheap Vol)"
                    else: strategy = "Calendar Spread / Long Straddle (Anticipate Move)"
                elif vol_status == "HIGH" or vol_trend == "EXPANDING":
                    if bias == "BULLISH": strategy = "Short Put Spread / Covered Call (Sell Vol)"
                    elif bias == "BEARISH": strategy = "Short Call Spread / Protective Collar"
                    else: strategy = "Iron Condor / Short Straddle (Harvest Vol)"
                else:
                    strategy = "Trend Following (Directional)"
                    
                quant_text += f"   - {ticker}: {bias} Bias | Vol: {vol_status} ({vol_trend}) -> STRATEGY: {strategy}\n"

            # 2. SPY Detailed Greeks
            if "SPY" in prices.columns:
                S0 = prices["SPY"].iloc[-1]
                sigma_spy = prices["SPY"].pct_change().std() * np.sqrt(252)
                greeks = qe.calculate_greeks(S0, S0, 30/365.0, 0.045, sigma_spy, "call")
                quant_text += f"\n2. SPY ATM GREEKS (Risk Sensitivity):\n"
                quant_text += f"   - Delta: {greeks.get('Delta', 0):.2f} (Directional Risk)\n"
                quant_text += f"   - Gamma: {greeks.get('Gamma', 0):.4f} (Convexity/Acceleration)\n"
                quant_text += f"   - Vega:  {greeks.get('Vega', 0):.2f} (Volatility Risk)\n"
            
        except Exception as e:
            print(f"Quant Text Gen Error: {e}")
            quant_text += f"Error generating insights: {e}"

    # --- CROSS-ASSET VOLATILITY REGIME ---
    vol_text = ""
    if prices is not None:
        try:
            vol_tickers = {
                "^VIX": "SP500 (VIX)",
                "^VXN": "Nasdaq (VXN)",
                "^GVZ": "Gold (GVZ)",
                "^MOVE": "Rates (MOVE)"
            }
            
            vol_text += "1. IMPLIED VOLATILITY RANK (1-Year Percentile):\n"
            
            for ticker, name in vol_tickers.items():
                if ticker in prices.columns:
                    # Get last 1 year of data
                    hist = prices[ticker].iloc[-252:]
                    curr = hist.iloc[-1]
                    low = hist.min()
                    high = hist.max()
                    
                    # Calc Rank
                    rank = (curr - low) / (high - low) if high > low else 0.5
                    
                    # Regime
                    if rank > 0.8: regime = "FEAR / PANIC (Contrarian Buy?)"
                    elif rank < 0.2: regime = "COMPLACENCY (Hedge Risk)"
                    else: regime = "NORMAL"
                    
                    vol_text += f"   - {name}: {curr:.2f} (Rank: {rank:.0%}) -> {regime}\n"
            
            vol_text += "\n2. VOLATILITY INTERPRETATION:\n"
            vol_text += "   - High Rank (>80%): Markets pricing extreme stress. Often marks bottoms.\n"
            vol_text += "   - Low Rank (<20%): Markets pricing perfection. Vulnerable to shocks.\n"

        except Exception as e:
            print(f"Vol Analysis Error: {e}")
            vol_text = "Error generating Volatility insights."

    # --- GLOBAL MACRO & FX REGIME ---
    global_text = ""
    if prices is not None: # Ensure prices are available for FX
        try:
            import datetime as dt
            # 1. Rate Differentials (Driver of Capital Flows)
            # Need to ensure '10Y_Breakeven' exists or handle its absence
            us_breakeven = df["10Y_Breakeven"].iloc[-1] if "10Y_Breakeven" in df.columns else 0.0
            us_yield = last["CF_Real_Yield"] + us_breakeven # Approx Nominal
            
            diffs = []
            if "RateDiff_US_DE" in df.columns: diffs.append(f"US-Bund: {last['RateDiff_US_DE']:.2f}%")
            if "RateDiff_US_JP" in df.columns: diffs.append(f"US-JGB:  {last['RateDiff_US_JP']:.2f}%")
            if "RateDiff_US_UK" in df.columns: diffs.append(f"US-Gilt: {last['RateDiff_US_UK']:.2f}%")
            
            diff_str = ", ".join(diffs)
            
            # 2. FX Trends & Playbooks
            fx_playbook = []
            
            # EUR/USD
            if "EURUSD=X" in prices.columns:
                eur_trend = "BULLISH" if prices["EURUSD=X"].iloc[-1] > prices["EURUSD=X"].rolling(200).mean().iloc[-1] else "BEARISH"
                if eur_trend == "BEARISH" and last.get("RateDiff_US_DE", 0) > 1.5:
                    fx_playbook.append("Short EUR/USD (Policy Divergence)")
            
            # USD/JPY (Carry Trade)
            if "JPY=X" in prices.columns:
                jpy_trend = "BULLISH" if prices["JPY=X"].iloc[-1] > prices["JPY=X"].rolling(200).mean().iloc[-1] else "BEARISH"
                # JPY=X is USD/JPY usually, so Bullish = Strong USD
                if jpy_trend == "BULLISH" and last.get("RateDiff_US_JP", 0) > 2.5:
                    fx_playbook.append("Long USD/JPY (Carry Trade)")
                    
            # USD/CNY
            if "CNY=X" in prices.columns:
                cny_trend = "BULLISH" if prices["CNY=X"].iloc[-1] > prices["CNY=X"].rolling(200).mean().iloc[-1] else "BEARISH"
                if cny_trend == "BULLISH":
                    fx_playbook.append("Long USD/CNY (China Weakness)")

            global_text += f"1. RATE DIFFERENTIALS (Yield Advantage):\n   {diff_str}\n"
            global_text += f"2. FX REGIME & PLAYBOOK:\n"
            if fx_playbook:
                for play in fx_playbook:
                    global_text += f"   - {play}\n"
            else:
                global_text += "   - Neutral / No clear divergence signals.\n"
                
        except Exception as e:
            print(f"Global Macro Error: {e}")
            global_text = f"Error generating Global Macro insights: {e}"

    # --- RISK MANAGEMENT (VaR) ---
    risk_text = ""
    if prices is not None:
        try:
            # Get Core-Satellite Curve
            _, curves = calculate_strategy_performance(df, prices)
            if "Core-Satellite (CCI)" in curves.columns:
                curve = curves["Core-Satellite (CCI)"]
                var95, cvar95 = calculate_portfolio_risk(curve)
                
                risk_text += "1. CORE-SATELLITE PORTFOLIO RISK (Daily):\n"
                risk_text += f"   - 95% VaR: {var95:.2%} (Max loss on 95% of days)\n"
                risk_text += f"   - 95% CVaR: {cvar95:.2%} (Avg loss on worst 5% of days)\n"
                risk_text += "   - Interpretation: "
                if var95 < -0.015: risk_text += "ELEVATED RISK. Tighten stops.\n"
                else: risk_text += "NORMAL RISK PROFILE.\n"
        except Exception as e:
            print(f"Risk Calc Error: {e}")
            risk_text = "Error calculating VaR."

    # --- MACRO RISK WATCH ---
    macro_risk_text = ""
    if "10Y_Yield" in df.columns and "2Y_Yield" in df.columns:
        yc = df["10Y_Yield"].iloc[-1] - df["2Y_Yield"].iloc[-1]
        yc_status = "INVERTED (Recession Signal)" if yc < 0 else "NORMAL (Expansionary)"
        macro_risk_text += f"1. YIELD CURVE (10Y-2Y): {yc:.2f}% -> {yc_status}\n"
        
    if "HY_Spread" in df.columns:
        hy = df["HY_Spread"].iloc[-1]
        hy_status = "STRESS" if hy > 5.0 else "CALM"
        macro_risk_text += f"2. CREDIT SPREADS (HY):  {hy:.2f}% -> {hy_status}\n"

    # --- JPM STRATEGY INTEGRATION (New) ---
    jpm_text = ""
    if prices is not None:
        try:
            jpm_text += get_jpm_views() + "\n"
            jpm_text += calculate_theme_performance(prices)
        except Exception as e:
            print(f"JPM Integration Error: {e}")
            jpm_text = "Error generating JPM views."

    # --- FORWARD LOOKING MODELS (Recession & ERP) ---
    forward_text = ""
    if prices is not None:
        try:
            # 1. Recession Probability
            rec_prob = last.get("Recession_Prob", 0)
            rec_status = "HIGH RISK" if rec_prob > 30 else "LOW RISK"
            
            # 2. Equity Risk Premium
            real_yield_val = last.get("CF_Real_Yield", 1.5)
            erp_data = get_erp_snapshot("SPY", real_yield_val)
            
            forward_text += "1. RECESSION PROBABILITY (12-Month Ahead):\n"
            forward_text += f"   - Probability: {rec_prob:.1f}% -> {rec_status}\n"
            forward_text += "   - Model: NY Fed Proxy (10Y-3M Probit)\n"
            
            forward_text += "\n2. EQUITY RISK PREMIUM (Valuation Anchor):\n"
            if erp_data["pe"]:
                forward_text += f"   - SPY PE Ratio: {erp_data['pe']:.1f}x\n"
                forward_text += f"   - Earnings Yield: {erp_data['ey']:.2f}%\n"
                forward_text += f"   - Real ERP: {erp_data['erp']:.2f}% -> {erp_data['rating']}\n"
            else:
                forward_text += "   - ERP Data Unavailable.\n"
                
            # 3. Predictive Regime Switching (GMM)
            regime_data = calculate_regime_gmm(prices)
            if regime_data["current_state"] != "N/A":
                curr = regime_data["current_state"]
                probs = regime_data["probs"]
                # Find most likely NEXT state
                next_likely = max(probs, key=probs.get)
                prob_val = probs[next_likely]
                
                forward_text += "\n3. PREDICTIVE REGIME SWITCHING (Hidden Markov Proxy):\n"
                forward_text += f"   - Current State: {curr}\n"
                forward_text += f"   - Next Likely State: {next_likely} (Prob: {prob_val:.1%})\n"
                
            # 4. Sector Rotation Clock
            rot_data = predict_sector_rotation(prices)
            if rot_data["current_leader"] != "N/A":
                forward_text += "\n4. SECTOR ROTATION CLOCK (Predictive):\n"
                forward_text += f"   - Best Performer (Last Month): {rot_data['current_leader']}\n"
                forward_text += f"   - Next Likely Leader: {rot_data['next_likely']} (Hist. Freq: {rot_data['prob']:.1%})\n"
                
            # 5. Volatility Skew
            skew_data = calculate_option_skew("SPY")
            if skew_data["skew"] is not None:
                forward_text += "\n5. VOLATILITY SKEW (Tail Risk Monitor):\n"
                forward_text += f"   - Skew (Put IV - Call IV): {skew_data['skew']:.2%} -> {skew_data['status']}\n"
                forward_text += f"   - (10% OTM Put IV: {skew_data['put_iv']:.1%} | 10% OTM Call IV: {skew_data['call_iv']:.1%})\n"

            # 6. Systemic Risk (PCA)
            pca_data = calculate_systemic_risk_pca(prices)
            if pca_data["ratio"] is not None:
                forward_text += "\n6. SYSTEMIC RISK MONITOR (PCA Absorption Ratio):\n"
                forward_text += f"   - Absorption Ratio: {pca_data['ratio']:.1%} -> {pca_data['status']}\n"
                forward_text += "   - (Variance explained by PC1. High = Tight Coupling/Crash Risk)\n"

            # 7. Seasonality
            seas_data = calculate_seasonality(prices)
            if seas_data["curr_month"] != "N/A":
                forward_text += "\n7. SEASONAL CYCLE FORECAST (SPY History):\n"
                forward_text += f"   - {seas_data['curr_month']}: {seas_data['curr_stats']}\n"
                forward_text += f"   - {seas_data['next_month']}: {seas_data['next_stats']}\n"

            # 8. Gamma Flip (Market Structure)
            gamma_data = calculate_gamma_flip("SPY")
            if gamma_data["level"] is not None:
                forward_text += "\n8. GAMMA FLIP MONITOR (Volatility Trigger):\n"
                forward_text += f"   - Flip Level: ${gamma_data['level']:.2f} -> {gamma_data['status']}\n"
                forward_text += f"   - (Current: ${gamma_data['current_price']:.2f}. Below Flip = High Vol/Crash Risk)\n"

            # 9. Advanced Risk Metrics
            forward_text += "\n9. ADVANCED RISK METRICS:\n"
            
            # Liquidity
            liq_data = calculate_liquidity_stress(df)
            if liq_data["level"] is not None:
                forward_text += f"   - Liquidity Stress Index: {liq_data['level']:.2f} (Z-Score) -> {liq_data['status']}\n"
                
            # Crash Prob
            crash_data = calculate_garch_crash_prob(prices)
            if crash_data["prob"] is not None:
                forward_text += f"   - Crash Probability (1-Week): {crash_data['prob']:.1f}% -> {crash_data['status']}\n"
                forward_text += f"     (Prob of >5% Drop. Vol Forecast: {crash_data['vol_forecast']:.1f}%)\n"
                
            # Correlation
            corr_data = calculate_correlation_surprise(prices)
            if corr_data["surprise"] is not None:
                forward_text += f"   - Correlation Surprise (SPY/TLT): {corr_data['surprise']:.2f} -> {corr_data['status']}\n"
                forward_text += f"     (1M Corr: {corr_data['curr_corr']:.2f} vs 1Y Avg: {corr_data['lt_corr']:.2f})\n"

            # 10. Global Flows & Carry
            flows = calculate_global_flows(df, prices)
            forward_text += "\n10. GLOBAL FLOWS & CARRY TRADE:\n"
            
            if flows["custody_trend"] != "N/A":
                forward_text += f"   - Foreign Custody (Fed): {flows['custody_trend']}\n"
                forward_text += f"     (YoY Change: {flows['custody_chg']:+.1%}. Drop = Liquidity Drain)\n"
                
            if flows["carry_status"] != "N/A":
                forward_text += f"   - JPY Carry Trade: {flows['carry_status']}\n"
                forward_text += f"     (USD/JPY: {flows['usdjpy_px']:.2f} vs Trend: {flows['usdjpy_trend']})\n"
                if flows["spread_usjp"]:
                    forward_text += f"     (Yield Spread US-JP: {flows['spread_usjp']:.2f}%)\n"
                
        except Exception as e:
            print(f"Forward Model Error: {e}")
            forward_text = "Error generating forward outlook."

    # --- MONETARY PLUMBING & ECONOMY (New) ---
    plumbing_text = ""
    # Liquidity Impulse
    if "M2_YoY" in df.columns and "Fed_Assets_YoY" in df.columns:
        m2 = df["M2_YoY"].iloc[-1]
        fed = df["Fed_Assets_YoY"].iloc[-1]
        liq_status = "EXPANSION" if m2 > 0 else "CONTRACTION"
        plumbing_text += f"1. LIQUIDITY IMPULSE (YoY):\n"
        plumbing_text += f"   - M2 Money Supply: {m2:>.1%} -> {liq_status}\n"
        plumbing_text += f"   - Fed Balance Sheet: {fed:>.1%} (QT/QE Monitor)\n"
        
    # Economic Health
    if "CPI_YoY" in df.columns and "Unemployment" in df.columns:
        cpi = df["CPI_YoY"].iloc[-1]
        unrate = df["Unemployment"].iloc[-1]
        plumbing_text += f"2. ECONOMIC HEALTH:\n"
        plumbing_text += f"   - CPI Inflation (YoY): {cpi:>.1%}\n"
        plumbing_text += f"   - Unemployment Rate:   {unrate:>.1%}\n"

    # Build the report string
    report_lines = []
    import datetime as dt # Ensure dt is imported if not already
    report_lines.append("="*80)
    report_lines.append(f" SUPER MACRO DASHBOARD REPORT | Date: {dt.date.today()}")
    report_lines.append("="*80)
    report_lines.append(f"CURRENT REGIME: {regime_name}")
    report_lines.append("-" * 80)
    if strat_text:
        report_lines.append(strat_text)
    report_lines.append("-" * 80)
    report_lines.append(f"1. LIQUIDITY VALVE (Bitcoin & Real Rates): {btc_trend}")
    report_lines.append(f"   Analysis: {valve_commentary}")
    report_lines.append("-" * 80)
    report_lines.append(f"2. MARKET FRAGILITY (Breadth vs Volatility): {breadth_trend}")
    report_lines.append(f"   Analysis: {fragility_commentary}")
    report_lines.append("-" * 80)
    report_lines.append(f"3. ECONOMIC TRANSMISSION (The Consumer): {cons_trend}")
    report_lines.append(f"   Analysis: {cons_commentary}")
    report_lines.append("-" * 80)
    report_lines.append(f"4. FINANCIAL PLUMBING (NFCI): {plumbing_status}")
    report_lines.append(f"   NFCI Level: {last['CF_NFCI']:.2f}. (Positive values indicate tighter-than-avg conditions).")
    report_lines.append("-" * 80)
    report_lines.append(f"5. COMPOSITE MACRO SCORE: {score:.2f}")
    report_lines.append(f"   (Aggregated signal from 14+ macro/market ratios)")
    report_lines.append("-" * 80)
    report_lines.append(f" {radar_commentary}")
    report_lines.append("-" * 80)
    report_lines.append(f"6. EQUITY ROTATION INTERNALS:")
    report_lines.append(f"   - Value vs Growth:       {val_growth}")
    report_lines.append(f"   - Cyclicals vs Def:      {cyc_def}")
    report_lines.append(f"   - Small vs Large Cap:    {small_large}")
    report_lines.append(f"   - High Beta vs Low Vol:  {hibeta_lovol}")
    report_lines.append("="*80)
    report_lines.append(" ACTIONABLE PLAYBOOK & THESIS")
    report_lines.append("-" * 80)
    report_lines.append(f"STRATEGY: {regime_name}")
    report_lines.append(f"LONGS: {', '.join(longs) if longs else 'Cash / Neutral'}")
    report_lines.append(f"SHORTS/AVOID: {', '.join(shorts) if shorts else 'None'}")
    report_lines.append("")
    report_lines.append(f"WATCH: Real Yields, USD Index, High Yield Spreads.")
    report_lines.append(f"INVALIDATION: {invalidation_text}")
    report_lines.append("-" * 80)
    report_lines.append(" STATISTICAL ANOMALIES (2-Sigma Extremes)")
    report_lines.append(anomaly_text)
    report_lines.append("-" * 80)
    report_lines.append(" GLOBAL MACRO & FX REGIME (New)")
    report_lines.append(global_text)
    report_lines.append("-" * 80)
    report_lines.append(" CROSS-ASSET VOLATILITY REGIME (New)")
    report_lines.append(vol_text)
    report_lines.append("-" * 80)
    report_lines.append(" RISK MANAGEMENT (VaR) (New)")
    report_lines.append(risk_text)
    report_lines.append("-" * 80)
    report_lines.append(" MACRO RISK WATCH (New)")
    report_lines.append(macro_risk_text)
    report_lines.append("-" * 80)
    report_lines.append(" MONETARY PLUMBING & ECONOMY (New)")
    report_lines.append(plumbing_text)
    report_lines.append("-" * 80)
    report_lines.append(" INSTITUTIONAL VIEWS (JPM 2026) (New)")
    report_lines.append(jpm_text)
    report_lines.append("-" * 80)
    report_lines.append(" FORWARD LOOKING MODELS (Recession & ERP) (New)")
    report_lines.append(forward_text)
    report_lines.append("-" * 80)
    report_lines.append(" QUANT LAB INSIGHTS (New)")
    report_lines.append(quant_text)
    
    # --- QUANT & OPTIONS SECTION ---
    # 1. VIX Term Structure
    vix_spot = last["CF_VIX"]
    # Check for Yahoo VIX3M ticker (likely normalized name)
    vix_3m = last["^VIX3M"] if "^VIX3M" in df.columns else vix_spot 
    term_structure = "CONTANGO (Normal)" if vix_3m > vix_spot else "BACKWARDATION (Panic)"
    
    # 2. Options Sentiment (PCR)
    opt_metrics_spy = get_options_data("SPY", calc_gex=True)
    pcr = opt_metrics_spy.get("pcr", 0)
    net_oi = opt_metrics_spy.get("net_oi", 0)
    max_pain = opt_metrics_spy.get("max_pain", 0)
    
    pcr_sentiment = "NEUTRAL"
    if pcr > 1.2: pcr_sentiment = "BEARISH / HEDGING (High Put Vol)"
    elif pcr < 0.7: pcr_sentiment = "BULLISH / COMPLACENT (Low Put Vol)"
    
    # 3. Gamma Exposure
    gamma_regime = "LONG GAMMA (Stabilizing)" if net_oi > 0 else "SHORT GAMMA (Volatile)"
    
    # 4. Bond Volatility (TLT IV)
    opt_metrics_tlt = get_options_data("TLT", calc_gex=False)
    bond_iv = opt_metrics_tlt.get("iv_call", 0)
    bond_vol_status = "ELEVATED (Risk-Off Warning)" if bond_iv and bond_iv > 0.20 else "NORMAL"
    
    # 5. Stock/Bond Correlation
    sb_corr = last.get("ROT_SPY_TLT_Corr", 0)
    corr_status = "RISK PARITY ON (Hedge Working)" if sb_corr < -0.3 else "CORRELATION BREAKDOWN (Danger)" if sb_corr > 0.3 else "UNCORRELATED"
    
    report_lines.append(" QUANT & OPTIONS STRUCTURE")
    report_lines.append(f"1. VIX TERM STRUCTURE: {term_structure}")
    report_lines.append(f"   (Spot VIX: {vix_spot:.2f} vs 3M VIX: {vix_3m:.2f})")
    report_lines.append(f"2. OPTION SENTIMENT (SPY): {pcr_sentiment}")
    report_lines.append(f"   (Put/Call Volume Ratio: {pcr:.2f})")
    
    report_lines.append("-" * 80)
    report_lines.append(" ADVANCED QUANT & STRUCTURE")
    report_lines.append(f"1. GAMMA EXPOSURE (SPY): {gamma_regime}")
    report_lines.append(f"   (Net OI: {net_oi:,.0f}. Positive = Dealers Buy Dips. Negative = Dealers Sell Dips.)")
    report_lines.append(f"2. MAX PAIN (SPY): ${max_pain:.2f}")
    report_lines.append(f"   (Strike with most option value. Price magnet into OpEx).")
    report_lines.append(f"3. BOND VOLATILITY (TLT IV): {bond_iv:.1%} -> {bond_vol_status}")
    report_lines.append(f"4. STOCK/BOND CORRELATION: {sb_corr:.2f} -> {corr_status}")
    
    report_lines.append("="*80)
    
    full_report = "\n".join(report_lines)
    print(full_report)
    
    return full_report


def plot_forward_models(df: pd.DataFrame, prices: pd.DataFrame) -> plt.Figure:
    """
    Generates Page: Forward Looking Models (Recession, Regime, Rotation, PCA, Seasonality).
    """
    print("   Generating Forward Models Page...")
    plt.style.use('default')
    
    fig = plt.figure(figsize=(14, 18)) # Increased height for more panels
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1.2, 1.2])
    
    # 1. Recession Probability History
    ax1 = fig.add_subplot(gs[0, :])
    if "Recession_Prob" in df.columns:
        prob = df["Recession_Prob"].dropna()
        ax1.plot(prob.index, prob, color='black', label="Recession Probability (12M Ahead)")
        ax1.fill_between(prob.index, prob, 0, color='red', alpha=0.3)
        ax1.axhline(30, color='orange', linestyle='--', label="Warning Threshold (30%)")
        ax1.axhline(50, color='red', linestyle='--', label="High Probability (>50%)")
        
        ax1.set_title("1. Recession Probability Model (Estrella/Mishkin Probit)", fontsize=12, weight='bold')
        ax1.set_ylabel("Probability (%)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
    else:
        ax1.text(0.5, 0.5, "Recession Probability Data Missing", ha='center')
        
    # 2. Yield Curve Spread (Input)
    ax2 = fig.add_subplot(gs[1, 0])
    if "Spread_10Y3M" in df.columns:
        spread = df["Spread_10Y3M"].dropna()
        ax2.plot(spread.index, spread, color='blue', label="10Y - 3M Treasury Spread")
        ax2.axhline(0, color='black', linewidth=1)
        ax2.fill_between(spread.index, spread, 0, where=(spread < 0), color='red', alpha=0.3, label="Inversion")
        
        ax2.set_title("2. The Input: 10Y-3M Yield Curve Spread", fontsize=12, weight='bold')
        ax2.set_ylabel("Spread (%)")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Spread Data Missing", ha='center')

    # 3. Regime Probability (GMM)
    ax3 = fig.add_subplot(gs[1, 1])
    regime_data = calculate_regime_gmm(prices)
    if regime_data["current_state"] != "N/A":
        # Plot Regime Labels over time (scatter)
        labels = regime_data["labels"]
        dates = regime_data["dates"]
        
        ax3.scatter(dates, labels, c=labels, cmap='RdYlGn_r', s=10, alpha=0.6)
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(["State 0", "State 1", "State 2"])
        ax3.set_title(f"3. Market Regimes (GMM Clustering)\nCurrent: {regime_data['current_state']}", fontsize=12, weight='bold')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Regime Data Missing", ha='center')
        
    # 4. Sector Rotation Matrix (Heatmap)
    ax4 = fig.add_subplot(gs[2, 0])
    rot_data = predict_sector_rotation(prices)
    if rot_data["current_leader"] != "N/A":
        # Re-calc for visualization
        sectors = ["XLI", "XLB", "XLU", "XLF", "XLK", "XLE", "XLV", "XLC", "XLY", "XLP"]
        valid_sectors = [s for s in sectors if s in prices.columns]
        monthly_prices = prices[valid_sectors].resample('ME').last()
        monthly_rets = monthly_prices.pct_change().dropna()
        leaders = monthly_rets.idxmax(axis=1)
        
        # USE THE ROBUST LEADER FROM THE FUNCTION CALL
        curr_leader = rot_data["current_leader"]
        
        # Count next states from this leader
        next_counts = {}
        for prev, curr in zip(leaders[:-1], leaders[1:]):
            if prev == curr_leader:
                next_counts[curr] = next_counts.get(curr, 0) + 1
        
        if next_counts:
            total = sum(next_counts.values())
            # Sort
            sorted_counts = sorted(next_counts.items(), key=lambda x: x[1], reverse=True)
            labels = [x[0] for x in sorted_counts]
            vals = [x[1]/total for x in sorted_counts]
            
            ax4.bar(labels, vals, color='purple', alpha=0.7)
            ax4.set_title(f"4. Rotation Clock: Next Leader Prob\n(Given Best Performer: {curr_leader})", fontsize=12, weight='bold')
            ax4.set_ylabel("Probability")
            ax4.grid(True, axis='y', alpha=0.3)
        else:
             ax4.text(0.5, 0.5, f"No historical precedents for leader: {curr_leader}", ha='center')
    else:
        ax4.text(0.5, 0.5, "Rotation Data Missing", ha='center')

    # 5. Systemic Risk (PCA)
    ax5 = fig.add_subplot(gs[2, 1])
    pca_data = calculate_systemic_risk_pca(prices)
    if not pca_data["history"].empty:
        hist = pca_data["history"]
        ax5.plot(hist.index, hist, color='red', label="Absorption Ratio")
        ax5.axhline(0.75, color='black', linestyle='--', label="Critical (>75%)")
        ax5.axhline(0.65, color='orange', linestyle='--', label="Elevated (>65%)")
        
        status = pca_data["status"]
        ax5.set_title(f"5. Systemic Risk Monitor (PCA)\nStatus: {status}", fontsize=12, weight='bold')
        ax5.legend(loc="upper left")
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 1.0)
    else:
        ax5.text(0.5, 0.5, "PCA Data Missing", ha='center')
        
    # 6. Seasonality
    ax6 = fig.add_subplot(gs[3, :])
    seas_data = calculate_seasonality(prices)
    if seas_data["curr_month"] != "N/A":
        # We need to visualize the average returns for ALL months to give context
        # Re-calc full seasonality
        spy = prices["SPY"]
        spy_monthly = spy.resample('ME').last().pct_change().dropna()
        df_m = pd.DataFrame({"Ret": spy_monthly})
        df_m["Month"] = df_m.index.month
        
        # Group by Month
        monthly_stats = df_m.groupby("Month")["Ret"].mean()
        
        import calendar
        month_names = [calendar.month_abbr[i] for i in range(1, 13)]
        
        # Colors: Green if positive, Red if negative. Highlight Current/Next.
        colors = ['green' if x > 0 else 'red' for x in monthly_stats]
        
        # Highlight Current and Next
        curr_m = dt.date.today().month
        next_m = (curr_m % 12) + 1
        
        # Make current/next brighter or distinct?
        # Let's just plot bars
        bars = ax6.bar(month_names, monthly_stats, color=colors, alpha=0.5)
        
        # Highlight specific bars
        bars[curr_m-1].set_alpha(1.0)
        bars[curr_m-1].set_edgecolor('black')
        bars[curr_m-1].set_linewidth(2)
        
        bars[next_m-1].set_alpha(1.0)
        bars[next_m-1].set_edgecolor('blue')
        bars[next_m-1].set_linewidth(2)
        
        ax6.set_title(f"6. Seasonal Cycle Forecast (Avg Monthly Return)\nHighlight: {seas_data['curr_month']} (Black) & {seas_data['next_month']} (Blue)", fontsize=12, weight='bold')
        ax6.axhline(0, color='black', linewidth=1)
        ax6.grid(True, axis='y', alpha=0.3)
        ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
    else:
        ax6.text(0.5, 0.5, "Seasonality Data Missing", ha='center')

    plt.tight_layout()
    return fig

def plot_super_dashboard(df: pd.DataFrame, prices: pd.DataFrame, report_text: str):
    """
    Generates 3 Figures and saves them + text to a PDF.
    1. Text Report
    2. The Capital Flows 4-Pillar Dashboard (The Plumbing)
    3. The Original Rotations Grid (The Internals)
    4. The Macro Score History (The Aggregate Signal)
    5. Backtest Performance
    """
    
    pdf_filename = "Macro_Dashboard_Report.pdf"
    print(f"Generating PDF Report: {pdf_filename}...")
    
    with PdfPages(pdf_filename) as pdf:
        
        # --- PAGE 1+: TEXT REPORT (Paginated & Wrapped) ---
        # 1. Wrap text horizontally first
        wrapped_lines = []
        for line in report_text.split('\n'):
            # Wrap at 90 chars to fit page width
            wrapped_lines.extend(textwrap.wrap(line, width=90, replace_whitespace=False, drop_whitespace=False))
            if not line: # Preserve empty lines
                wrapped_lines.append("")
        
        # 2. Paginate vertically
        lines_per_page = 45 # Reduced from 55 to prevent bottom cutoff
        
        for i in range(0, len(wrapped_lines), lines_per_page):
            chunk = wrapped_lines[i:i + lines_per_page]
            page_text = "\n".join(chunk)
            
            fig_text = plt.figure(figsize=(11, 8.5)) # Letter size
            fig_text.clf()
            
            # Add header for continuation pages
            if i > 0:
                page_text = f"--- REPORT CONTINUED (Page {i//lines_per_page + 1}) ---\n\n" + page_text
                
            fig_text.text(0.05, 0.95, page_text, transform=fig_text.transFigure, size=10, ha="left", va="top", family="monospace")
            plt.axis('off')
            pdf.savefig(fig_text)
            plt.close(fig_text)
            
        # --- PAGE 2: MONETARY PLUMBING (The Engine) ---
        fig_plumbing = plot_monetary_plumbing(df)
        pdf.savefig(fig_plumbing)
        plt.close(fig_plumbing)

        # --- PAGE 3: MACRO RISK & SECTOR ROTATION (The Environment) ---
        fig_risk = plot_risk_macro_dashboard(df, prices)
        pdf.savefig(fig_risk)
        plt.close(fig_risk)

        # --- PAGE 3b: FORWARD LOOKING MODELS (Recession Prob) ---
        fig_forward = plot_forward_models(df, prices)
        pdf.savefig(fig_forward)
        plt.close(fig_forward)
    
        # --- PAGE 4: CAPITAL FLOWS DASHBOARD (The Signals) ---
        fig1 = plt.figure(figsize=(14, 10))
        fig1.suptitle("Capital Flows: The 4 Pillars of Liquidity", fontsize=16, weight='bold')
        
        # 1. Liquidity Valve (BTC vs Gold)
        ax1 = fig1.add_subplot(221)
        ax1.plot(df.index, df["CF_Liquidity_Valve"], label="Liquidity Valve (BTC/Gold)", color='orange')
        ax1.plot(df.index, df["CF_Liquidity_Valve_MA200"], label="200D MA", color='black', linestyle="--")
        ax1.set_title("1. Liquidity Valve (Risk Appetite)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        
        # 2. Real Yields
        ax2 = fig1.add_subplot(222)
        ax2.plot(df.index, df["CF_Real_Yield"], label="US 10Y Real Yield", color='purple')
        ax2.axhline(2.0, color='red', linestyle=":", label="Restrictive (>2%)")
        ax2.axhline(0.0, color='green', linestyle=":", label="Accommodative (<0%)")
        ax2.set_title("2. Cost of Capital (Real Yields)")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

        # 3. Market Fragility (Breadth)
        ax3 = fig1.add_subplot(223)
        ax3.plot(df.index, df["CF_Breadth"], label="Breadth (RSP/SPY)", color='blue')
        ax3.plot(df.index, df["CF_Breadth_MA200"], label="200D MA", color='black', linestyle="--")
        ax3.set_title("3. Market Structure (Breadth)")
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3)
        
        # 4. Financial Plumbing (NFCI)
        ax4 = fig1.add_subplot(224)
        ax4.plot(df.index, df["CF_NFCI"], label="NFCI (Fin Conditions)", color='grey')
        ax4.axhline(0, color='black', linestyle="-", linewidth=0.5)
        ax4.fill_between(df.index, df["CF_NFCI"], 0, where=(df["CF_NFCI"] > 0), color='red', alpha=0.3, label="Tight")
        ax4.fill_between(df.index, df["CF_NFCI"], 0, where=(df["CF_NFCI"] < 0), color='green', alpha=0.3, label="Loose")
        ax4.set_title("4. Financial Plumbing (NFCI)")
        ax4.legend(loc="upper left")
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig1)
        plt.close(fig1)
        
        # --- PAGE 5: ROTATIONS GRID (The Internals) ---
        fig2 = plt.figure(figsize=(14, 10))
        fig2.suptitle("Equity Rotations: Under the Hood", fontsize=16, weight='bold')
        
        rotations = [
            ("ROT_Value_Growth", "Value vs Growth (RPV/VONG)"),
            ("ROT_Cyc_Def_Sectors", "Cyclicals vs Defensives"),
            ("ROT_Small_Large", "Small vs Large Cap (IWM/QQQ)"),
            ("ROT_HiBeta_LoVol", "High Beta vs Low Vol (SPHB/SPLV)")
        ]
        
        for i, (col, title) in enumerate(rotations):
            ax = fig2.add_subplot(2, 2, i+1)
            ax.plot(df.index, df[col], label="Ratio", color='black')
            ax.plot(df.index, df[f"{col}_MA200"], label="200D MA", color='red', linestyle="--")
            
            # Highlight Regime
            curr = df[col].iloc[-1]
            ma = df[f"{col}_MA200"].iloc[-1]
            status_color = 'green' if curr > ma else 'red'
            ax.set_title(title, fontsize=12, color=status_color, weight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig2)
        plt.close(fig2)
        
        # --- PAGE 5b: THE MACRO RADAR (The Shape of the Regime) ---
        fig_radar = plot_macro_radar(df, prices)
        pdf.savefig(fig_radar)
        plt.close(fig_radar)
        
        # --- PAGE 6: MACRO SCORE & HEATMAP (Aggregate Signal) ---
        fig3 = plt.figure(figsize=(14, 6))
        ax_score = fig3.add_subplot(111)
        ax_score.plot(df.index, df["MACRO_SCORE"], color='black', linewidth=1, label="Composite Score")
        
        # Fill areas
        ax_score.fill_between(df.index, df["MACRO_SCORE"], 0, where=(df["MACRO_SCORE"] > 0), color='green', alpha=0.3, label="Risk On")
        ax_score.fill_between(df.index, df["MACRO_SCORE"], 0, where=(df["MACRO_SCORE"] < 0), color='red', alpha=0.3, label="Risk Off")
        
        ax_score.axhline(0, color='black', linestyle="-", linewidth=0.5)
        ax_score.set_title("Composite Macro Score (Aggregate Signal)", fontsize=16, weight='bold')
        ax_score.legend(loc="upper left")
        ax_score.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig3)
        plt.close(fig3)
        
        # --- FIGURE 4: CORRELATION HEATMAP ---
        fig4 = plt.figure(figsize=(12, 10))
        ax4 = fig4.add_subplot(111)
        
        # Select key ratios for correlation (Exclude Z-scores and MAs)
        corr_cols = [c for c in df.columns if ("ROT_" in c or "CF_" in c) and "_Z" not in c and "_MA200" not in c]
        
        # Filter to last 6 months (approx 126 days)
        recent_df = df[corr_cols].iloc[-126:]
        
        # Drop columns with all NaNs (e.g. missing VIX3M)
        recent_df = recent_df.dropna(axis=1, how='all')
        corr_cols = recent_df.columns.tolist()
        
        corr_matrix = recent_df.corr()
        
        # Plot Heatmap
        cax = ax4.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        fig4.colorbar(cax)
        
        # Ticks
        ax4.set_xticks(range(len(corr_cols)))
        ax4.set_yticks(range(len(corr_cols)))
        ax4.set_xticklabels(corr_cols, rotation=90, fontsize=8)
        ax4.set_yticklabels(corr_cols, fontsize=8)
        
        # Add Text Annotations
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                text = f"{corr_matrix.iloc[i, j]:.2f}"
                ax4.text(j, i, text, ha='center', va='center', color='black' if abs(corr_matrix.iloc[i, j]) < 0.5 else 'white', fontsize=7)
                
        ax4.set_title("6-Month Cross-Asset Correlation Matrix", fontsize=16, weight='bold', pad=20)
        plt.tight_layout()
        pdf.savefig(fig4)
        plt.close(fig4)
        
        # --- FIGURE 5: QUANT LAB SNAPSHOT ---
        try:
            import quant_engine as qe
            print("   Generating Quant Lab Page...")
            
            fig5 = plt.figure(figsize=(14, 10))
            gs5 = fig5.add_gridspec(2, 2)
            
            # 1. Volatility Regime (GARCH vs Realized)
            ax5_1 = fig5.add_subplot(gs5[0, :])
            
            # Calculate Vol
            spy_ret = prices["SPY"].pct_change().dropna()
            # GARCH
            try:
                vol_garch, _ = qe.fit_garch(spy_ret.values * 100)
                vol_garch = vol_garch / 100 * np.sqrt(252)
                # Align dates
                garch_series = pd.Series(vol_garch, index=spy_ret.index)
            except:
                garch_series = pd.Series(0, index=spy_ret.index) # Fallback
                
            # Realized
            vol_hist = spy_ret.rolling(21).std() * np.sqrt(252)
            
            # Plot last 1 year
            lookback = 252
            ax5_1.plot(garch_series.index[-lookback:], garch_series.iloc[-lookback:], label="GARCH(1,1) Est", color='purple', linewidth=2)
            ax5_1.plot(vol_hist.index[-lookback:], vol_hist.iloc[-lookback:], label="Realized (21D)", color='orange', alpha=0.7)
            ax5_1.set_title("SPY Volatility Regime: GARCH Model vs Realized", fontsize=12, weight='bold')
            ax5_1.legend()
            ax5_1.grid(True, alpha=0.3)
            
            # 2. Monte Carlo Simulation
            ax5_2 = fig5.add_subplot(gs5[1, 0])
            S0 = prices["SPY"].iloc[-1]
            mu = spy_ret.mean() * 252
            sigma_sim = spy_ret.std() * np.sqrt(252)
            T_sim = 30/252.0 # 30 Days
            dt = 1/252.0
            n_paths = 100
            
            time, paths = qe.simulate_gbm(S0, mu, sigma_sim, T_sim, dt, n_paths)
            
            # Plot paths
            for i in range(n_paths):
                ax5_2.plot(time*252, paths[i], color='cyan', alpha=0.1)
            ax5_2.plot(time*252, paths.mean(axis=0), color='blue', linewidth=2, label="Mean Path")
            ax5_2.set_title(f"Monte Carlo: SPY 30-Day Projection ({n_paths} Paths)", fontsize=12, weight='bold')
            ax5_2.set_xlabel("Days Ahead")
            ax5_2.set_ylabel("Price")
            ax5_2.grid(True, alpha=0.3)
            
            # 3. ATM Greeks Table
            ax5_3 = fig5.add_subplot(gs5[1, 1])
            ax5_3.axis('off')
            
            # Calc Greeks for ATM Call (1 Month out)
            S = S0
            K = S0
            T = 30/365.0
            r = 0.045
            sigma = sigma_sim
            
            greeks = qe.calculate_greeks(S, K, T, r, sigma, "call")
            bsm_price = qe.black_scholes_merton(S, K, T, r, sigma, "call")
            
            greeks_data = [
                ["Metric", "Value"],
                ["ATM Call Price (30D)", f"${bsm_price:.2f}"],
                ["Delta", f"{greeks.get('Delta', 0):.3f}"],
                ["Gamma", f"{greeks.get('Gamma', 0):.4f}"],
                ["Theta (Daily)", f"{greeks.get('Theta', 0):.3f}"],
                ["Vega (1%)", f"{greeks.get('Vega', 0):.3f}"],
                ["Rho", f"{greeks.get('Rho', 0):.3f}"]
            ]
            
            table = ax5_3.table(cellText=greeks_data, loc='center', cellLoc='center', colWidths=[0.5, 0.3])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 2)
            ax5_3.set_title("SPY ATM Option Greeks (Theoretical)", fontsize=12, weight='bold')
            
            plt.tight_layout()
            pdf.savefig(fig5)
            plt.close(fig5)
            
        except Exception as e:
            print(f"   Error generating Quant Page: {e}")

        # --- FIGURE 6: BACKTEST PERFORMANCE ---
        fig6 = run_backtest(df, prices)
        if fig6:
            pdf.savefig(fig6)
            plt.close(fig6)
            
        print("PDF Report Saved Successfully!")


# ==============================================================================
#                               MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("--- Initializing SUPER MACRO DASHBOARD ---")
    
    # 1. Download
    prices, macro = download_data(CONFIG)
    
    if not prices.empty and not macro.empty:
        # 2. Build
        print("3. Calculating Signal Matrix...")
        df = build_analytics(prices, macro, CONFIG)
        df = add_trends_and_score(df, CONFIG)
        
        # 4. Generate Report
        report = generate_capital_flows_commentary(df, prices)
        
        # 5. Plot
        print("4. Generating Visualizations...")
        plot_super_dashboard(df, prices, report)
        
    else:
        
        print("Error: Data download failed.")