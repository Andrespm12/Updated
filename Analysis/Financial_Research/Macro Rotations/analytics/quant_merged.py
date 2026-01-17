"""
Quantitative Analytics: Options, Volatility, and Backtesting.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.decomposition import PCA
from scipy.stats import norm
from typing import Dict, Tuple

def calculate_option_skew(ticker_symbol: str = "SPY") -> Dict:
    """Calculates Volatility Skew (Put IV - Call IV)."""
    print(f"   Calculating Vol Skew for {ticker_symbol}...")
    res = {"skew": None, "status": "N/A"}
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        history = ticker.history(period="1d")
        if history.empty: return res
        current_price = history["Close"].iloc[-1]
        
        exps = ticker.options
        if not exps: return res
        
        # Get nearest monthly
        opt = ticker.option_chain(exps[0])
        calls = opt.calls
        puts = opt.puts
        
        # Find 10% OTM Puts (Strike ~ 90% of Price)
        target_put = current_price * 0.90
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

def calculate_gamma_flip(ticker_symbol: str = "SPY") -> Dict:
    """Estimates the 'Gamma Flip' Level (Zero Gamma)."""
    print(f"   Calculating Gamma Flip Level for {ticker_symbol}...")
    res = {"level": None, "current_price": None, "status": "N/A"}
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        history = ticker.history(period="1d")
        if history.empty: return res
        current_price = history["Close"].iloc[-1]
        
        exps = ticker.options
        if not exps: return res
        
        # Aggregate OI across first few expirations
        strikes = []
        call_ois = []
        put_ois = []
        
        for date in exps[:2]:
            opt = ticker.option_chain(date)
            calls = opt.calls
            puts = opt.puts
            
            df = pd.merge(calls[['strike', 'openInterest']], puts[['strike', 'openInterest']], on='strike', how='outer', suffixes=('_call', '_put')).fillna(0)
            strikes.extend(df['strike'].tolist())
            call_ois.extend(df['openInterest_call'].tolist())
            put_ois.extend(df['openInterest_put'].tolist())
            
        df_total = pd.DataFrame({"strike": strikes, "call_oi": call_ois, "put_oi": put_ois})
        df_total = df_total.groupby("strike").sum().reset_index()
        df_total["net_oi"] = df_total["call_oi"] - df_total["put_oi"]
        
        # Find zero crossing
        df_total = df_total.sort_values("strike")
        df_total["net_oi_smooth"] = df_total["net_oi"].rolling(5, center=True).mean()
        
        crossings = df_total[np.sign(df_total["net_oi_smooth"]).diff() != 0]
        
        if crossings.empty:
            flip_level = current_price
        else:
            # Find closest to current price
            crossings["dist"] = abs(crossings["strike"] - current_price)
            flip_level = crossings.sort_values("dist").iloc[0]["strike"]
            
        dist_pct = (current_price - flip_level) / flip_level
        if dist_pct > 0.02: status = "BULLISH (Above Flip)"
        elif dist_pct < -0.02: status = "BEARISH (Below Flip)"
        else: status = "VOLATILE (At Flip)"
        
        res = {"level": flip_level, "current_price": current_price, "status": status}
        
    except Exception as e:
        print(f"   Gamma Flip Error: {e}")
        
    return res

def get_options_data(ticker_symbol: str = "SPY", calc_gex: bool = False) -> Dict:
    """Fetches real-time options data for Sentiment & Structure."""
    print(f"   Fetching Options Data for {ticker_symbol}...")
    metrics = {"pcr": None, "iv_call": None, "net_oi": None, "max_pain": None}
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        exps = ticker.options
        if not exps: return metrics
        
        opt = ticker.option_chain(exps[0])
        calls = opt.calls
        puts = opt.puts
        
        call_vol = calls['volume'].sum()
        put_vol = puts['volume'].sum()
        metrics["pcr"] = put_vol / call_vol if call_vol > 0 else 0
        
        if 'impliedVolatility' in calls.columns:
            metrics["iv_call"] = calls['impliedVolatility'].mean()
            
        if calc_gex:
            # Net OI
            df_opts = pd.merge(calls[['strike', 'openInterest']], puts[['strike', 'openInterest']], on='strike', suffixes=('_call', '_put'))
            df_opts['net_oi'] = df_opts['openInterest_call'] - df_opts['openInterest_put']
            metrics["net_oi"] = df_opts['net_oi'].sum()
            
            # Max Pain (simplified)
            strikes = df_opts['strike'].values
            min_pain = float('inf')
            pain_strike = 0
            
            # Sample strikes to speed up
            if len(strikes) > 50:
                 candidates = strikes[::int(len(strikes)/50)]
            else:
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
    """Calculates Systemic Risk using PCA Absorption Ratio."""
    print("   Calculating Systemic Risk (PCA)...")
    res = {"ratio": None, "status": "N/A", "history": pd.Series(dtype=float)}
    
    basket = ["SPY", "TLT", "GLD", "UUP", "XLE"]
    valid = [b for b in basket if b in prices.columns]
    
    if len(valid) < 3: return res
    
    rets = prices[valid].pct_change().dropna()
    window = 126
    absorption_history = []
    dates = []
    
    if len(rets) < window: return res
    
    lookback = 500
    start_idx = max(0, len(rets) - lookback)
    
    pca = PCA(n_components=1)
    
    for i in range(start_idx, len(rets)):
        if i < window: continue
        
        subset = rets.iloc[i-window:i]
        subset_std = (subset - subset.mean()) / subset.std()
        subset_std = subset_std.dropna(axis=1)
        
        if subset_std.shape[1] < 3: continue
        
        pca.fit(subset_std)
        var_ratio = pca.explained_variance_ratio_[0]
        
        absorption_history.append(var_ratio)
        dates.append(rets.index[i])
        
    if not absorption_history: return res
    
    series = pd.Series(absorption_history, index=dates)
    curr_val = series.iloc[-1]
    
    if curr_val > 0.75: status = "CRITICAL (High Correlation)"
    elif curr_val > 0.65: status = "ELEVATED"
    else: status = "NORMAL (Decoupled)"
    
    res = {"ratio": curr_val, "status": status, "history": series}
    return res

def calculate_garch_crash_prob(prices: pd.DataFrame) -> Dict:
    """Estimates 'Crash Probability' using simplified GARCH-like EWMA."""
    print("   Calculating GARCH Crash Probability...")
    res = {"prob": None, "vol_forecast": None, "status": "N/A"}
    
    if "SPY" not in prices.columns: return res
    
    spy = prices["SPY"]
    rets = spy.pct_change().dropna() * 100
    
    lambda_param = 0.94
    vol_series = rets.ewm(alpha=(1-lambda_param)).std()
    current_vol_daily = vol_series.iloc[-1]
    
    vol_5d = current_vol_daily * np.sqrt(5)
    
    # Prob of Drop > 5%
    z_score = -5.0 / vol_5d
    prob_crash = norm.cdf(z_score) * 100
    
    if prob_crash > 5.0: status = "HIGH RISK (>5%)"
    elif prob_crash > 1.0: status = "ELEVATED"
    else: status = "LOW"
    
    res = {"prob": prob_crash, "vol_forecast": vol_5d, "status": status}
    return res

def calculate_correlation_surprise(prices: pd.DataFrame) -> Dict:
    """Calculates Correlation Surprise (Stocks vs Bonds)."""
    print("   Calculating Correlation Surprise...")
    res = {"surprise": None, "status": "N/A", "curr_corr": None}
    
    if "SPY" not in prices.columns or "TLT" not in prices.columns: return res
    
    spy = prices["SPY"].pct_change().dropna()
    tlt = prices["TLT"].pct_change().dropna()
    
    df_c = pd.DataFrame({"SPY": spy, "TLT": tlt}).dropna()
    
    corr_1m = df_c["SPY"].rolling(21).corr(df_c["TLT"]).iloc[-1]
    corr_1y = df_c["SPY"].rolling(252).corr(df_c["TLT"]).iloc[-1]
    
    surprise = corr_1m - corr_1y
    
    if surprise > 0.5: status = "SHOCK (Correlations Spiking)"
    elif surprise > 0.2: status = "TIGHTENING (Fading Diversification)"
    elif surprise < -0.2: status = "DECOUPLING (Available Diversification)"
    else: status = "STABLE"
    
    res = {"surprise": surprise, "status": status, "curr_corr": corr_1m, "lt_corr": corr_1y}
    return res

def calculate_portfolio_risk(curve: pd.Series, confidence_level: float = 0.95):
    """Calculates Historical VaR and CVaR."""
    rets = curve.pct_change().dropna()
    var_95 = rets.quantile(1 - confidence_level)
    cvar_95 = rets[rets <= var_95].mean()
    return var_95, cvar_95

def calculate_strategy_performance(df: pd.DataFrame, prices: pd.DataFrame):
    """Calculates metrics for strategies."""
    # Assets needed
    assets = ["SPY", "TLT", "GLD", "BTC-USD", "RSP", "XLY", "XLP", "QQQ", "EEM", "XBI", "SHY"]
    missing = [a for a in assets if a not in prices.columns]
    if missing: return None, None
        
    rets = prices[assets].pct_change().dropna()
    spy_ret = rets["SPY"]
    
    # 1. Macro Regime
    sig_macro = (df["MACRO_SCORE"] > 0).astype(int).reindex(rets.index).shift(1).fillna(0)
    ret_macro = sig_macro * rets["SPY"] + (1 - sig_macro) * rets["SHY"]
    
    # 2. Liquidity Valve
    sig_liq = (df["CF_Liquidity_Valve"] > df["CF_Liquidity_Valve_MA200"]).astype(int).reindex(rets.index).shift(1).fillna(0)
    ret_liq = sig_liq * rets["QQQ"] + (1 - sig_liq) * rets["SHY"]
    
    # 3. Breadth Trend
    sig_breadth = (df["CF_Breadth"] > df["CF_Breadth_MA200"]).astype(int).reindex(rets.index).shift(1).fillna(0)
    ret_breadth = sig_breadth * rets["SPY"] + (1 - sig_breadth) * rets["SHY"]
    
    # 4. Consumer Rotation
    sig_cons = (df["ROT_Cyc_Def_Sectors"] > df["ROT_Cyc_Def_Sectors_MA200"]).astype(int).reindex(rets.index).shift(1).fillna(0)
    ret_cons = sig_cons * rets["XLY"] + (1 - sig_cons) * rets["XLP"]
    
    # 5. VIX Filter
    if "CF_VIX" in df.columns:
        sig_vix = (df["CF_VIX"] < 20).astype(int).reindex(rets.index).shift(1).fillna(0)
        ret_vix = sig_vix * rets["SPY"] + (1 - sig_vix) * rets["SHY"]
    else:
        ret_vix = rets["SPY"]
        
    # 6. Core-Satellite (CCI PB)
    sat_assets = ["QQQ", "EEM", "GLD", "XBI"]
    mom_3m = prices[sat_assets].pct_change(63).dropna()
    try:
        monthly_mom = mom_3m.resample('ME').last() # Use ME for pandas > 2.0
    except:
        monthly_mom = mom_3m.resample('M').last()

    monthly_winner = monthly_mom.idxmax(axis=1)
    daily_signal = monthly_winner.reindex(rets.index, method='ffill').shift(1)
    
    sat_ret = pd.Series(0.0, index=rets.index)
    for asset in sat_assets:
        mask = (daily_signal == asset)
        sat_ret[mask] = rets[asset][mask]
        
    ret_core_sat = 0.50 * rets["SPY"] + 0.15 * rets["TLT"] + 0.35 * sat_ret
    
    # Combine
    curves = pd.DataFrame({
        "Core-Satellite (CCI)": (1 + ret_core_sat).cumprod(),
        "Macro Regime": (1 + ret_macro).cumprod(),
        "Liquidity Valve": (1 + ret_liq).cumprod(),
        "Breadth Trend": (1 + ret_breadth).cumprod(),
        "Consumer Rotation": (1 + ret_cons).cumprod(),
        "VIX Filter": (1 + ret_vix).cumprod(),
    })
    curves["SPY (Hold)"] = (1 + spy_ret).cumprod()
    
    # 8. Sector Leaders (Top 3)
    try:
        sectors = ["XLI", "XLB", "XLU", "XLF", "XLK", "XLE", "XLV", "XLC", "XLY", "XLP"]
        valid_secs = [s for s in sectors if s in prices.columns]
        if len(valid_secs) >= 3:
            sec_prices = prices[valid_secs]
            try:
                sec_m = sec_prices.resample('ME').last()
            except:
                sec_m = sec_prices.resample('M').last()
            mom_3m = sec_m.pct_change(3)
            
            top3_mask = mom_3m.apply(lambda x: x >= x.nlargest(3).min(), axis=1)
            top3_mask = top3_mask.shift(1).dropna()
            daily_mask = top3_mask.reindex(spy_ret.index, method='ffill').dropna()
            
            sec_rets = sec_prices.pct_change()
            common_idx = daily_mask.index.intersection(sec_rets.index)
            d_mask = daily_mask.loc[common_idx]
            d_rets = sec_rets.loc[common_idx]
            
            daily_strat_ret = (d_rets * d_mask).sum(axis=1) / 3
            curves["Sector Leaders (Top 3)"] = (1 + daily_strat_ret).cumprod()
    except Exception as e:
        print(f"Sector Strat Error: {e}")
        
    # 9. Vol Control
    try:
        target_vol = 0.12
        real_vol = spy_ret.rolling(20).std() * np.sqrt(252)
        weight = (target_vol / real_vol).clip(0, 1.5).shift(1)
        vol_ctrl_ret = spy_ret * weight
        curves["Vol Control (12%)"] = (1 + vol_ctrl_ret).cumprod()
    except Exception as e:
        print(f"Vol Ctrl Error: {e}")

    # Metrics
    metrics = {}
    for col in curves.columns:
        s = curves[col].dropna()
        if s.empty:
            metrics[col] = {"CAGR": 0.0, "Sharpe": 0.0, "MaxDD": 0.0}
            continue
            
        start_date = s.index[0]
        end_date = s.index[-1]
        days = (end_date - start_date).days
        years = max(1, days / 365.25)
        
        total_ret = (s.iloc[-1] / s.iloc[0]) - 1
        cagr = (1 + total_ret) ** (1 / years) - 1
        
        daily_rets = s.pct_change().dropna()
        mean = daily_rets.mean() * 252
        std = daily_rets.std() * np.sqrt(252)
        sharpe = mean / std if std > 0 else 0
        
        roll_max = s.cummax()
        dd = (s / roll_max) - 1
        max_dd = dd.min()
        
        metrics[col] = {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd}
        
    return metrics, curves
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

# ==========================================
# MODULE 1: STOCHASTIC PROCESSES (GBM)
# ==========================================
def simulate_gbm(S0, mu, sigma, T, dt, n_paths):
    """
    Simulates Geometric Brownian Motion (GBM) paths.
    dS = mu*S*dt + sigma*S*dW
    """
    n_steps = int(T / dt)
    time = np.linspace(0, T, n_steps + 1)
    
    # Generate random Brownian Motion
    # dW ~ N(0, sqrt(dt))
    dW = np.random.normal(0, np.sqrt(dt), size=(n_paths, n_steps))
    
    # Cumulative sum for Brownian Path
    W = np.cumsum(dW, axis=1)
    # Add starting 0
    W = np.hstack([np.zeros((n_paths, 1)), W])
    
    # Exact solution for GBM: S(t) = S0 * exp((mu - 0.5*sigma^2)*t + sigma*W(t))
    # Broadcast time across paths
    time_matrix = np.tile(time, (n_paths, 1))
    
    exponent = (mu - 0.5 * sigma**2) * time_matrix + sigma * W
    paths = S0 * np.exp(exponent)
    
    return time, paths

# ==========================================
# MODULE 2: DERIVATIVES & GREEKS (BSM)
# ==========================================
def black_scholes_merton(S, K, T, r, sigma, option_type="call"):
    """
    Calculates BSM Option Price.
    S: Spot Price
    K: Strike Price
    T: Time to Maturity (years)
    r: Risk-free rate
    sigma: Volatility
    """
    if T <= 0:
        return max(0, S - K) if option_type == "call" else max(0, K - S)
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
    return price

def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Calculates Delta, Gamma, Theta, Vega, Rho.
    """
    if T <= 0: return {}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    N_prime = norm.pdf(d1)
    
    # Delta
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
        
    # Gamma (Same for Call/Put)
    gamma = N_prime / (S * sigma * np.sqrt(T))
    
    # Vega (Same for Call/Put)
    vega = S * np.sqrt(T) * N_prime / 100 # Divide by 100 for % change
    
    # Theta
    if option_type == "call":
        theta = (- (S * sigma * N_prime) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (- (S * sigma * N_prime) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
                 
    # Rho
    if option_type == "call":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta": theta,
        "Rho": rho
    }

# ==========================================
# MODULE 3: VOLATILITY (GARCH)
# ==========================================
def garch_1_1_likelihood(params, returns):
    """
    Negative Log-Likelihood for GARCH(1,1).
    params: [omega, alpha, beta]
    sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}
    """
    omega, alpha, beta = params
    n = len(returns)
    sigma_2 = np.zeros(n)
    
    # Initialize
    sigma_2[0] = np.var(returns)
    
    for t in range(1, n):
        sigma_2[t] = omega + alpha * returns[t-1]**2 + beta * sigma_2[t-1]
        
    # Log Likelihood
    # L = -0.5 * sum(log(sigma^2) + epsilon^2 / sigma^2)
    # We return negative L for minimization
    
    # Avoid log(0) or negative variance
    sigma_2 = np.maximum(sigma_2, 1e-6)
    
    log_lik = -0.5 * np.sum(np.log(sigma_2) + returns**2 / sigma_2)
    return -log_lik

def fit_garch(returns):
    """
    Fits a GARCH(1,1) model to returns using MLE.
    Returns estimated volatility series.
    """
    # Initial guess
    # omega ~ var * (1 - alpha - beta)
    # alpha + beta < 1 for stationarity
    initial_params = [np.var(returns)*0.05, 0.05, 0.90]
    
    # Constraints: params > 0, alpha + beta < 1
    bounds = ((1e-6, None), (1e-6, 1), (1e-6, 1))
    
    def constraint(params):
        return 1 - params[1] - params[2] # alpha + beta < 1
        
    cons = ({'type': 'ineq', 'fun': constraint})
    
    result = minimize(garch_1_1_likelihood, initial_params, args=(returns,), 
                      bounds=bounds, constraints=cons, method='SLSQP')
                      
    omega, alpha, beta = result.x
    
    # Reconstruct volatility path
    n = len(returns)
    sigma_2 = np.zeros(n)
    sigma_2[0] = np.var(returns)
    for t in range(1, n):
        sigma_2[t] = omega + alpha * returns[t-1]**2 + beta * sigma_2[t-1]
        
    return np.sqrt(sigma_2), (omega, alpha, beta)

# ==========================================
# MODULE 4: REGRESSION ANALYSIS
# ==========================================
def rolling_regression(y, x, window=60):
    """
    Calculates Rolling Beta and Alpha.
    y: Asset Returns
    x: Benchmark Returns
    """
    from scipy.stats import linregress
    
    betas = []
    alphas = []
    r_squareds = []
    dates = []
    
    # Align data
    common_idx = y.index.intersection(x.index)
    y = y.loc[common_idx]
    x = x.loc[common_idx]
    
    for i in range(window, len(common_idx)):
        y_win = y.iloc[i-window:i]
        x_win = x.iloc[i-window:i]
        
        slope, intercept, r_value, p_value, std_err = linregress(x_win, y_win)
        
        betas.append(slope)
        alphas.append(intercept * 252) # Annualized Alpha
        r_squareds.append(r_value**2)
        dates.append(common_idx[i])
        
    return pd.DataFrame({
        "Beta": betas,
        "Alpha": alphas,
        "R2": r_squareds
    }, index=dates)
