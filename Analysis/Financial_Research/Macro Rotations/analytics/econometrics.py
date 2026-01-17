"""
Econometrics Models for Advanced Quant Lab.
Includes:
- Markov Regime Switching (HMM) via Statsmodels
- Ornstein-Uhlenbeck (Mean Reversion) fitting
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from typing import Dict, Tuple

def fit_markov_regime_switching(returns: pd.Series, k_regimes: int = 2) -> Dict:
    """
    Fits a Markov Switching Regression (HMM) to detect Volatility Regimes.
    Model: Constant Mean, Switching Variance.
    
    Returns:
    - Summary
    - Smoothed Probabilities
    - Regime Map (0=Low Vol, 1=High Vol)
    """
    print(f"   Fitting Markov Regime Switching ({k_regimes} Regimes)...")
    res = {"model": None, "probs": pd.DataFrame(), "regimes": {}, "params": {}}
    
    if len(returns) < 100: return res
    
    try:
        # Scale returns for numerical stability (often pct_change is small)
        y = returns * 100 
        
        # We model variance switching. Mean could be switching too but Variance is clearer.
        model = MarkovRegression(y, k_regimes=k_regimes, trend='c', switching_variance=True)
        results = model.fit(disp=False)
        
        # Extract Volatility Params (sigma^2)
        # Statsmodels naming depends on version, usually "sigma2[0]", "sigma2[1]"
        params = results.params
        sigmas = {i: np.sqrt(params.get(f"sigma2[{i}]", 0)) for i in range(k_regimes)}
        
        # Identify High vs Low Vol
        # Sort regimes by sigma
        sorted_regimes = sorted(sigmas.items(), key=lambda x: x[1])
        
        regime_map = {}
        if k_regimes == 2:
            regime_map[sorted_regimes[0][0]] = "Low Vol (Calm)"
            regime_map[sorted_regimes[1][0]] = "High Vol (Stress)"
        elif k_regimes == 3:
            regime_map[sorted_regimes[0][0]] = "Low Vol"
            regime_map[sorted_regimes[1][0]] = "Med Vol"
            regime_map[sorted_regimes[2][0]] = "High Vol (Crisis)"
            
        prob_df = results.smoothed_marginal_probabilities
        # Rename columns using map
        prob_df.columns = [regime_map.get(c, f"Regime {c}") for c in prob_df.columns]
        
        res = {
            "probs": prob_df,
            "regimes": regime_map,
            "sigma_vals": sigmas,
            "aic": results.aic
        }
        
    except Exception as e:
        print(f"   HMM Error: {e}")
        
    return res

def fit_ou_process(series: pd.Series) -> Dict:
    """
    Fits an Ornstein-Uhlenbeck Mean Reverting Process.
    dX_t = theta * (mu - X_t) * dt + sigma * dW_t
    
    Discretized (AR1):
    X_{t+1} = X_t * e^{-theta dt} + mu * (1 - e^{-theta dt}) + epsilon
    
    Regression: X_{t+1} = a * X_t + b + error
    slope = e^{-theta dt}
    intercept = mu * (1 - slope)
    
    Returns: theta (speed), mu (mean), sigma (vol), half_life
    """
    res = {"theta": np.nan, "mu": np.nan, "sigma": np.nan, "half_life": np.nan}
    
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty or len(series) < 10: return res
    
    X_t = series.iloc[:-1].values
    X_tp1 = series.iloc[1:].values
    
    # Linear Regression X_tp1 ~ X_t
    X_t_exog = sm.add_constant(X_t)
    model = sm.OLS(X_tp1, X_t_exog)
    results = model.fit()
    
    a = results.params[1] # slope
    b = results.params[0] # intercept
    resid_std = results.resid.std()
    
    dt = 1/252.0
    
    # Solve for OU params
    # a = exp(-theta * dt)  => theta = -ln(a) / dt
    if a >= 1.0: 
        # Non-stationary / Random Walk (No mean reversion)
        theta = 0.0
    else:
        theta = -np.log(a) / dt
        
    # mu = b / (1 - a)
    mu_ou = b / (1 - a) if (1-a) != 0 else np.nan
    
    # sigma_eq = std(error) * sqrt( 2*theta / (1-a^2) ) ... approx sigma * sqrt(dt)
    # sigma_ou = resid_std / sqrt( (1 - exp(-2*theta*dt)) / (2*theta) )
    # Simplified Euler: sigma * sqrt(dt) = resid_std
    sigma_ou = resid_std / np.sqrt(dt)
    
    half_life = np.log(2) / theta if theta > 0 else np.inf
    
    res = {
        "theta": theta,
        "mu": mu_ou,
        "sigma": sigma_ou,
        "half_life": half_life,
        "r_squared": results.rsquared
    }
    
    return res
