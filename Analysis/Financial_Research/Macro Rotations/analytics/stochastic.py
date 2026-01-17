"""
Stochastic Calculus Models for Advanced Quant Lab.
Includes:
- Heston Stochastic Volatility Model
- Merton Jump Diffusion Model
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict

def simulate_heston(S0, v0, mu, kappa, theta, xi, rho, T, n_steps, n_paths) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates Heston Stochastic Volatility Model.
    
    dS_t = mu * S_t * dt + sqrt(v_t) * S_t * dW_1
    dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_2
    
    Parameters:
    - kappa: Mean reversion speed of vol
    - theta: Long-run variance
    - xi:    Vol of vol
    - rho:   Correlation between price and vol (usually negative for equities)
    
    Returns: time_grid, price_paths, vol_paths
    """
    dt = T / n_steps
    time = np.linspace(0, T, n_steps + 1)
    
    # Correlated Brownian Motions
    means = [0, 0]
    covs = [[1, rho], [rho, 1]]
    
    # Generate dW1, dW2
    covariance_matrix = np.array(covs)
    dB = np.random.multivariate_normal(means, covariance_matrix, size=(n_paths, n_steps)) * np.sqrt(dt)
    dW1 = dB[:, :, 0]
    dW2 = dB[:, :, 1]
    
    # Initialize arrays
    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    
    S[:, 0] = S0
    v[:, 0] = v0
    
    for t in range(n_steps):
        # Euler-Maruyama for Volatility (with max(v,0) constraint)
        # Full Truncation Scheme is robust for Heston
        # Treat v_curr as the MAX(v_prev, 0)
        v_prev = v[:, t]
        v_curr = np.maximum(v_prev, 0)
        
        # dW2 term requires sqrt(v_curr)
        vol_term = np.sqrt(v_curr)
        
        dv = kappa * (theta - v_curr) * dt + xi * vol_term * dW2[:, t]
        v[:, t+1] = v[:, t] + dv
        
        # Euler-Maruyama for Price
        # Use v_curr for price diffusion too (Full Truncation)
        dS = mu * S[:, t] * dt + vol_term * S[:, t] * dW1[:, t]
        S[:, t+1] = S[:, t] + dS
        
    return time, S, v

def simulate_merton_jump(S0, mu, sigma, T, n_steps, n_paths, lambda_jump, mu_jump, sigma_jump) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates Merton Jump Diffusion Model.
    GBM + Poisson Jumps.
    
    dS/S = (mu - lambda * k)dt + sigma * dW + (Y - 1)dN
    
    Parameters:
    - lambda_jump: Jump intensity (jumps per year)
    - mu_jump:     Mean log-jump size
    - sigma_jump:  Std dev of log-jump size
    """
    dt = T / n_steps
    time = np.linspace(0, T, n_steps + 1)
    
    # 1. Standard GBM Component
    # dW
    Z1 = np.random.normal(0, 1, size=(n_paths, n_steps))
    
    # 2. Jump Component
    # Poisson process for number of jumps in dt
    # Probability of jump approx lambda * dt
    # Using Poisson distribution directly
    N_jumps = np.random.poisson(lambda_jump * dt, size=(n_paths, n_steps))
    
    # Jump size Y ~ LogNormal. Log(Y) ~ N(mu_jump, sigma_jump)
    # Sum of Log-Jumps over N events
    # If N=0, sum is 0. If N=1, sum is one draw, etc.
    # To vectorize efficiently:
    # Drift correction for mean jump size k = E[Y]-1 = exp(mu_j + 0.5*sig_j^2) - 1
    k = np.exp(mu_jump + 0.5 * sigma_jump**2) - 1
    
    # Simulation in Log-Space
    # d(ln S) = (mu - 0.5*sigma^2 - lambda*k) dt + sigma dW + Sum(ln Y)
    
    drift_term = (mu - 0.5 * sigma**2 - lambda_jump * k) * dt
    diffusion_term = sigma * np.sqrt(dt) * Z1
    
    # Jump Term: Sum of N Gaussians with (mu_jump, sigma_jump)
    # Since N is random, we can just draw J ~ N(N*mu_jump, N*sigma_jump^2) ???
    # Actually, sum of i.i.d normals is Normal.
    # Sum_{i=1 to N} X_i where X_i ~ N(m, s)  ==> Z ~ N(N*m, N*s^2)
    
    jump_mean = N_jumps * mu_jump
    jump_std = np.sqrt(N_jumps) * sigma_jump
    # Sample jump magnitude
    J_log = np.random.normal(jump_mean, jump_std)
    # Handle cases where N=0 (std=0) - numpy handles simulation ok, std=0 -> result mean.
    
    d_log_S = drift_term + diffusion_term + J_log
    
    # Accumulate
    log_S = np.cumsum(d_log_S, axis=1)
    log_S = np.hstack([np.zeros((n_paths, 1)), log_S]) 
    
    S = S0 * np.exp(log_S)
    
    return time, S

def simulate_hawkes_intensity(mu, alpha, beta, T, n_steps) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates a self-exciting Hawkes Process Intesity.
    lambda(t) = mu + sum(alpha * exp(-beta * (t - t_i)))
    
    This is often used to model Order Flow arrival or Volatility Clustering.
    We just simulate the intensity path for visualization.
    
    Parameters:
    - mu: Base intensity
    - alpha: Excitation magnitude
    - beta: Decay rate (must be > alpha for stationarity)
    
    Returns: time_grid, intensity_path, event_times
    """
    # 1. Simulate Event Times (Ogata's Thinning Algorithm)
    # This is a bit complex for a quick demo. We will use a simplified discretization.
    # Discretized:
    # lambda(t+dt) = mu + (lambda(t) - mu) * exp(-beta * dt) + alpha * dN(t)
    
    dt = T / n_steps
    time = np.linspace(0, T, n_steps + 1)
    
    intensity = np.zeros(n_steps + 1)
    events = np.zeros(n_steps + 1)
    
    intensity[0] = mu
    
    # We will just simulate the intensity path assuming random Poisson arrivals driving it
    # True Hawkes generates events based on intensity.
    
    current_time = 0
    event_times = []
    
    # Basic Thinning (Lewis-Shedler)
    # Max intensity approx?
    # Let's do step-wise simulation
    
    lamb = mu
    for t in range(n_steps):
        # 1. Decay
        intensity[t] = lamb
        
        # 2. Probability of event in dt is lamb * dt
        # Check for event
        if np.random.rand() < lamb * dt:
            events[t] = 1
            event_times.append(time[t])
            # Jump in intensity
            lamb = lamb + alpha
        else:
            # Decay towards mu
            lamb = mu + (lamb - mu) * np.exp(-beta * dt)
            
    intensity[-1] = lamb
    return time, intensity, np.array(event_times)
