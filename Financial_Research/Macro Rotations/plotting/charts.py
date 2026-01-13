"""
Visualization Functions (Matplotlib).
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm
from typing import Dict, List
from analytics.rotations import calculate_rrg_metrics, predict_sector_rotation, calculate_seasonality
from analytics.macro_models import calculate_macro_radar, calculate_regime_gmm
from analytics.quant import calculate_strategy_performance, calculate_systemic_risk_pca, fit_garch, simulate_gbm, calculate_greeks, black_scholes_merton, calculate_antifragility_metrics, calculate_market_internals
from analytics.stochastic import simulate_heston, simulate_merton_jump, simulate_hawkes_intensity
from analytics.econometrics import fit_markov_regime_switching, fit_ou_process
from analytics.scenarios import calculate_move_probabilities, generate_contingencies

def run_backtest_plot(df: pd.DataFrame, prices: pd.DataFrame) -> plt.Figure:
    """Runs a vectorised backtest for multiple strategies and plots the results."""
    print("   Running Backtest & Projections...")
    
    metrics, curves = calculate_strategy_performance(df, prices)
    if curves is None or curves.empty:
        print("   Backtest failed: Missing assets.")
        return None
        
    # Simplified Projection: Monte Carlo using Full History
    projections = pd.DataFrame()
    proj_metrics = {}
    
    future_days = 252
    n_sims = 1000
    last_date = curves.index[-1]
    future_dates = pd.date_range(start=last_date, periods=future_days+1, freq='B')
    np.random.seed(42)
    
    for strat in curves.columns:
        series = curves[strat]
        rets = series.pct_change().infer_objects(copy=False).dropna()
        mu = rets.mean()
        sigma = rets.std()
        last_price = series.iloc[-1]
        
        ret_sim = np.random.normal(mu, sigma, (future_days, n_sims))
        price_paths = last_price * (1 + ret_sim).cumprod(axis=0)
        median_path = np.median(price_paths, axis=1)
        
        full_proj = np.concatenate(([last_price], median_path))
        projections[strat] = full_proj
        
        final_val = median_path[-1]
        exp_ret = (final_val / last_price) - 1
        proj_metrics[strat] = exp_ret 

    # Plotting
    plt.style.use('default')
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    
    ax0 = fig.add_subplot(gs[0])
    colors = {
        "Core-Satellite (CCI)": "#1f77b4",
        "Macro Regime": "#2ca02c",
        "Liquidity Valve": "#d62728",
        "Breadth Trend": "#9467bd",
        "Consumer Rotation": "#ff7f0e",
        "VIX Filter": "#e377c2",
        "Sector Leaders (Top 3)": "#8c564b",
        "Vol Control (12%)": "#17becf",
        "SPY (Hold)": "black"
    }
    
    for strat in curves.columns:
        cagr = metrics[strat]["CAGR"]
        label = f"{strat} (CAGR: {cagr:.1%})"
        color = colors.get(strat, "grey")
        style = "--" if "Hold" in strat else "-"
        width = 3 if "Core-Satellite" in strat else (1.5 if "Hold" in strat else 1.5)
        alpha = 1.0 if "Core-Satellite" in strat else 0.7
        
        ax0.plot(curves.index, curves[strat], label=label, color=color, linestyle=style, linewidth=width, alpha=alpha)
        
        if strat in projections.columns:
            ax0.plot(future_dates, projections[strat], color=color, linestyle=":", linewidth=width, alpha=0.6)
            ax0.scatter(future_dates[-1], projections[strat].iloc[-1], color=color, s=20)

    ax0.set_title("Backtest & 1-Year Projection (Median Path)", fontsize=16, weight='bold', color='black')
    ax0.set_yscale('log')
    ax0.legend(loc="upper left", fontsize=10, frameon=True, facecolor='white', edgecolor='grey')
    ax0.grid(True, which="both", alpha=0.3, color='grey', linestyle=':')
    ax0.set_ylabel("Cumulative Return (Log Scale)", fontsize=12)
    ax0.axvline(last_date, color='black', linestyle='-', linewidth=1)
    ax0.text(last_date, ax0.get_ylim()[0], "  TODAY", rotation=90, va='bottom', weight='bold')
    
    ax1 = fig.add_subplot(gs[1])
    ax1.axis('off')
    
    table_data = [["Strategy", "Hist. CAGR", "Sharpe", "Max Drawdown", "Exp. CAGR (1Y)"]]
    sorted_strats = sorted(metrics.keys(), key=lambda x: metrics[x]['Sharpe'], reverse=True)
    
    for strat in sorted_strats:
        m = metrics[strat]
        exp = proj_metrics.get(strat, 0)
        table_data.append([
            strat, 
            f"{m['CAGR']:.1%}", 
            f"{m['Sharpe']:.2f}", 
            f"{m['MaxDD']:.1%}",
            f"{exp:.1%}"
        ])
    
    table = ax1.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3, 0.12, 0.12, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    plt.tight_layout()
    return fig

def plot_risk_macro_dashboard(df: pd.DataFrame, prices: pd.DataFrame) -> plt.Figure:
    """Generates Page 3: Macro Risk & Sector Rotation."""
    print("   Generating Risk & Macro Page...")
    plt.style.use('default')
    
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.5]) 
    
    # 1. Yield Curve
    ax1 = fig.add_subplot(gs[0])
    has_yields = "10Y_Yield" in df.columns and "2Y_Yield" in df.columns
    if has_yields:
        yc = (df["10Y_Yield"] - df["2Y_Yield"]).dropna()
        if not yc.empty:
            ax1.plot(yc.index, yc, color='black', label="10Y-2Y Spread")
            ax1.axhline(0, color='red', linestyle='--', linewidth=1)
            ax1.fill_between(yc.index, yc, 0, where=(yc < 0), color='red', alpha=0.3)
            ax1.fill_between(yc.index, yc, 0, where=(yc > 0), color='green', alpha=0.1)
            ax1.set_title("1. Yield Curve (10Y - 2Y): Recession Watch", fontsize=12, weight='bold')
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "Yield Curve Data Missing", ha='center')
        
    # 2. Credit Stress
    ax2 = fig.add_subplot(gs[1])
    if "HY_Spread" in df.columns:
        hy = df["HY_Spread"].dropna()
        ax2.plot(hy.index, hy, color='purple', label="High Yield Option-Adjusted Spread")
        ax2.axhline(hy.mean(), color='orange', linestyle='--', label="Avg Spread")
        ax2.set_title("2. Credit Stress (High Yield Spreads)", fontsize=12, weight='bold')
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Credit Spread Data Missing", ha='center')

    # 3. Bond Market Fear
    ax3 = fig.add_subplot(gs[2])
    if "MOVE_Index" in df.columns:
        move = df["MOVE_Index"].dropna()
        ax3.plot(move.index, move, color='blue', label="MOVE Index (Bond Volatility)")
        ax3.axhline(100, color='red', linestyle='--', label="Stress Threshold (100)")
        curr = move.iloc[-1]
        status = "ELEVATED (Risk Off)" if curr > 100 else "NORMAL"
        color = 'red' if curr > 100 else 'green'
        ax3.set_title(f"3. Bond Market Fear (MOVE Index): {status}", fontsize=12, weight='bold', color=color)
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "MOVE Index Data Missing", ha='center')
        
    # 4. Sector RRG
    ax4 = fig.add_subplot(gs[3])
    rrg = calculate_rrg_metrics(prices)
    if not rrg.empty:
        ax4.axhline(0, color='black', linestyle='-', linewidth=1)
        ax4.axvline(0, color='black', linestyle='-', linewidth=1)
        
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
        
        x_abs = max(abs(rrg['RS'].min()), abs(rrg['RS'].max()), 0.05) * 1.2
        y_abs = max(abs(rrg['Momentum'].min()), abs(rrg['Momentum'].max()), 0.05) * 1.2
        ax4.set_xlim(-x_abs, x_abs)
        ax4.set_ylim(-y_abs, y_abs)
        
        ax4.text(x_abs*0.9, y_abs*0.9, "LEADING", color='green', alpha=0.5, weight='bold', ha='right', va='top')
        ax4.text(x_abs*0.9, -y_abs*0.9, "WEAKENING", color='orange', alpha=0.5, weight='bold', ha='right', va='bottom')
        ax4.text(-x_abs*0.9, -y_abs*0.9, "LAGGING", color='red', alpha=0.5, weight='bold', ha='left', va='bottom')
        ax4.text(-x_abs*0.9, y_abs*0.9, "IMPROVING", color='blue', alpha=0.5, weight='bold', ha='left', va='top')

    plt.tight_layout()
    
    # --- Interpretation Box (Bottom) ---
    fig.subplots_adjust(bottom=0.15) # Make room
    
    interp_text = "Analysis & Key Signals:\n"
    
    # 1. Yield Curve
    if has_yields and "10Y_Yield" in df.columns:
        curr_spread = (df["10Y_Yield"].iloc[-1] - df["2Y_Yield"].iloc[-1])
        if curr_spread < 0:
            interp_text += f"• Yield Curve (10Y-2Y): INVERTED ({curr_spread:.2f}%). Strong historical recession signal.\n"
        else:
            interp_text += f"• Yield Curve (10Y-2Y): NORMAL ({curr_spread:.2f}%). No immediate recession signal from rates.\n"
            
    # 2. Credit
    if "HY_Spread" in df.columns:
        curr_hy = df["HY_Spread"].dropna().iloc[-1]
        if curr_hy > 5.0:
            interp_text += f"• Credit Spreads: STRESSED ({curr_hy:.2f}%). High default risk pricing. Equity-negative.\n"
        else:
            interp_text += f"• Credit Spreads: CALM ({curr_hy:.2f}%). Corporate bond market shows no panic.\n"
            
    # 3. Bond Vol
    if "MOVE_Index" in df.columns:
         curr_move = df["MOVE_Index"].dropna().iloc[-1]
         if curr_move > 100:
             interp_text += f"• Bond Vol (MOVE): ELEVATED ({curr_move:.0f}). Treasury market instability poses risk to stocks.\n"
             
    fig.text(0.05, 0.02, interp_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round'))
    return fig



def plot_monetary_plumbing(df: pd.DataFrame) -> plt.Figure:
    """Generates Page 4: Monetary & Economic Plumbing."""
    print("   Generating Monetary Plumbing Page...")
    plt.style.use('default')
    
    fig = plt.figure(figsize=(14, 18))
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 1])
    
    # 1. Liquidity Impulse
    ax1 = fig.add_subplot(gs[0])
    if "M2_YoY" in df.columns and "Fed_Assets_YoY" in df.columns:
        m2 = df["M2_YoY"].dropna()
        fed = df["Fed_Assets_YoY"].dropna()
        
        ax1.plot(m2.index, m2, color='green', label="M2 Money Supply (YoY)", linewidth=2)
        ax1.plot(fed.index, fed, color='blue', linestyle='--', label="Fed Balance Sheet (YoY)", linewidth=1.5)
        ax1.axhline(0, color='black', linewidth=1)
        ax1.fill_between(m2.index, m2, 0, where=(m2 > 0), color='green', alpha=0.1)
        ax1.fill_between(m2.index, m2, 0, where=(m2 < 0), color='red', alpha=0.1)
        
        ax1.set_title("1. Liquidity Impulse: Money Supply & Fed Assets", fontsize=12, weight='bold')
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        if "M2_Velocity" in df.columns:
            m2v = df["M2_Velocity"].dropna()
            ax1_twin = ax1.twinx()
            ax1_twin.plot(m2v.index, m2v, color='orange', linestyle=':', label="M2 Velocity (Right)", linewidth=1.5)
            ax1_twin.set_ylabel("Velocity Ratio", color='orange', fontsize=10)
            ax1_twin.tick_params(axis='y', labelcolor='orange')
            ax1_twin.legend(loc="upper right")
    else:
        ax1.text(0.5, 0.5, "Liquidity Data Missing", ha='center')

    # 2. Overnight Rates Monitor (SOFR vs Fed Funds) - NEW
    ax2 = fig.add_subplot(gs[1])
    if "SOFR" in df.columns and "Fed_Funds" in df.columns:
        sofr = df["SOFR"].dropna()
        dff = df["Fed_Funds"].dropna()
        
        # Align
        common_idx = sofr.index.intersection(dff.index)
        sofr = sofr.loc[common_idx]
        dff = dff.loc[common_idx]
        
        ax2.plot(dff.index, dff, color='black', linestyle=':', label="Fed Funds Effective Rate (Policy)", linewidth=1.5)
        ax2.plot(sofr.index, sofr, color='blue', label="SOFR (Secured Overnight)", linewidth=1)
        
        # Stress Detection (SOFR > DFF + 5bps)
        spread = sofr - dff
        stress_dates = spread[spread > 0.05].index
        
        # Highlight Stress
        for date in stress_dates:
             ax2.axvline(date, color='red', alpha=0.3)
             
        curr_sofr = sofr.iloc[-1]
        curr_dff = dff.iloc[-1]
        
        status = "NORMAL"
        if curr_sofr > curr_dff + 0.05: status = "STRESS (Collateral Shortage)"
        elif curr_sofr < curr_dff - 0.10: status = "EXCESS LIQUIDITY (RRP Floor)"
        
        ax2.set_title(f"2. Overnight Rates Monitor (Plumbing): {curr_sofr:.2f}% vs Fed {curr_dff:.2f}% -> {status}", fontsize=12, weight='bold')
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
        
        # Interpretation
        ax2.text(0.02, 0.6, "Normal: SOFR trades tight to Fed Funds.\nSpike > Fed Funds = Collateral Shortage (Repo Crisis Risk).", 
                 transform=ax2.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'))
    else:
        ax2.text(0.5, 0.5, "Overnight Rate Data Missing (SOFR/DFF)", ha='center')

    # 3. DXY
    ax3 = fig.add_subplot(gs[2])
    dxy_col = "DX-Y.NYB" if "DX-Y.NYB" in df.columns else "UUP"
    if dxy_col in df.columns:
        dxy = df[dxy_col].dropna()
        ma = dxy.rolling(200).mean()
        ax3.plot(dxy.index, dxy, color='green', label="USD Index (DXY)")
        ax3.plot(ma.index, ma, color='black', linestyle='--')
        
        curr = dxy.iloc[-1]
        ma_val = ma.iloc[-1]
        status = "BULLISH" if curr > ma_val else "BEARISH"
        color = 'red' if curr > ma_val else 'green'
        ax3.set_title(f"3. Global Liquidity Wrecking Ball (DXY): {status}", fontsize=12, weight='bold', color=color)
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "DXY Data Missing", ha='center')

    # 4. Inflation Trend
    ax4 = fig.add_subplot(gs[3])
    if "CPI_YoY" in df.columns:
        cpi = df["CPI_YoY"].dropna()
        ax4.plot(cpi.index, cpi, color='purple', label="CPI Inflation (YoY)", linewidth=2)
        ax4.axhline(0.02, color='red', linestyle='--', label="Fed Target (2%)")
        ax4.set_title("4. Inflation Trend (CPI)", fontsize=12, weight='bold')
        ax4.legend(loc="upper left")
        ax4.grid(True, alpha=0.3)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    else:
        ax4.text(0.5, 0.5, "Inflation Data Missing", ha='center')
        
    # 5. Labor Market
    ax5 = fig.add_subplot(gs[4])
    if "Unemployment" in df.columns:
        unrate = df["Unemployment"].dropna()
        unrate_ma = unrate.rolling(12).mean()
        ax5.plot(unrate.index, unrate, color='black', label="Unemployment Rate", linewidth=2)
        ax5.plot(unrate_ma.index, unrate_ma, color='red', linestyle='--', label="12-Month Moving Avg")
        ax5.set_title("5. Labor Market Health (Unemployment)", fontsize=12, weight='bold')
        ax5.legend(loc="upper left")
        ax5.grid(True, alpha=0.3)
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    else:
        ax5.text(0.5, 0.5, "Labor Data Missing", ha='center')

    plt.tight_layout()
    return fig

def plot_forward_models(df: pd.DataFrame, prices: pd.DataFrame) -> plt.Figure:
    """Generates Page: Forward Looking Models (Recession, Regime, Rotation, PCA, Seasonality)."""
    print("   Generating Forward Models Page...")
    plt.style.use('default')
    
    fig = plt.figure(figsize=(14, 18)) 
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
    if regime_data.get("current_state") != "N/A":
        labels = regime_data["labels"]
        dates = regime_data["dates"]
        ax3.scatter(dates, labels, c=labels, cmap='RdYlGn_r', s=10, alpha=0.6)
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(["State 0", "State 1", "State 2"])
        ax3.set_title(f"3. Market Regimes (GMM Clustering)\nCurrent: {regime_data['current_state']}", fontsize=12, weight='bold')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Regime Data Missing", ha='center')
        
    # 4. Sector Rotation Matrix
    ax4 = fig.add_subplot(gs[2, 0])
    rot_data = predict_sector_rotation(prices)
    if rot_data["current_leader"] != "N/A":
        sectors = ["XLI", "XLB", "XLU", "XLF", "XLK", "XLE", "XLV", "XLC", "XLY", "XLP"]
        valid_sectors = [s for s in sectors if s in prices.columns]
        
        # Recalculate basic transitions for bar chart
        monthly_prices = prices[valid_sectors].resample('ME').last()
        monthly_rets = monthly_prices.pct_change().infer_objects(copy=False).dropna()
        leaders = monthly_rets.idxmax(axis=1)
        curr_leader = rot_data["current_leader"]
        
        next_counts = {}
        for prev, curr in zip(leaders[:-1], leaders[1:]):
            if prev == curr_leader:
                next_counts[curr] = next_counts.get(curr, 0) + 1
        
        if next_counts:
            total = sum(next_counts.values())
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
        ax5.set_title(f"5. Systemic Risk Monitor (PCA)\nStatus: {pca_data['status']}", fontsize=12, weight='bold')
        ax5.legend(loc="upper left")
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 1.0)
    else:
        ax5.text(0.5, 0.5, "PCA Data Missing", ha='center')
        
    # 6. Seasonality
    ax6 = fig.add_subplot(gs[3, :])
    seas_data = calculate_seasonality(prices)
    if seas_data["curr_month"] != "N/A":
        # Recalculate monthly seasonality
        spy = prices["SPY"]
        spy_monthly = spy.resample('ME').last().pct_change().infer_objects(copy=False).dropna()
        df_m = pd.DataFrame({"Ret": spy_monthly})
        df_m["Month"] = df_m.index.month
        monthly_stats = df_m.groupby("Month")["Ret"].mean()
        
        import calendar
        month_names = [calendar.month_abbr[i] for i in range(1, 13)]
        colors = ['green' if x > 0 else 'red' for x in monthly_stats]
        
        bars = ax6.bar(month_names, monthly_stats, color=colors, alpha=0.5)
        
        curr_m = dt.date.today().month
        next_m = (curr_m % 12) + 1
        
        # Highlight
        if 0 <= curr_m-1 < 12: 
            bars[curr_m-1].set_alpha(1.0)
            bars[curr_m-1].set_edgecolor('black')
            bars[curr_m-1].set_linewidth(2)
        if 0 <= next_m-1 < 12:
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
    
    # --- Interpretation Box (Bottom) ---
    fig.subplots_adjust(bottom=0.12)
    
    interp_text = "Forward Looking Signals:\n"
    
    # 1. Recession Prob
    if "Recession_Prob" in df.columns:
        curr_prob = df["Recession_Prob"].iloc[-1]
        if curr_prob > 30:
            interp_text += f"• Recession Model: WARNING ({curr_prob:.1f}%). Probability exceeds safety threshold.\n"
        else:
            interp_text += f"• Recession Model: LOW RISK ({curr_prob:.1f}%). Yield curve does not signal imminent downturn.\n"
            
    # 2. Regime
    interp_text += "• Market Regime: Verify with GMM plot. (State 0 = Bull, State 1 = Volatile/Bear).\n"
    
    # 3. Seasonality
    curr_m = dt.date.today().month
    import calendar
    m_name = calendar.month_abbr[curr_m]
    
    interp_text += f"• Seasonality ({m_name}): Check bar chart. Historic avg return provides bias.\n"

    fig.text(0.05, 0.02, interp_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round'))
    return fig

def plot_quant_lab_dashboard(prices: pd.DataFrame) -> plt.Figure:
    """Generates Page: Quant Lab (Vol, Monte Carlo, Greeks)."""
    print("   Generating Quant Lab Page...")
    plt.style.use('default')
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    
    # 1. Volatility Regime
    ax1 = fig.add_subplot(gs[0, :])
    if "SPY" in prices.columns:
        spy_ret = prices["SPY"].pct_change().infer_objects(copy=False).dropna()
        
        # GARCH Fit
        try:
            vol_garch, _ = fit_garch(spy_ret.values * 100)
            vol_garch = vol_garch / 100 * np.sqrt(252)
            garch_series = pd.Series(vol_garch, index=spy_ret.index)
        except:
            garch_series = pd.Series(0, index=spy_ret.index)
            
        vol_hist = spy_ret.rolling(21).std() * np.sqrt(252)
        lookback = 252
        
        ax1.plot(garch_series.index[-lookback:], garch_series.iloc[-lookback:], label="GARCH(1,1) Est", color='purple', linewidth=2)
        ax1.plot(vol_hist.index[-lookback:], vol_hist.iloc[-lookback:], label="Realized (21D)", color='orange', alpha=0.7)
        ax1.set_title("SPY Volatility Regime: GARCH Model vs Realized", fontsize=12, weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "SPY Data Missing", ha='center')

    # 2. Monte Carlo
    ax2 = fig.add_subplot(gs[1, 0])
    if "SPY" in prices.columns:
        spy_ret = prices["SPY"].pct_change().infer_objects(copy=False).dropna()
        S0 = prices["SPY"].iloc[-1]
        mu = spy_ret.mean() * 252
        sigma_sim = spy_ret.std() * np.sqrt(252)
        T_sim = 30/252.0 # 30 Days
        dt_sim = 1/252.0
        n_paths = 100
        
        try:
            time, paths = simulate_gbm(S0, mu, sigma_sim, T_sim, dt_sim, n_paths)
            for i in range(n_paths):
                ax2.plot(time*252, paths[i], color='cyan', alpha=0.1)
            ax2.plot(time*252, paths.mean(axis=0), color='blue', linewidth=2, label="Mean Path")
            ax2.set_title(f"Monte Carlo: SPY 30-Day Projection ({n_paths} Paths)", fontsize=12, weight='bold')
            ax2.set_xlabel("Days Ahead")
            ax2.set_ylabel("Price")
            ax2.grid(True, alpha=0.3)
        except Exception as e:
            ax2.text(0.5, 0.5, f"Sim Error: {e}", ha='center')
    else:
        ax2.text(0.5, 0.5, "Data Missing", ha='center')

    # 3. ATM Greeks
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    if "SPY" in prices.columns:
        S = prices["SPY"].iloc[-1]
        K = S
        T = 30/365.0
        r = 0.045
        sigma = 0.15 # Fallback or calc
        if "sigma_sim" in locals(): sigma = sigma_sim
        
        greeks = calculate_greeks(S, K, T, r, sigma, "call")
        bsm_price = black_scholes_merton(S, K, T, r, sigma, "call")
        
        greeks_data = [
            ["Metric", "Value"],
            ["ATM Call Price (30D)", f"${bsm_price:.2f}"],
            ["Delta", f"{greeks.get('Delta', 0):.3f}"],
            ["Gamma", f"{greeks.get('Gamma', 0):.4f}"],
            ["Theta (Daily)", f"{greeks.get('Theta', 0):.3f}"],
            ["Vega (1%)", f"{greeks.get('Vega', 0):.3f}"],
            ["Rho", f"{greeks.get('Rho', 0):.3f}"]
        ]
        
        table = ax3.table(cellText=greeks_data, loc='center', cellLoc='center', colWidths=[0.5, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        ax3.set_title("SPY ATM Option Greeks (Theoretical)", fontsize=12, weight='bold')
    else:
        ax3.text(0.5, 0.5, "Data Missing", ha='center')

    plt.tight_layout()
    
    # --- Interpretation Box (Bottom) ---
    fig.subplots_adjust(bottom=0.15)
    
    interp_text = "Quant Lab Insights:\n"
    
    # 1. Vol
    if "SPY" in prices.columns:
        curr_vol = vol_hist.iloc[-1]
        interp_text += f"• Volatility Regime (21D Realized): {curr_vol:.1f}%. "
        if curr_vol < 12: interp_text += "Low Volatility (Complacency/Bull Trend).\n"
        elif curr_vol > 25: interp_text += "High Volatility (Fear/Crash Risk).\n"
        else: interp_text += "Normal Volatility.\n"
        
    # 2. Monte Carlo
    if "SPY" in prices.columns:
        mean_path = paths.mean(axis=0)[-1]
        chg = (mean_path/S0 - 1) * 100
        interp_text += f"• Monte Carlo Projection (Mean): {chg:.1f}% expected return over 30 days.\n"
        
    # 3. Greeks
    interp_text += "• Option Greeks: Delta measures directional exposure. Vega measures sensitivity to volatility spikes.\n"

    fig.text(0.05, 0.02, interp_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round'))
    return fig

def plot_alpha_factors_page(prices: pd.DataFrame, macro: pd.DataFrame, alpha_data: Dict) -> plt.Figure:
    """Generates Page 6: Institutional Alpha Factors."""
    print("   Generating Alpha Factors Page...")
    plt.style.use('default')
    
    fig = plt.figure(figsize=(14, 16)) # Increased height
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1, 1])
    
    # 1. Net Liquidity vs SPY
    ax1 = fig.add_subplot(gs[0])
    nl_data = alpha_data.get("net_liquidity", {})
    if not nl_data.get("series", pd.Series()).empty:
        nl = nl_data["series"]
        spy = prices["SPY"].reindex(nl.index).ffill()
        
        # Plot Net Liquidity (Left)
        color_nl = 'darkblue'
        ax1.plot(nl.index, nl, color=color_nl, linewidth=2, label="Net Liquidity ($Trillion)")
        ax1.set_ylabel("Net Liquidity ($T)", color=color_nl, fontsize=12)
        ax1.tick_params(axis='y', labelcolor=color_nl)
        
        # Plot SPY (Right)
        ax1_twin = ax1.twinx()
        color_spy = 'black'
        ax1_twin.plot(spy.index, spy, color=color_spy, linestyle='--', alpha=0.6, label="S&P 500 (Right)")
        ax1_twin.set_ylabel("S&P 500 Price", color=color_spy, fontsize=12)
        ax1_twin.tick_params(axis='y', labelcolor=color_spy)
        
        # Status Validation
        status = nl_data.get("status", "N/A")
        raw_val = nl_data.get("latest", 0)
        ax1.set_title(f"1. The Real Liquidity Engine (Fed - TGA - RRP)\nCurrent: ${raw_val:.2f}T | Trend: {status}", fontsize=14, weight='bold')
        
        # Combined Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "Net Liquidity Data Unavailable", ha='center')

    # 2. Volatility Term Structure (Crash Signal)
    ax2 = fig.add_subplot(gs[1])
    vol_data = alpha_data.get("vol_structure", {})
    if not vol_data.get("ratio", pd.Series()).empty:
        ratio = vol_data["ratio"]
        ax2.plot(ratio.index, ratio, color='purple', label="VIX / VIX3M Ratio")
        
        # Thresholds
        ax2.axhline(1.0, color='black', linestyle='-', linewidth=1, label="Contango/Backwardation Flip")
        ax2.fill_between(ratio.index, ratio, 1.0, where=(ratio > 1.0), color='red', alpha=0.3, label="Crash Risk (>1.0)")
        ax2.fill_between(ratio.index, ratio, 1.0, where=(ratio <= 1.0), color='green', alpha=0.1, label="Normal (<1.0)")
        
        curr_sig = vol_data.get("signal", "N/A")
        color_sig = 'red' if "CRASH" in curr_sig else 'green'
        ax2.set_title(f"2. Volatility Term Structure (Crash Signal)\nStatus: {curr_sig}", fontsize=12, weight='bold', color=color_sig)
        ax2.set_ylabel("VIX / VIX3M Ratio")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Volatility Data Missing", ha='center')

    # 3. Tail Risk (SKEW)
    ax3 = fig.add_subplot(gs[2])
    skew_data = alpha_data.get("tail_risk", {})
    if not skew_data.get("series", pd.Series()).empty:
        skew = skew_data["series"]
        ax3.plot(skew.index, skew, color='darkred', label="CBOE SKEW Index")
        
        # Zones
        ax3.axhline(135, color='red', linestyle='--', label="High Risk (>135)")
        ax3.axhline(115, color='green', linestyle='--', label="Complacency (<115)")
        
        curr_skew = skew_data.get("signal", "N/A")
        ax3.set_title(f"3. Tail Risk Monitor (Whale Positioning)\nStatus: {curr_skew}", fontsize=12, weight='bold')
        ax3.set_ylabel("SKEW Index")
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "SKEW Data Missing", ha='center')

    plt.tight_layout()
    
    # --- Interpretation Box (Bottom) ---
    fig.subplots_adjust(bottom=0.12)
    
    interp_text = "Institutional Flows & Risk:\n"
    
    # 1. Net Liquidity
    nl_lat = alpha_data.get("net_liquidity", {}).get("latest", 0)
    interp_text += f"• Net Liquidity: ${nl_lat:.2f}T. Tracks Fed Balance Sheet - TGA - RRP. Rising = Asset Support.\n"
    
    # 2. VIX Term Structure
    v_rat = alpha_data.get("vol_structure", {}).get("ratio")
    if not isinstance(v_rat, pd.Series): v_rat = pd.Series([0]) 
    curr_v = v_rat.iloc[-1] if not v_rat.empty else 0
    
    if curr_v > 1.0:
        interp_text += f"• Vol Term Structure: BACKWARDATION ({curr_v:.2f}). CRASH WARNING. Immediate fear > future fear.\n"
    else:
        interp_text += f"• Vol Term Structure: CONTANGO ({curr_v:.2f}). Normal market structure.\n"
        
    # 3. SKEW
    s_val = alpha_data.get("tail_risk", {}).get("series")
    if not isinstance(s_val, pd.Series): s_val = pd.Series([0])
    curr_s = s_val.iloc[-1] if not s_val.empty else 0
    
    if curr_s > 135: interp_text += f"• Tail Risk (SKEW): HIGH ({curr_s:.0f}). Whales are hedging against a crash.\n"
    elif curr_s < 115: interp_text += f"• Tail Risk (SKEW): COMPLACENT ({curr_s:.0f}). Vulnerable to shocks.\n"
    else: interp_text += f"• Tail Risk (SKEW): NORMAL ({curr_s:.0f}).\n"

    fig.text(0.05, 0.02, interp_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round'))
    return fig

def plot_cross_asset_page(prices: pd.DataFrame, corr_data: Dict) -> plt.Figure:
    """Generates Page 9: Cross-Asset Regime."""
    print("   Generating Cross-Asset Correlation Page...")
    plt.style.use('default')
    
    fig = plt.figure(figsize=(14, 16)) # Increased height 
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.8])
    
    # 1. Correlation Matrix Heatmap
    ax1 = fig.add_subplot(gs[0])
    matrix = corr_data.get("matrix", pd.DataFrame())
    
    if not matrix.empty:
        cax = ax1.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax, ax=ax1, fraction=0.046, pad=0.04)
        
        # Labels
        labels = matrix.columns
        ax1.set_xticks(np.arange(len(labels)))
        ax1.set_yticks(np.arange(len(labels)))
        ax1.set_xticklabels(labels, fontsize=10)
        ax1.set_yticklabels(labels, fontsize=10)
        
        # Annotate
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = matrix.iloc[i, j]
                text_color = "white" if abs(val) > 0.5 else "black"
                ax1.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=12, weight='bold')
                
        ax1.set_title("1. Cross-Asset Correlation Matrix (30-Day Rolling)\nGreen/Red = Diversification | Dark = High Correlation", fontsize=14, weight='bold')
    else:
        ax1.text(0.5, 0.5, "Insufficient Data for Correlation Matrix", ha='center')

    # 2. SPY vs TLT Rolling Correlation (Regime Check)
    ax2 = fig.add_subplot(gs[1])
    rolling = corr_data.get("spy_tlt_rolling", pd.Series())
    
    if not rolling.empty:
        ax2.plot(rolling.index, rolling, color='black', linewidth=1.5, label="SPY vs TLT (6M Rolling)")
        
        # Zones
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.axhline(0.5, color='red', linestyle=':', label="danger (>0.5)")
        ax2.axhline(-0.5, color='green', linestyle=':', label="Diversified (<-0.5)")
        
        # Fill
        ax2.fill_between(rolling.index, rolling, 0.5, where=(rolling > 0.5), color='red', alpha=0.3, label="Inflation/Rate Risk")
        ax2.fill_between(rolling.index, rolling, -0.5, where=(rolling < -0.5), color='green', alpha=0.2, label="Deflation/Growth Risk")
        
        curr = rolling.iloc[-1]
        regime = "INFLATION FEAR" if curr > 0.5 else ("DEFLATION/GROWTH FEAR" if curr < -0.5 else "NORMAL DIVERSIFICATION")
        
        ax2.set_title(f"2. Stock-Bond Correlation Regime\nCurrent: {curr:.2f} -> {regime}", fontsize=12, weight='bold')
        ax2.set_ylabel("Correlation")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "SPY/TLT Data Missing", ha='center')

    plt.tight_layout()
    
    # --- Interpretation Box (Bottom) ---
    fig.subplots_adjust(bottom=0.15)
    
    interp_text = "Cross-Asset Regime:\n"
    interp_text += "• Correlation Matrix: Dark Red = Assets moving together (Systemic Risk). Blue = Diversification working.\n"
    
    # SPY vs TLT
    rolling = corr_data.get("spy_tlt_rolling", pd.Series())
    if not rolling.empty:
        curr_c = rolling.iloc[-1]
        interp_text += f"• Stock/Bond Correlation: {curr_c:.2f}. "
        if curr_c > 0.5: interp_text += "HIGHLY CORRELATED. Bonds will NOT protect portfolios. Inflation risk dominant.\n"
        elif curr_c < -0.5: interp_text += "NEGATIVELY CORRELATED. Magnificent diversification. Growth risk dominant.\n"
        else: interp_text += "UNCORRELATED. Standard diversification environment.\n"

    fig.text(0.05, 0.02, interp_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round'))
    return fig

def plot_efficient_frontier_page(optimization_data: Dict) -> plt.Figure:
    """Generates Page 10: Portfolio Optimization Lab."""
    print("   Generating Efficient Frontier Page...")
    plt.style.use('default')
    
    if not optimization_data or "results" not in optimization_data:
        return None
        
    results = optimization_data["results"]
    max_sharpe = optimization_data["max_sharpe"]
    min_vol = optimization_data["min_vol"]
    assets = optimization_data["assets"]
    
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 0.5])
    
    # 1. Efficient Frontier Scatter
    ax1 = fig.add_subplot(gs[0])
    sc = ax1.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='o', s=10, alpha=0.5)
    fig.colorbar(sc, ax=ax1, label="Sharpe Ratio")
    
    # Mark Max Sharpe
    ax1.scatter(max_sharpe["metrics"][1], max_sharpe["metrics"][0], marker='*', color='r', s=500, label=f"Max Sharpe ({max_sharpe['metrics'][2]:.2f})")
    
    # Mark Min Vol
    ax1.scatter(min_vol["metrics"][1], min_vol["metrics"][0], marker='o', color='b', s=200, label=f"Min Volatility (Vol: {min_vol['metrics'][1]:.1%})", edgecolors='white', linewidth=2)
    
    ax1.set_title("1. THE EFFICIENT FRONTIER (5000 Simulated Portfolios)\nRisk vs Return Trade-off", fontsize=16, weight='bold')
    ax1.set_xlabel("Annualized Volatility (Risk)")
    ax1.set_ylabel("Annualized Return")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # 2. Optimal Allocation Table
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    
    table_data = []
    # Header
    table_data.append(["Asset", "Min Volatility", "Max Sharpe (Optimal)"])
    
    for asset in assets:
        min_w = min_vol["weights"].get(asset, 0)
        max_w = max_sharpe["weights"].get(asset, 0)
        # Fix formatting for zero weights
        min_str = f"{min_w:.1%}" if min_w > 0.001 else "-"
        max_str = f"{max_w:.1%}" if max_w > 0.001 else "-"
        table_data.append([asset, min_str, max_str])
        
    table = ax2.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.2, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2.0)
    
    # Style Header
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#404040')
            
    ax2.set_title("2. OPTIMAL ASSET ALLOCATION (Mean-Variance)", fontsize=14, weight='bold')
    
    plt.tight_layout()
    
    # --- Interpretation Box (Bottom) ---
    fig.subplots_adjust(bottom=0.15)
    
    interp_text = "Portfolio Optimization Insights:\n"
    # Max Sharpe Asset
    if "weights" in max_sharpe:
         best_asset = max(max_sharpe["weights"], key=max_sharpe["weights"].get)
         best_w = max_sharpe["weights"][best_asset]
         interp_text += f"• Max Sharpe Portfolio: Heaviest allocation is {best_asset} ({best_w:.1%}). Maximizes risk-adjusted return.\n"
         
    # Min Vol Asset 
    if "weights" in min_vol:
         safe_asset = max(min_vol["weights"], key=min_vol["weights"].get)
         safe_w = min_vol["weights"][safe_asset]
         interp_text += f"• Min Volatility Portfolio: Heaviest allocation is {safe_asset} ({safe_w:.1%}). Focuses on capital preservation.\n"
         
    interp_text += "• Efficient Frontier: Portfolios on the top-left edge offer the highest return for a given level of risk."

    fig.text(0.05, 0.02, interp_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round'))
    return fig

def plot_predictive_models_page(df: pd.DataFrame, internals: Dict, recession_prob: pd.Series) -> plt.Figure:
    """Page 11: Predictive Analytics (Recession & Internals)."""
    print("   Generating Predictive Models Page...")
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("PREDICTIVE ANALYTICS: MACRO & MARKET INTERNALS", fontsize=16, weight='bold', y=0.98)
    
    # Grid: 2 Rows. Top = Recession. Bottom = Internals.
    gs = fig.add_gridspec(2, 1, hspace=0.3)
    
    # --- Panel 1: Recession Probability ---
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Combine Model & actual Recessions if available?
    # We'll just plot the Probit Probability
    if not recession_prob.empty:
        prob = recession_prob.iloc[-1260:] # Last 5 years
        ax1.plot(prob.index, prob.values, color='red', linewidth=2, label="Recession Prob (12M Fwd)")
        ax1.fill_between(prob.index, prob.values, 0, color='red', alpha=0.3)
        
        # Add Threshold Line
        ax1.axhline(30, color='black', linestyle='--', alpha=0.5, label="Warning Threshold (30%)")
        
        curr_prob = prob.iloc[-1]
        ax1.set_title(f"NY Fed Recession Probability Model (Current: {curr_prob:.1f}%)", fontsize=12, weight='bold')
        ax1.set_ylabel("Probability (%)")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Annotate
        if curr_prob > 30:
            ax1.text(prob.index[-1], curr_prob, " HIGH RISK", color='red', weight='bold')
    else:
        ax1.text(0.5, 0.5, "Data Unavailable", ha='center')
        
    # --- Panel 2: Market Internals (Leading Indicators) ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("Market Internals: Leading Ratios (Normalized)", fontsize=12, weight='bold')
    
    for name, data in internals.items():
        series = data["Series"].dropna().iloc[-252:] # Last 1 Year
        if series.empty: continue
        
        # Normalize to start at 0%
        norm_series = (series / series.iloc[0] - 1) * 100
        
        if name == "Defensive":
             # Invert Defensive for visualization? No, let's keep as is but specific color
             ax2.plot(norm_series.index, norm_series.values, label=name, linestyle='--', alpha=0.7)
        else:
             ax2.plot(norm_series.index, norm_series.values, label=name, linewidth=1.5)
             
    ax2.set_ylabel("Change (%)")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # --- Interpretation Box (Bottom) ---
    fig.subplots_adjust(bottom=0.15, top=0.93)
    
    interp_text = "Predictive Analytics:\n"
    # Recession
    if not recession_prob.empty:
         rp = recession_prob.iloc[-1]
         if rp > 30: interp_text += f"• Recession Risk: HIGH ({rp:.1f}%). Yield curve signals economic contraction ahead.\n"
         else: interp_text += f"• Recession Risk: LOW ({rp:.1f}%). No immediate curve-driven signal.\n"
         
    # Internals
    interp_text += "• Leading Indicators: Watch for 'Cyclical' vs 'Defensive' divergence. If Defensives lead, risk-off/recession is likely."
    
    fig.text(0.05, 0.02, interp_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round'))
    return fig
        
    # Add Explainer Text box
    # Footer (Enhanced Commentary)
    text_content = (
        "INTERPRETATION GUIDE:\n"
        "1. RECESSION MODEL: Uses the Yield Curve (10Y-3M) to predict recession chance in next 12 months.\n"
        "   - >30% = Warning. >50% = High Probability.\n"
        "2. MARKET INTERNALS (Leading Indicators):\n"
        "   - Risk Appetite (XLY/XLP): Rising means investors prefer Growth/Cyclicals (Bullish).\n"
        "   - Breadth (RSP/SPY): Rising means broad participation (Healthy).\n"
        "   - Credit (HYG/IEF): Rising means Credit Markets are ignoring risk (Bullish/Complacent)."
    )
    fig.text(0.05, 0.02, text_content, fontsize=9, family='monospace', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='darkblue', linewidth=1.5))
             
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2) # Make room for text
    return fig

def plot_monte_carlo_cone(prices: pd.DataFrame, ticker: str = "SPY", days: int = 60, n_sims: int = 1000) -> plt.Figure:
    """Page 12: Quant Lab Simulation (Brownian Motion Cone)."""
    print(f"   Generating Monte Carlo Cone for {ticker}...")
    
    if ticker not in prices.columns: return None
    
    # 1. Calibrate Model
    series = prices[ticker].dropna()
    rets = series.pct_change().dropna()
    
    S0 = series.iloc[-1]
    mu = rets.mean() * 252
    sigma = rets.std() * np.sqrt(252)
    
    # 2. Simulate
    T = days / 252.0
    dt_step = 1 / 252.0
    
    time_sim, paths = simulate_gbm(S0, mu, sigma, T, dt_step, n_sims)
    
    # 3. Plot
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(f"QUANT LAB: MONTE CARLO SIMULATION ({ticker})", fontsize=16, weight='bold', y=0.98)
    
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    
    # Historical Context (6 Months)
    hist_window = 126
    hist_dates = series.index[-hist_window:]
    hist_prices = series.values[-hist_window:]
    
    ax1.plot(hist_dates, hist_prices, color='black', linewidth=2, label="Historical Price")
    
    # Generate Future Dates
    last_date = series.index[-1]
    future_dates = [last_date + dt.timedelta(days=i) for i in range(len(time_sim))] 
    # Note: simulate_gbm returns time steps 0..T. 0 is today.
    # We need to map 'business days' ideally, but T+timedelta is fine for viz.
    # Let's use business days logic for cleaner x-axis if possible, or just standard days
    future_dates = pd.date_range(start=last_date, periods=len(time_sim), freq='B')
    
    # Plot Paths (First 100)
    for i in range(min(100, n_sims)):
        ax1.plot(future_dates, paths[i, :], color='gray', alpha=0.1, linewidth=0.5)
        
    # Percentiles
    p5 = np.percentile(paths, 5, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    
    ax1.plot(future_dates, p50, color='blue', linewidth=2, label="Median Path (P50)")
    ax1.plot(future_dates, p5, color='red', linestyle='--', linewidth=1.5, label="95% Confidence Interval")
    ax1.plot(future_dates, p95, color='red', linestyle='--', linewidth=1.5)
    
    ax1.fill_between(future_dates, p5, p95, color='blue', alpha=0.1)
    
    ax1.set_title(f"Geometric Brownian Motion: {days}-Day Forecast Cone", fontsize=12, weight='bold')
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Annotate Final Range
    final_p5 = p5[-1]
    final_p95 = p95[-1]
    ax1.text(future_dates[-1], final_p95, f"${final_p95:.0f}", color='red', va='bottom')
    ax1.text(future_dates[-1], final_p5, f"${final_p5:.0f}", color='red', va='top')
    
    # --- Panel 2: Distribution of Returns ---
    ax2 = fig.add_subplot(gs[1])
    final_prices = paths[:, -1]
    rets_sim = (final_prices / S0) - 1
    
    ax2.hist(rets_sim, bins=50, color='navy', alpha=0.7, density=True)
    ax2.axvline(0, color='black', linestyle='-')
    
    # Stats
    win_prob = (rets_sim > 0).mean()
    exp_val = rets_sim.mean()
    
    ax2.set_title(f"Distribution of Simulated Returns (Win Probability: {win_prob:.1%})", fontsize=12, weight='bold')
    ax2.set_xlabel("Return (%)")
    ax2.set_ylabel("Probability Density")
    ax2.grid(True, alpha=0.3)
    
    # Text Box
    # Footer (Enhanced Commentary)
    text_content = (
        "INTERPRETATION GUIDE:\n"
        "1. THE CONE: Projecting future price paths using Geometric Brownian Motion (Random Walk + Drift).\n"
        "2. PROBABILITY BANDS: 68% (1 Sigma) and 95% (2 Sigma) confidence intervals.\n"
        "3. USE CASE: Setting realistic profit targets (Upper Band) and stop-losses (Lower Band) based on volatility.\n"
        "4. PARAMETERS: Drift = Expected Return, Volatility = Average Risk."
    )
    fig.text(0.05, 0.02, text_content, fontsize=9, family='monospace', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='darkblue', linewidth=1.5))
             
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    
    # --- Interpretation Box ---
    interp_text = (
        "INTERPRETATION GUIDE:\n"
        "• THE CONE: Shows the range of probable price paths based on drift (trend) and diffusion (volatility).\n"
        "• MEDIAN PATH (Blue): The 'Base Case' projection.\n"
        "• 95% CONFIDENCE (Light Blue): Outlier scenarios. If price hits the edge, it is empirically overextended.\n"
        "• USE CASE: Validates price targets. If your target is outside the cone, it requires an extreme event."
    )
    fig.text(0.05, 0.02, interp_text, fontsize=9, family='monospace', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='darkblue', linewidth=1.5))
             
    return fig

def plot_stochastic_page(prices: pd.DataFrame, ticker: str = "SPY") -> plt.Figure:
    """Page 13: Stochastic Volatility & Regime Switching."""
    print(f"   Generating Stochastic Models Page for {ticker}...")
    
    if ticker not in prices.columns: return None
    series = prices[ticker].dropna()
    
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(f"STOCHASTIC MODELLING & REGIME DETECTION ({ticker})", fontsize=16, weight='bold', y=0.98)
    
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1, 1], hspace=0.4)
    
    # 1. Heston Calib (Simplified)
    rets = series.pct_change().dropna()
    S0 = series.iloc[-1]
    
    # Simplified manual params for visual demo
    # Real Heston calibrating is complex optimization, we use illustrative parameters
    v0 = rets.var() * 252 # Annualized variance
    mu = 0.08
    kappa = 2.0  # Mean reversion speed
    theta = 0.04 # Long run variance (20% vol squared)
    xi = 0.3     # Vol of Vol
    rho = -0.7   # Leverage effect
    T = 1.0      # 1 Year
    n_sims = 100 
    
    time_sim, S, v = simulate_heston(S0, v0, mu, kappa, theta, xi, rho, T, 252, n_sims)
    
    # 2. HMM Regime Fit
    hmm_res = fit_markov_regime_switching(series.pct_change().dropna(), k_regimes=2)
    
    # Plot
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("QUANT LAB: ADVANCED STOCHASTIC MODELS", fontsize=16, weight='bold', y=0.98)
    
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1, 1], hspace=0.4)
    
    # Panel 1: Heston Price Paths
    ax1 = fig.add_subplot(gs[0])
    # Show history + future
    # Just show future paths for clarity
    future_dates = pd.date_range(start=series.index[-1], periods=len(time_sim), freq='B')
    
    for i in range(min(50, n_sims)): # Plot 50 paths
        ax1.plot(future_dates, S[i, :], color='blue', alpha=0.15, linewidth=0.5)
        
    ax1.set_title(f"1. Heston Stochastic Volatility Model (1-Year Simulation)", fontsize=12, weight='bold')
    ax1.set_ylabel("Price ($)")
    
    # Panel 2: Heston Volatility Paths
    ax2 = fig.add_subplot(gs[1])
    for i in range(min(50, n_sims)):
        vol_path = np.sqrt(v[i, :]) * 100 # Convert variance -> vol %
        ax2.plot(future_dates, vol_path, color='orange', alpha=0.15, linewidth=0.5)
        
    ax2.set_title("2. Simulated Volatility Paths (Stochastic Process)", fontsize=12, weight='bold')
    ax2.set_ylabel("Volatility (%)")
    
    # Panel 3: Markov Regime Probabilities
    ax3 = fig.add_subplot(gs[2])
    probs = hmm_res.get("probs", pd.DataFrame())
    
    if not probs.empty:
        # Plot only last 500 days for visibility
        subset = probs.iloc[-500:]
        # Stacked area
        ax3.stackplot(subset.index, subset.T.values, labels=subset.columns, alpha=0.6, colors=['green', 'red'])
        
        curr_regime = "Unknown"
        if not subset.empty:
            curr_regime = subset.iloc[-1].idxmax()
            
        ax3.set_title(f"3. Markov Regime Switching (Current: {curr_regime})", fontsize=12, weight='bold')
        ax3.set_ylabel("Probability")
        ax3.legend(loc='upper left')
    else:
        ax3.text(0.5, 0.5, "HMM Fit Failed (Insufficient Data)", ha='center')
        
    # Footer (Enhanced Commentary)
    text_content = (
        "INTERPRETATION GUIDE:\n"
        "1. HESTON MODEL: Simulates Stochastic Volatility (Vol is not constant, it's a process).\n"
        "2. REGIME SWITCHING (HMM): Detects 'Calm' (Low Vol) vs 'Turbulent' (High Vol) market states.\n"
        "3. USE CASE: Calibrating options strategies. Buy Volatility when regime switches to 'Turbulent'."
    )
    fig.text(0.05, 0.02, text_content, fontsize=9, family='monospace', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='darkblue', linewidth=1.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    return fig

def plot_mean_reversion_page(prices: pd.DataFrame) -> plt.Figure:
    """Page 14: Mean Reversion (Ornstein-Uhlenbeck)."""
    print("   Generating Mean Reversion Page...")
    
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("QUANT LAB: MEAN REVERSION (OU PROCESS)", fontsize=16, weight='bold', y=0.98)
    
    gs = fig.add_gridspec(2, 1, hspace=0.3)
    
    # --- PAIR 1: YIELD CURVE (IEF vs SHY) ---
    ax1 = fig.add_subplot(gs[0])
    pair1_name = "Yield Curve Spread (IEF 7-10Y - SHY 1-3Y)"
    if "IEF" in prices.columns and "SHY" in prices.columns:
        s1 = prices["IEF"]
        s2 = prices["SHY"]
        # Normalize roughly or just take raw spread? Prices are different scales (approx $94 vs $81).
        # Better: Log Spread or Ratio. Let's use Ratio for stationarity.
        spread = np.log(s1 / s2)
        
        # Fit OU
        ou_params = fit_ou_process(spread)
        theta = ou_params.get("theta", 0)
        mu = ou_params.get("mu", 0)
        sigma = ou_params.get("sigma", 0)
        hl = ou_params.get("half_life", 0)
        
        # Plot
        ax1.plot(spread.index, spread.values, color='black', label="Log Spread (IEF/SHY)")
        ax1.axhline(mu, color='blue', linestyle='--', label=f"Long Run Mean ({mu:.4f})")
        
        # Sigma Bands
        if not np.isnan(sigma) and theta > 0:
            # Stationary variance = sigma^2 / (2*theta)
            std_dev = sigma / np.sqrt(2*theta)
            ax1.axhline(mu + 2*std_dev, color='red', linestyle=':', label="+2 Sigma")
            ax1.axhline(mu - 2*std_dev, color='red', linestyle=':', label="-2 Sigma")
            
            # Current Z-Score
            curr = spread.iloc[-1]
            z_score = (curr - mu) / std_dev
            ax1.set_title(f"1. {pair1_name}\nMean Reversion Speed (Theta): {theta:.2f} | Half-Life: {hl:.1f} Days | Current Z-Score: {z_score:.2f}", fontsize=12, weight='bold')
            
            # Annotate Trade Signal
            if z_score > 2.0:
                 ax1.text(spread.index[-1], curr, " OVERBOUGHT (Short Spread)", color='red', weight='bold')
            elif z_score < -2.0:
                 ax1.text(spread.index[-1], curr, " OVERSOLD (Long Spread)", color='green', weight='bold')
        
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "Data Missing (IEF/SHY)", ha='center')
        
    # --- PAIR 2: USD vs YEN (UUP vs FXY) ---
    ax2 = fig.add_subplot(gs[1])
    pair2_name = "USD/JPY Proxy (UUP - FXY)" # Spread between Dollar ETF and Yen ETF
    if "UUP" in prices.columns and "FXY" in prices.columns:
        # FXY is Yen inverted (Yen strength). UUP is Dollar strength.
        # Just use log ratio again.
        spread2 = np.log(prices["UUP"] / prices["FXY"])
        
        # Fit OU
        ou_params2 = fit_ou_process(spread2)
        theta2 = ou_params2.get("theta", 0)
        mu2 = ou_params2.get("mu", 0)
        sigma2 = ou_params2.get("sigma", 0)
        hl2 = ou_params2.get("half_life", 0)
        
        ax2.plot(spread2.index, spread2.values, color='purple', label="Log Spread (UUP/FXY)")
        ax2.axhline(mu2, color='blue', linestyle='--', label="Mean")
        
        if not np.isnan(sigma2) and theta2 > 0:
             std_dev2 = sigma2 / np.sqrt(2*theta2)
             ax2.axhline(mu2 + 2*std_dev2, color='red', linestyle=':')
             ax2.axhline(mu2 - 2*std_dev2, color='red', linestyle=':')
             
             curr2 = spread2.iloc[-1]
             z2 = (curr2 - mu2) / std_dev2
             
             ax2.set_title(f"2. {pair2_name}\nMean Reversion Speed (Theta): {theta2:.2f} | Half-Life: {hl2:.1f} Days | Current Z-Score: {z2:.2f}", fontsize=12, weight='bold')
        else:
             ax2.set_title(f"2. {pair2_name} (Trending / Non-Stationary)", fontsize=12, weight='bold')
             
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Data Missing (UUP/FXY)", ha='center')

    # Footer (Enhanced Commentary)
    text_content = (
        "INTERPRETATION GUIDE:\n"
        "1. ORNSTEIN-UHLENBECK (OU): Models Mean-Reverting assets (Spreads, Pairs).\n"
        "2. Z-SCORE: Measures distance from the mean. > 2.0 is statistically stretched (95% confidence).\n"
        "3. SIGNAL: High Z-Score + High Mean Reversion Speed (Theta) = Strong probability of snap-back."
    )
    fig.text(0.05, 0.02, text_content, fontsize=9, family='monospace', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='darkblue', linewidth=1.5))
             
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20) # Increase bottom margin
    return fig

def plot_microstructure_page(prices: pd.DataFrame, ticker: str = "SPY") -> plt.Figure:
    """Page 15: Jump Diffusion & Hawkes Microstructure."""
    print("   Generating Microstructure & Jumps Page...")
    
    if ticker not in prices.columns: return None
    
    idx = prices.index
    series = prices[ticker].dropna()
    S0 = series.iloc[-1]
    
    # --- 1. Merton Simulation ---
    # Scenarios: "Fat Tail" risks
    mu = 0.08
    sigma = 0.15
    # Crash Params
    lambda_jump = 2.0  # 2 jumps per year on average
    mu_jump = -0.10    # Average jump is -10%
    sigma_jump = 0.05  # Std dev of jump
    T = 1.0
    n_sims = 100
    
    time, S = simulate_merton_jump(S0, mu, sigma, T, 252, n_sims, lambda_jump, mu_jump, sigma_jump)
    
    # --- 2. Hawkes Simulation ---
    # Simulate Order Flow / Volatility Clustering
    # mu (base) = 1.0 events/day
    # alpha (excitation) = 0.8
    # beta (decay) = 1.2
    h_mu = 1.0
    h_alpha = 0.8
    h_beta = 1.2
    
    time_h, intensity, events = simulate_hawkes_intensity(h_mu, h_alpha, h_beta, 100, 1000)
    
    # Plot
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("QUANT LAB: JUMPS & MARKET MICROSTRUCTURE", fontsize=16, weight='bold', y=0.98)
    
    gs = fig.add_gridspec(2, 1, hspace=0.3)
    
    # Panel 1: Merton Jumps
    ax1 = fig.add_subplot(gs[0])
    future_dates = pd.date_range(start=idx[-1], periods=len(time), freq='B')
    
    for i in range(min(50, n_sims)):
        ax1.plot(future_dates, S[i, :], color='blue', alpha=0.1, linewidth=0.5)
        
    ax1.set_title(f"1. Merton Jump Diffusion (Simulating 'Fat Tail' Risks)\nParams: {lambda_jump} Jumps/Yr, Mean Size {mu_jump:.0%} (Crash Scenarios)", fontsize=12, weight='bold')
    ax1.set_ylabel("Price ($)")
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Hawkes Intensity
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(time_h, intensity, color='purple', linewidth=1.5, label="Intensity (Event Arrival Rate)")
    
    # Mark events
    # events array is timepoints
    # We used direct simulation in stochastics returning simple arrays
    # if events is array of times:
    if len(events) > 0:
        y_ev = np.full_like(events, 0.5) # Plot dots at bottom? Or verify return type
        # My stochastics.simulate_hawkes returned time, intensity, event_times (array)
        # Check signature: returns time, intensity, event_times
        # So "events" here is event_times.
        ax2.scatter(events, np.full_like(events, intensity.min()), color='red', marker='|', alpha=0.6, label="Event (Order/shock)")
        
    ax2.set_title("2. Hawkes Process (Self-Exciting Clustering)\nModeling 'Volatility Clustering' or 'Flash Crashes'", fontsize=12, weight='bold')
    ax2.set_ylabel("Intensity (lambda)")
    ax2.set_xlabel("Time (Arbitrary Units)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Footer (Enhanced Commentary)
    text_content = (
        "INTERPRETATION GUIDE:\n"
        "1. MERTON JUMPS: Models sudden 'Crashes' (Jumps) that normal models miss. Shows 'Gap Risk'.\n"
        "2. HAWKES PROCESS: Models 'Self-Excitement' (Clustering). One shock triggers others (Feedback Loops).\n"
        "3. APPLICATION: Stress-testing portfolios against 'Black Swans' and 'Flash Crashes'."
    )
    fig.text(0.05, 0.02, text_content, fontsize=9, family='monospace', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='darkblue', linewidth=1.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20) # Increase bottom margin
    return fig

def plot_antifragility_page(prices: pd.DataFrame, ticker: str = "SPY") -> plt.Figure:
    """Page 16: Taleb Anti-Fragility & Tail Risk."""
    print("   Generating Anti-Fragility Analysis Page...")
    
    if ticker not in prices.columns: return None
    
    series = prices[ticker].dropna()
    returns = series.pct_change().dropna()
    
    metrics = calculate_antifragility_metrics(returns)
    skew = metrics.get("skew", 0)
    kurt = metrics.get("kurtosis", 0)
    taleb_ratio = metrics.get("taleb_ratio", 0)
    status = metrics.get("status", "N/A")
    
    # Plot
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("QUANT LAB: ANTI-FRAGILITY & BLACK SWAN VALIDATION", fontsize=16, weight='bold', y=0.98)
    
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.25)
    
    # Panel 1: Return Distribution (Fat Tails)
    ax1 = fig.add_subplot(gs[0, :]) # Top full width
    
    import seaborn as sns
    sns.histplot(returns, bins=100, kde=True, stat="density", color="blue", alpha=0.3, ax=ax1, label="Actual Distribution")
    
    # Normal Distribution Overlay
    mu, std = norm.fit(returns)
    xmin, xmax = ax1.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax1.plot(x, p, 'r--', linewidth=2, label=f"Normal Dist (Gaussian)")
    
    ax1.set_title(f"1. Tail Risk Analysis (Actual vs Normal)\nSkew: {skew:.2f} (Target > 0) | Kurtosis: {kurt:.2f} (Fat Tails)", fontsize=12, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Taleb Ratio Visual (Upside vs Downside Vol)
    ax2 = fig.add_subplot(gs[1, 0])
    
    upside = returns[returns > 0]
    downside = returns[returns < 0] # Make positive for comparison
    
    ax2.boxplot([upside, abs(downside)], labels=["Upside Returns", "Downside Risk (Abs)"], patch_artist=True, 
                boxprops=dict(facecolor="lightblue"))
    
    ax2.set_title(f"2. Asymmetry Analysis (Taleb Ratio)\nRatio: {taleb_ratio:.2f} (Target > 1.1)", fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Fragility Gauge (Status)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    # Draw simple text gauge
    color = "green" if "ANTI-FRAGILE" in status else "red" if "FRAGILE" in status else "orange"
    
    ax3.text(0.5, 0.7, "PORTFOLIO CLASSIFICATION:", ha='center', fontsize=12)
    ax3.text(0.5, 0.5, status, ha='center', fontsize=16, weight='bold', color=color, 
             bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=1'))
             
    ax3.text(0.5, 0.3, f"The 'Turkey' Score: {metrics.get('turkey_score', 0):.2f}\n(Hidden Tail Risk)", ha='center', fontsize=10)

    # Footer (Enhanced Commentary)
    text_content = (
        "INTERPRETATION GUIDE:\n"
        "1. TAIL RISK: We want the blue distribution to shift RIGHT (Positive Skew). Fat left tails indicate crash risk.\n"
        "2. TALEB RATIO: Measures payoff assymetry. Ratio > 1.1 means upside volatility > downside volatility (Good).\n"
        "3. TURKEY SCORE: A high negative score means steady small gains but massive hidden tail risk (like a Turkey before Thanksgiving).\n"
        "4. GOAL: We seek 'Anti-Fragility' -> Positioning that benefits from volatility and disorder (Convex Payoffs)."
    )
    fig.text(0.05, 0.02, text_content, fontsize=9, family='monospace', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='darkblue', linewidth=1.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    return fig

def plot_scenario_page(prices: pd.DataFrame, ticker: str = "SPY") -> plt.Figure:
    """Page 17: Scenario Analysis & Contingency Planning."""
    print("   Generating Scenario Analysis Page...")
    
    if ticker not in prices.columns: return None
    
    series = prices[ticker].dropna()
    current_price = series.iloc[-1]
    
    # 1. Calculate Probabilities
    # Using 3 months (60 days) horizon for standard table
    probs_1m = calculate_move_probabilities(series, days=21, n_sims=2000)
    probs_3m = calculate_move_probabilities(series, days=63, n_sims=2000)
    
    # 2. Generate Contingencies
    # Simple proxies for Regime classification (expand later)
    # Using 1M Vol approx
    curr_vol = series.pct_change().std() * np.sqrt(252)
    # Skew
    curr_skew = series.pct_change().dropna().skew()
    
    regime = "Normal"
    if curr_vol > 0.20: regime = "High Vol"
    elif curr_vol < 0.10: regime = "Low Vol"
    
    # Vol Score (0-1 normalized roughly)
    vol_score = min(max((curr_vol - 0.10) / 0.20, 0), 1)
    
    contingencies = generate_contingencies(current_price, regime, vol_score, curr_skew)
    
    # Plot
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("QUANT LAB: SCENARIO ANALYSIS & CONTINGENCIES", fontsize=16, weight='bold', y=0.98)
    
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.35, wspace=0.25)
    
    # Panel 1: Probability Table (Text)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    ax1.set_title("1. Move Probabilities (Monte Carlo)", fontsize=12, weight='bold')
    
    # Build Table Text
    table_text = [["Target", "1-Month Prob", "3-Month Prob"]]
    for target in probs_1m.index:
        p1 = probs_1m.loc[target, "Upside Prob"]
        # p1_down = probs_1m.loc[target, "Downside Risk"]
        p3 = probs_3m.loc[target, "Upside Prob"]
        # p3_down = probs_3m.loc[target, "Downside Risk"]
        
        # Display as range? Or just Upside for now simpler
        # Actually show Upside vs Downside side by side?
        # Let's simplify: Display Probability of touching +/- X%
        
        row = [f"{target}", f"{p1:.1%}", f"{p3:.1%}"]
        table_text.append(row)
        
    table = ax1.table(cellText=table_text, loc='center', cellLoc='center', colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Panel 2: Visual Cone (Simplified from Page 12, focused on Targets)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Re-simulate for plot small batch
    mu = series.pct_change().mean() * 252
    sigma = series.pct_change().std() * np.sqrt(252)
    T = 63/252
    t_sim, paths = simulate_gbm(current_price, mu, sigma, T, 1/252, 100)
    
    future_dates = pd.date_range(start=series.index[-1], periods=len(t_sim), freq='B')
    
    # Plot Cone
    p5 = np.percentile(paths, 5, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    
    ax2.plot(future_dates, p50, 'b-', label="Median")
    ax2.fill_between(future_dates, p5, p95, color='blue', alpha=0.1, label="90% Cone")
    
    # Horizontal Lines for Targets
    for target_pct in [0.05, -0.05]:
        level = current_price * (1 + target_pct)
        color = 'green' if target_pct > 0 else 'red'
        ax2.axhline(level, linestyle='--', color=color, alpha=0.5)
        ax2.text(future_dates[0], level, f"{target_pct:+.0%} Target", color=color, fontsize=8, va='bottom')
        
    ax2.set_title("2. Target Visualization (3-Month)", fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Contingency Playbook (Bottom Full Width)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    ax3.set_title(f"3. TACTICAL CONTINGENCY PLAYBOOK (Regime: {regime})", fontsize=12, weight='bold')
    
    # Format DataFrame as Table
    cols = list(contingencies.columns)
    cell_text = []
    for row in contingencies.itertuples(index=False):
        cell_text.append(list(row))
        
    # Add colors based on Scenario type?
    colors = []
    # Create colors matrix matching dimensions (n_rows x n_cols)
    for row in contingencies["Scenario"]:
        row_colors = []
        if "Critical" in row or "Crash" in row: base_color = "#ffcccc" # Red tint
        elif "Euphoria" in row: base_color = "#ccffcc" # Green tint
        else: base_color = "white"
        
        for _ in range(len(cols)): row_colors.append(base_color)
        colors.append(row_colors)
    
    table3 = ax3.table(cellText=cell_text, colLabels=cols, loc='center', cellLoc='left', 
                       colWidths=[0.25, 0.25, 0.35, 0.15],
                       cellColours=colors if colors else None)
                       
    table3.auto_set_font_size(False)
    table3.set_fontsize(10)
    table3.scale(1, 2.0) # More vertical space
    
    # Footer
    text_content = (
        "INTERPRETATION GUIDE:\n"
        "1. MOVE PROBABILITIES: Probability of touching price levels based on current volatility regime.\n"
        "2. CONTINGENCIES: Pre-planned actions to remove emotion. If 'Condition' is met, execute 'Action'.\n"
        "3. FRAGILITY CHECK: Playbook adapts to Skew/Kurtosis. Negative Skew = Expensive Puts = Risk Reversals preferred."
    )
    fig.text(0.05, 0.02, text_content, fontsize=9, family='monospace', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='darkblue', linewidth=1.5))
             
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    return fig

def plot_valuation_page(df: pd.DataFrame, fundamentals: Dict) -> plt.Figure:
    """Generates Page: Valuation & Real Rates (Real Yields & ERP)."""
    print("   Generating Valuation & Real Rates Page...")
    plt.style.use('default')
    
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    
    # 1. Real Interest Rates (10Y - Breakeven)
    ax1 = fig.add_subplot(gs[0])
    if "CF_Real_Yield" in df.columns:
        real_yield = df["CF_Real_Yield"].dropna()
        ax1.plot(real_yield.index, real_yield, color='blue', linewidth=2, label="US 10Y Real Yield")
        
        # Zones
        ax1.axhline(2.0, color='red', linestyle='--', label="Restrictive (>2.0%)")
        ax1.axhline(0.5, color='green', linestyle='--', label="Accommodative (<0.5%)")
        ax1.axhline(0.0, color='black', linewidth=1)
        
        # Fill
        ax1.fill_between(real_yield.index, real_yield, 2.0, where=(real_yield > 2.0), color='red', alpha=0.2)
        ax1.fill_between(real_yield.index, real_yield, 0.0, where=(real_yield < 0.0), color='green', alpha=0.1, label="Negative Real Rates")
        
        curr_real = real_yield.iloc[-1]
        status = "RESTRICTIVE" if curr_real > 2.0 else ("NEUTRAL" if curr_real > 0.5 else "STIMULATIVE")
        
        ax1.set_title(f"1. Real Interest Rates (The Cost of Capital)\nCurrent: {curr_real:.2f}% -> {status}", fontsize=14, weight='bold')
        ax1.set_ylabel("Real Yield (%)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        
        # Commentary Box
        text = (
            "Implications:\n"
            "• High Real Rates (>2%): Tight financial conditions. Headwind for Gold, Tech, & Multiples.\n"
            "• Low/Negative Real Rates: Loose conditions. Tailwind for Hard Assets & Speculation.\n"
            "• Trend is key: Rapidly rising real rates often trigger deleveraging events."
        )
        ax1.text(0.02, 0.05, text, transform=ax1.transAxes, fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey', boxstyle='round,pad=0.5'))
        
    else:
        ax1.text(0.5, 0.5, "Real Yield Data Missing", ha='center')

    # 2. Equity Risk Premium (ERP) - The Fed Model
    ax2 = fig.add_subplot(gs[1])
    
    pe = fundamentals.get("SPY_PE")
    # Fetch latest 10Y Yield from DF if not in fundamentals (it should be in DF)
    nominal_10y = df["10Y_Yield"].iloc[-1] if "10Y_Yield" in df.columns else None
    
    if pe and nominal_10y:
        earnings_yield = (1 / pe) * 100
        erp = earnings_yield - nominal_10y
        
        # Bar Chart
        labels = ['10Y Treasury Yield', 'S&P 500 Earnings Yield (1/PE)']
        values = [nominal_10y, earnings_yield]
        colors = ['red', 'green']
        
        bars = ax2.barh(labels, values, color=colors, alpha=0.7)
        ax2.set_xlim(0, max(values) * 1.3)
        
        # Annotate Values
        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2, f"{width:.2f}%", va='center', fontsize=12, weight='bold')
            
        # Draw Spread (ERP)
        # We can visualize this as a bracket or text
        ax2.text(max(values) * 0.5, 0.5, f"Equity Risk Premium (Spread): {erp:.2f}%", 
                 ha='center', va='center', transform=ax2.transAxes, fontsize=16, weight='bold', 
                 bbox=dict(facecolor='#f0f0f0', edgecolor='black', boxstyle='round,pad=1'))
        
        # Status
        valuation = "ATTRACTIVE (Stocks Cheap)" if erp > 3.0 else ("FAIR VALUE" if erp > 0 else "EXPENSIVE (Stocks Rich)")
        color_val = 'green' if erp > 0 else 'red'
        
        ax2.set_title(f"2. Equity Risk Premium (Fed Model Snapshot)\nValuation: {valuation}", fontsize=14, weight='bold', color=color_val)
        ax2.set_xlabel("Yield (%)")
        
        # ERP Context
        context_text = (
            "Interpretation:\n"
            "• ERP = Earnings Yield - Risk Free Rate.\n"
            "• Positive ERP: Stocks offer excess return over bonds.\n"
            "• Negative ERP: Stocks yield less than bonds (Speculative territory needs Growth)."
        )
        ax2.text(0.7, 0.1, context_text, transform=ax2.transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'))
        
    else:
        ax2.text(0.5, 0.5, "Insufficient Data for Valuation (PE or Yield Missing)", ha='center')
        
    plt.tight_layout()
    return fig

def plot_inflation_swap_curve(df: pd.DataFrame) -> plt.Figure:
    """Generates Page: Inflation Expectations Term Structure (Swaps Proxy)."""
    print("   Generating Inflation Expectations Page...")
    plt.style.use('default')
    
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    
    # 1. Term Structure: 5Y vs 10Y vs 5Y5Y
    ax1 = fig.add_subplot(gs[0])
    
    # Check Data Availability
    has_5y = "5Y_Breakeven" in df.columns
    has_10y = "10Y_Breakeven" in df.columns
    has_5y5y = "5Y5Y_Forward" in df.columns
    
    if has_5y5y or has_10y:
        if has_5y:
            breakeven_5 = df["5Y_Breakeven"].dropna()
            ax1.plot(breakeven_5.index, breakeven_5, color='green', alpha=0.6, label="5-Year Breakeven (Near Term)")
        
        if has_10y:
            breakeven_10 = df["10Y_Breakeven"].dropna()
            ax1.plot(breakeven_10.index, breakeven_10, color='blue', alpha=0.6, label="10-Year Breakeven (Medium Term)")
            
        if has_5y5y:
            fwd_5y5y = df["5Y5Y_Forward"].dropna()
            ax1.plot(fwd_5y5y.index, fwd_5y5y, color='red', linewidth=2, label="5Y, 5Y Forward (Long Term Anchor)")
            
            # Fill between 5Y and 5Y5Y to show curve slope if both exist
            if has_5y:
                 common = fwd_5y5y.index.intersection(breakeven_5.index)
                 ax1.fill_between(common, fwd_5y5y.loc[common], breakeven_5.loc[common], color='gray', alpha=0.1, label="Term Premium / Curve Slope")

        ax1.axhline(2.0, color='black', linestyle='--', linewidth=1.5, label="Fed Target (2.0%)")
        
        # Get latest values for title
        last_val = "N/A"
        status = "ANCHORED"
        if has_5y5y:
            curr = fwd_5y5y.iloc[-1]
            last_val = f"{curr:.2f}%"
            if curr > 2.5: status = "DE-ANCHORING (High)"
            elif curr < 1.5: status = "DEFLATIONARY RISK"
        
        ax1.set_title(f"1. Inflation Expectations Term Structure\nLong-Term Anchor (5Y5Y): {last_val} -> {status}", fontsize=14, weight='bold')
        ax1.set_ylabel("Inflation Rate (%)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        
        # --- Interpretation Box ---
        interp_text = "Interpretation:\n"
        
        # 1. Slope (5Y vs 5Y5Y)
        if has_5y5y and has_5y:
             slope = fwd_5y5y.iloc[-1] - breakeven_5.iloc[-1]
             if slope > 0.1:
                 interp_text += "• Curve Slope: CONTANGO (Normal). Market sees inflation rising to long-term avg.\n"
             elif slope < -0.1:
                 interp_text += "• Curve Slope: INVERTED (Front-Loaded). High short-term inflation expected to cool.\n"
             else:
                 interp_text += "• Curve Slope: FLAT. Inflation expectations are uniform across horizons.\n"
                 
        # 2. Anchor Level
        if has_5y5y:
            curr_anchor = fwd_5y5y.iloc[-1]
            if curr_anchor > 2.5:
                interp_text += "• Anchor Status: ELEVATED. Long-term expectations > 2.5% (Fed concern).\n"
            elif curr_anchor < 1.8:
                interp_text += "• Anchor Status: LOW. Risk of deflationary trap.\n"
            else:
                 interp_text += "• Anchor Status: STABLE. Near Fed target (2.0%).\n"
                 
        ax1.text(0.02, 0.1, interp_text, transform=ax1.transAxes, fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='grey'))
    else:
        ax1.text(0.5, 0.5, "Inflation Swap Data Missing (Check Tickers)", ha='center')

    # 2. Inflation Risk Premium (5Y5Y vs Spot CPI)
    # Allows us to see if the market is pricing higher inflation than current realized
    ax2 = fig.add_subplot(gs[1])
    
    if has_5y5y and "CPI_YoY" in df.columns:
        fwd = df["5Y5Y_Forward"].dropna()
        cpi = df["CPI_YoY"].dropna() * 100 # Scale to %
        
        # Align
        common_idx = fwd.index.intersection(cpi.index)
        fwd = fwd.loc[common_idx]
        cpi = cpi.loc[common_idx]
        
        spread = fwd - cpi
        
        ax2.plot(spread.index, spread, color='purple', label="Inflation Risk Premium (5Y5Y Fwd - Current CPI)")
        ax2.axhline(0, color='black', linewidth=1)
        
        ax2.fill_between(spread.index, spread, 0, where=(spread > 0), color='green', alpha=0.2, label="Market Expects HIGHER Inflation")
        ax2.fill_between(spread.index, spread, 0, where=(spread < 0), color='red', alpha=0.2, label="Market Expects LOWER Inflation")
        
        curr_spread = spread.iloc[-1]
        
        ax2.set_title(f"2. Inflation Term Premium (Expectations vs Reality)\nSpread: {curr_spread:.2f}%", fontsize=14, weight='bold')
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
        
        # Text Context
        text_ctx = (
            "Interpretation:\n"
            "• Positive Spread: Market believes current inflation is temporary/low and will rise to long-term avg.\n"
            "• Negative Spread: Market believes current inflation is too high and will fall (Mean Reversion).\n"
            "• Deep Negative: High conviction in Disinflation."
        )
        ax2.text(0.02, 0.1, text_ctx, transform=ax2.transAxes, fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'))
        
    else:
         ax2.text(0.5, 0.5, "Data Missing for Premium Calculation", ha='center')
         
    plt.tight_layout()
    return fig

def plot_global_macro_fx_page(df: pd.DataFrame, prices: pd.DataFrame, macro: pd.DataFrame, global_flows: Dict) -> plt.Figure:
    """Generates Page: Global Macro, FX & Capital Flows."""
    print("   Generating Global FX & Rates Page...")
    plt.style.use('default')
    
    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])
    
    # 1. Global Sovereign Yields Overlay (10Y)
    ax1 = fig.add_subplot(gs[0])
    
    # US 10Y
    if "10Y_Yield" in macro.columns:
        us10 = macro["10Y_Yield"].dropna()
        ax1.plot(us10.index, us10, color='blue', linewidth=2, label="US 10Y Treasury")
        
    # Germany 10Y
    if "Germany_10Y" in macro.columns:
        de10 = macro["Germany_10Y"].dropna()
        ax1.plot(de10.index, de10, color='black', linestyle='--', label="Germany 10Y Bund")
        
    # Japan 10Y
    if "Japan_10Y" in macro.columns:
        jp10 = macro["Japan_10Y"].dropna()
        ax1.plot(jp10.index, jp10, color='red', linestyle='--', label="Japan 10Y JGB")
        
    # UK 10Y
    if "UK_10Y" in macro.columns:
        uk10 = macro["UK_10Y"].dropna()
        ax1.plot(uk10.index, uk10, color='green', linestyle=':', label="UK 10Y Gilt")
        
    ax1.set_title("1. Global Sovereign Bond Yields (10Y Nominal)\nMonitor: Yield Divergence drives Capital Flows", fontsize=14, weight='bold')
    ax1.set_ylabel("Yield (%)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # 2. The Carry Trade (US-JP Spread vs USD/JPY)
    ax2 = fig.add_subplot(gs[1])
    
    if "10Y_Yield" in macro.columns and "Japan_10Y" in macro.columns and "JPY=X" in prices.columns:
        # Align Data
        us = macro["10Y_Yield"]
        jp = macro["Japan_10Y"]
        fx = prices["JPY=X"]
        
        common = us.index.intersection(jp.index).intersection(fx.index)
        
        spread = (us.loc[common] - jp.loc[common])
        usd_jpy = fx.loc[common]
        
        # Plot Spread (Left Axis)
        color_spread = 'darkblue'
        ax2.plot(spread.index, spread, color=color_spread, label="US-Japan 10Y Yield Spread")
        ax2.set_ylabel("Yield Spread (%)", color=color_spread, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color_spread)
        
        # Plot FX (Right Axis)
        ax2_twin = ax2.twinx()
        color_fx = 'darkred'
        ax2_twin.plot(usd_jpy.index, usd_jpy, color=color_fx, linestyle='--', alpha=0.7, label="USD/JPY Exchange Rate")
        ax2_twin.set_ylabel("USD/JPY", color=color_fx, fontsize=12)
        ax2_twin.tick_params(axis='y', labelcolor=color_fx)
        
        # Status
        carry_status = global_flows.get("japan_carry", {}).get("status", "N/A")
        
        ax2.set_title(f"2. The 'Carry Trade' Engine (Yield Spread vs FX)\nStatus: {carry_status}", fontsize=14, weight='bold')
        
        # Legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Insufficient Data for Carry Trade Analysis", ha='center')

    # 3. Currency Relative Performance (1-Year Normalized)
    ax3 = fig.add_subplot(gs[2])
    
    currencies = {
        "USD (DXY)": "DX-Y.NYB" if "DX-Y.NYB" in prices.columns else "UUP",
        "Euro (FXE)": "FXE",
        "Yen (FXY)": "FXY",
        "Yuan (CNY=X)": "CNY=X"
    }
    
    # Filter for existing
    valid_curr = {k: v for k, v in currencies.items() if v in prices.columns}
    
    if valid_curr:
        # Get last 252 days
        start_idx = -252
        
        for name, ticker in valid_curr.items():
            series = prices[ticker].iloc[start_idx:].dropna()
            if not series.empty:
                # Normalize to 100
                norm = (series / series.iloc[0]) * 100
                
                # Invert logic for standard pair convention if needed?
                # DXY: Long USD.
                # FXE: Long Euro (Short USD).
                # FXY: Long Yen (Short USD).
                # CNY=X: USD/CNY usually. If it's USD/CNY, rising = Weak Yuan.
                # Let's plot raw ETF performance for simplicity as they represent "Long that Currency against USD" (except DXY).
                # Note: CNY=X in Yahoo is usually USD/CNY.
                
                linewidth = 2 if "USD" in name else 1.5
                alpha = 1.0 if "USD" in name else 0.7
                
                ax3.plot(norm.index, norm, label=name, linewidth=linewidth, alpha=alpha)
                
        ax3.axhline(100, color='black', linestyle='--', linewidth=1)
        ax3.set_title("3. Currency Momentum (1-Year Relative Performance, Normalized=100)", fontsize=14, weight='bold')
        ax3.set_ylabel("Rel Perf")
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3)
        
    else:
        ax3.text(0.5, 0.5, "Currency Data Missing", ha='center')
        
    plt.tight_layout()
    
    # Interpretation
    fig.subplots_adjust(bottom=0.12)
    interp_text = "Global Macro & FX Insights:\n"
    
    # 1. Yield Divergence
    if "10Y_Yield" in macro.columns:
        us_y = macro["10Y_Yield"].dropna().iloc[-1]
        interp_text += f"• US Yields: {us_y:.2f}%. "
        if "Germany_10Y" in macro.columns:
             de_y = macro["Germany_10Y"].dropna().iloc[-1]
             diff = us_y - de_y
             interp_text += f"vs Bunds: {diff:.2f}% spread. "
    
    # 2. Carry
    carry_stat = global_flows.get("japan_carry", {}).get("status", "N/A")
    if "UNWINDING" in carry_stat:
        interp_text += "\n• CARRY UNWIND ALERT: JPY Strengthening while Yield Spread compresses. Risk-Off signal."
    elif "ACCELERATING" in carry_stat:
        interp_text += "\n• CARRY TRADE ON: JPY Weakening + Yield Spread widening. Supports Global Liquidity."
        
    fig.text(0.05, 0.02, interp_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round'))
    
    return fig
