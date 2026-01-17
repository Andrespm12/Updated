"""
Main entry point for the Macro Rotations Dashboard.
Orchestrates Config -> Data -> Analytics -> Plotting -> Report.
"""
import sys
# Make sure we can import local modules
sys.path.append(".")
import pandas as pd

from core.config import CONFIG
from data.loader import download_data
from analytics.macro_models import build_analytics, add_trends_and_score, calculate_global_flows
from analytics.alpha_models import calculate_net_liquidity, calculate_vol_term_structure, calculate_tail_risk
from analytics.quant import calculate_cross_asset_correlations, calculate_cta_momentum, optimize_portfolio, calculate_forward_returns, calculate_market_internals
from plotting.charts import (
    run_backtest_plot, plot_risk_macro_dashboard, 
    plot_monetary_plumbing, plot_forward_models, plot_quant_lab_dashboard, 
    plot_alpha_factors_page, plot_cross_asset_page, plot_efficient_frontier_page,
    plot_predictive_models_page, plot_monte_carlo_cone, plot_stochastic_page,
    plot_mean_reversion_page, plot_microstructure_page, plot_antifragility_page,
    plot_scenario_page, plot_valuation_page, plot_inflation_swap_curve,
    plot_global_macro_fx_page
)
from plotting.report import generate_pdf_report

def main():
    print("--- Initializing SUPER MACRO DASHBOARD (Modular) ---")
    
    # 1. Data Ingestion
    prices, macro, fundamentals = download_data(CONFIG, use_cache=True)
    
    # 2. Analytics Pipeline
    print("Running Analytics Pipeline...")
    df = build_analytics(prices, macro, CONFIG)
    df = add_trends_and_score(df, CONFIG)
    
    # 2b. Alpha Models
    print("Running Alpha Models...")
    alpha_data = {
        "net_liquidity": calculate_net_liquidity(macro),
        "vol_structure": calculate_vol_term_structure(prices),
        "tail_risk": calculate_tail_risk(prices)
    }
    
    # 2c. Cross-Asset & CTA
    print("Running Cross-Asset & CTA Models...")
    corr_data = calculate_cross_asset_correlations(prices)
    cta_data = calculate_cta_momentum(prices)
    
    # 2d. Global FX Flows
    global_flows = calculate_global_flows(prices, macro)
    
    # 2d. Portfolio Optimization (Phase 7: Macro-Adjusted)
    print("Running Portfolio Optimization (Macro-Adjusted)...")
    last_score = df["MACRO_SCORE"].iloc[-1]
    last_cpi = macro["CPI"].pct_change(12).iloc[-1]
    
    fwd_rets = calculate_forward_returns(prices, last_score, last_cpi)
    opt_data = optimize_portfolio(prices, expected_returns=fwd_rets)
    
    # Add Score to Opt Data for Report
    opt_data["macro_score"] = last_score
    
    # 2e. Predictive Analytics (Phase 8)
    print("Running Predictive Analytics...")
    internals = calculate_market_internals(prices)
    pred_data = {
        "internals": internals,
        "recession_prob": df.get("Recession_Prob", pd.Series(dtype=float))
    }
    
    # 3. Visualization Pipeline
    print("Generating Visualizations...")
    figures = {}
    
    # Backtest
    fig_bt = run_backtest_plot(df, prices)
    if fig_bt: figures["backtest"] = fig_bt
    
    # Risk & Macro
    fig_risk = plot_risk_macro_dashboard(df, prices)
    figures["risk_macro"] = fig_risk
    
    # Plumbing
    fig_plumb = plot_monetary_plumbing(df)
    figures["plumbing"] = fig_plumb
    
    # Global FX
    fig_fx = plot_global_macro_fx_page(df, prices, macro, global_flows)
    figures["global_fx"] = fig_fx
    


    # Forward Models
    fig_fwd = plot_forward_models(df, prices)
    figures["forward"] = fig_fwd

    # Quant Lab
    fig_quant = plot_quant_lab_dashboard(prices)
    figures["quant"] = fig_quant
    
    # Alpha Factors
    fig_alpha = plot_alpha_factors_page(prices, macro, alpha_data)
    figures["alpha"] = fig_alpha
    
    # Cross-Asset
    fig_cross = plot_cross_asset_page(prices, corr_data)
    figures["cross_asset"] = fig_cross
    
    # Efficient Frontier
    fig_frontier = plot_efficient_frontier_page(opt_data)
    figures["frontier"] = fig_frontier
    
    # Predictive Models (Phase 8)
    fig_pred = plot_predictive_models_page(df, internals, df.get("Recession_Prob", pd.Series(dtype=float)))
    figures["predictive"] = fig_pred
    
    # Monte Carlo Quant Lab (Phase 9)
    fig_gbm = plot_monte_carlo_cone(prices, ticker="SPY", days=63, n_sims=1000)
    figures["monte_carlo"] = fig_gbm
    
    # Stochastic & Regimes (Phase 10)
    fig_stoch = plot_stochastic_page(prices, ticker="SPY")
    figures["stochastic"] = fig_stoch
    
    # Mean Reversion (Phase 11)
    fig_ou = plot_mean_reversion_page(prices)
    figures["mean_reversion"] = fig_ou
    
    # Jumps & Microstructure (Phase 12)
    fig_micro = plot_microstructure_page(prices, ticker="SPY")
    figures["microstructure"] = fig_micro
    
    # Anti-Fragility (Phase 13)
    fig_anti = plot_antifragility_page(prices, ticker="SPY")
    figures["antifragility"] = fig_anti
    
    # Scenario Analysis (Phase 14)
    # Scenario Analysis (Phase 14)
    fig_scen = plot_scenario_page(prices, ticker="SPY")
    figures["scenarios"] = fig_scen
    
    # Valuation & Real Rates (New)
    fig_val = plot_valuation_page(df, fundamentals)
    figures["valuation"] = fig_val

    # Inflation Swaps (New)
    fig_inf = plot_inflation_swap_curve(df)
    figures["inflation_swaps"] = fig_inf

    # 4. Reporting
    generate_pdf_report(df, prices, figures, alpha_data, cta_data, opt_data, pred_data)
    
    # 5. IC Briefing (New)
    from plotting.ic_report import generate_ic_report
    generate_ic_report(df, prices, macro, alpha_data, pred_data)
    
    print("Done!")

if __name__ == "__main__":
    main()
