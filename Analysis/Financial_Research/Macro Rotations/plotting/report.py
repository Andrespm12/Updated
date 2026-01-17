"""
Report Generation and Text Analytics.
"""
import textwrap
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict
from matplotlib.backends.backend_pdf import PdfPages
import analytics.quant as quant_engine





def generate_narrative_summary(df: pd.DataFrame, alpha_data: Dict) -> str:
    """Synthesizes a rule-based executive narrative paragraph."""
    last = df.iloc[-1]
    score = last["MACRO_SCORE"]
    
    # 1. The "Weather" (Regime)
    weather = ""
    if score > 0.3: weather = "Markets are currently in a SUPPORTIVE RISK-ON regime."
    elif score < -0.3: weather = "Markets are currently in a DEFENSIVE RISK-OFF regime."
    else: weather = "Markets are currently in a TRANSITIONAL / CHOPPY regime with no clear trend."
    
    # 2. The "Fuel" (Liquidity)
    liq_status = alpha_data.get("net_liquidity", {}).get("status", "NEUTRAL")
    liq_text = ""
    if "EXPANDING" in liq_status:
        liq_text = "Liquidity conditions are improving (Fed Net Liquidity Rising), providing a tailwind for asset prices."
    elif "CONTRACTING" in liq_status:
        liq_text = "However, liquidity is withdrawing from the system, creating a headwind for valuations."
    else:
        liq_text = "Liquidity conditions are stable."
        
    # 3. The "Threats" (Crash/Inflation)
    threats = []
    vol_sig = alpha_data.get("vol_structure", {}).get("signal", "NORMAL")
    if "CRASH" in vol_sig:
        threats.append("CRITICAL WARNING: Volatility structure is in Backwardation, signaling imminent crash risk.")
    elif "CAUTION" in vol_sig:
        threats.append("Volatility markets are showing signs of stress.")
        
    cpi = last.get("CPI_YoY", 0)
    if cpi > 0.03:
        threats.append(f"Inflation remains sticky at {cpi:.1%}, limiting Fed flexibility.")
        
    threat_text = " ".join(threats) if threats else "No structural crash signals are currently active."
    
    # 4. The "Punchline"
    punchline = ""
    if score > 0 and "CRASH" not in vol_sig:
        punchline = "ENVIRONMENT FAVORS: Long exposures to Growth & Momentum factors."
    elif score < 0 or "CRASH" in vol_sig:
        punchline = "ENVIRONMENT FAVORS: Capital Preservation, Cash, and Volatility Longs."
    else:
        punchline = "ENVIRONMENT FAVORS: Tactical trading and Stock Selection over broad Beta."
        
    return f"{weather} {liq_text} {threat_text}\n\n   >> {punchline}"

def generate_quant_synthesis(df: pd.DataFrame, prices: pd.DataFrame, alpha_data: Dict) -> str:
    """Generates the Executive Quant Synthesis (The 'Know What To Do' Paragraph)."""
    if prices is None or "SPY" not in prices.columns: return ""
    
    spy = prices["SPY"].dropna()
    rets = spy.pct_change().infer_objects(copy=False).dropna()
    curr_price = spy.iloc[-1]
    
    # 1. On-the-fly Metrics
    # (A) Regime
    score = df["MACRO_SCORE"].iloc[-1]
    regime = "RISK-ON" if score > 0 else "RISK-OFF"
    
    # (B) Volatility of Volatility (Tail Risk)
    from analytics.quant import calculate_antifragility_metrics
    af_metrics = calculate_antifragility_metrics(rets)
    turkey_score = af_metrics.get("turkey_score", 0)
    skew = af_metrics.get("skew", 0)
    
    # (C) Probabilities (Gaussian Approx for speed in text, consistent with MC)
    vol = rets.std() * np.sqrt(252)
    # Prob of +5% in 1 month (21 days)
    from scipy.stats import norm
    t = 21/252
    sigma_t = vol * np.sqrt(t)
    
    # Upside +5%
    d2_up = (np.log(1.05)) / sigma_t
    prob_up = 1 - norm.cdf(d2_up) # Approx
    
    # Downside -5%
    d2_down = (np.log(0.95)) / sigma_t
    prob_down = norm.cdf(d2_down)
    
    # 2. Construct Actionable Advice
    advice = []
    
    # Layer 1: Directional Bias (Regime + Momentum)
    if regime == "RISK-ON":
        if prob_up > prob_down:
            advice.append("PRIMARY BIAS: ACCUMULATE RISK. Macro score and probability skew favor upside.")
        else:
            advice.append("PRIMARY BIAS: CAUTIOUS BULL. Macro is positive but volatility pricing suggests resistance.")
    else:
        if turkey_score < -1:
            advice.append("PRIMARY BIAS: DEFENSIVE HEDGING. High fragility and weak macro score.")
        else:
            advice.append("PRIMARY BIAS: REDUCE EXPOSURE. Macro headwinds dominate.")
            
    # Layer 2: Volatility / Contingency
    if turkey_score < -2.0:
        advice.append("CRITICAL: HIDDEN TAIL RISK. Buy OTM Puts (90-120 DTE). Spend ~2% of AUM/yr. Goal: Partial Hedge (Not Delta Neutral).")
    elif skew < -1.0:
        advice.append("EXECUTION: SKEW EXTREME. Puts are expensive. Use Risk Reversals (Sell 30d Puts / Buy 30d Calls) to finance delta.")
    elif skew > 0.5:
        advice.append("EXECUTION: CALLS ARE EXPENSIVE. Take profits or write covered calls.")
     
    # Layer 3: Liquidity
    liq_status = alpha_data.get('net_liquidity', {}).get('status', 'NEUTRAL')
    if "EXPANDING" in liq_status:
        advice.append("TAILWIND: Liquidity is expanding. Buy dips in High Beta sectors.")
    elif "CONTRACTING" in liq_status:
        advice.append("HEADWIND: Liquidity is drying up. Tighten stop-losses.")

    advice_text = "\n".join([f"   > {a}" for a in advice])
    
    text = (
        f"********************************************************************************\n"
        f" EXECUTIVE QUANT SYNTHESIS (THE 'GAME PLAN')\n"
        f"********************************************************************************\n"
        f" CONTEXT:  {regime} Regime | Vol: {vol:.1%} | Fragility Score: {turkey_score:.2f}\n"
        f" PROBS (1M): Odds of +5% Rally: {prob_up:.0%}  vs  Odds of -5% Drop: {prob_down:.0%}\n\n"
        f" ACTIONABLE PLAYBOOK:\n"
        f"{advice_text}\n\n"
        f" SENTIMENT: {'EUPHORIC' if prob_up > 0.4 else 'FEARFUL' if prob_down > 0.4 else 'NEUTRAL'}\n"
        f"********************************************************************************\n"
    )
    return text

def generate_capital_flows_commentary(df: pd.DataFrame, prices: pd.DataFrame = None, alpha_data: Dict = {}, cta_data: pd.DataFrame = None, opt_data: Dict = None, pred_data: Dict = None):
    """Analyzes the results and returns full commentary text."""
    last = df.iloc[-1]
    
    # --- EXECUTIVE SYNTHESIS ---
    synthesis_text = generate_quant_synthesis(df, prices, alpha_data)
    
    # --- STRATEGY SNAPSHOT ---
    strat_text = ""
    if prices is not None:
        try:
            metrics, _ = quant_engine.calculate_strategy_performance(df, prices)
            if metrics:
                strat_text += " STRATEGY PERFORMANCE SNAPSHOT (CAGR | Sharpe | MaxDD)\n"
                strat_text += "-" * 80 + "\n"
                sorted_strats = sorted(metrics.items(), key=lambda x: x[1]['Sharpe'], reverse=True)
                for name, m in sorted_strats:
                    strat_text += f" {name:<20} | {m['CAGR']:>6.1%} | {m['Sharpe']:>6.2f} | {m['MaxDD']:>6.1%}\n"
                strat_text += "-" * 80 + "\n"
        except Exception as e:
            pass

    # 1. Liquidity Valve
    btc_trend = "OPEN" if last["CF_Liquidity_Valve"] > df["CF_Liquidity_Valve_MA200"].iloc[-1] else "CLOSED"
    real_yield = last["CF_Real_Yield"]
    
    valve_commentary = ""
    if btc_trend == "OPEN" and real_yield < 1.5:
        valve_commentary = "The Liquidity Valve is OPEN. With real yields contained, capital is flowing out the risk curve into speculative assets."
    elif btc_trend == "CLOSED" and real_yield > 2.0:
        valve_commentary = "The Liquidity Valve is CLOSED. High real yields are sucking liquidity out of the system. Expect pressure on long-duration assets."
    else:
        valve_commentary = "The Liquidity Valve is NEUTRAL. Markets are chopping as real rates seek equilibrium."

    # 2. Fragility
    breadth_trend = "HEALTHY" if last["CF_Breadth"] > df["CF_Breadth_MA200"].iloc[-1] else "NARROW"
    vix_level = last["CF_VIX"]
    
    fragility_commentary = ""
    if breadth_trend == "NARROW" and vix_level < 15:
        fragility_commentary = "WARNING: Market Fragility is HIGH. Breadth is narrowing while VIX is low (Complacency). This divergence often precedes 'air pockets' or sharp corrections."
    elif breadth_trend == "HEALTHY":
        fragility_commentary = "Market Structure is ROBUST. Broad participation (Equal Weight outperforming) suggests a healthy accumulation phase."
    else:
        fragility_commentary = "Market Structure is MIXED. Monitor for further narrowing in participation."

    # 3. Consumer
    cons_trend = "EXPANSION" if last["CF_Consumer"] > df["CF_Consumer_MA200"].iloc[-1] else "RETRENCHMENT"
    cons_commentary = ""
    if cons_trend == "RETRENCHMENT":
        cons_commentary = "The Consumer is RETRENCHING. Cyclicals lagging Staples indicates household budgets are tightening, a classic late-cycle signal."
    else:
        cons_commentary = "The Consumer is RESILIENT. Outperformance in Discretionary stocks suggests the economy is avoiding immediate recession."

    # 4. Plumbing
    plumbing_status = "FUNCTIONAL"
    if "SOFR" in df.columns and "Fed_Funds" in df.columns and last["SOFR"] > last["Fed_Funds"] + 0.05:
         plumbing_status = "CRITICAL (Repo Stress - SOFR Spike)"
    elif last["CF_NFCI"] > 0:
         plumbing_status = "STRESSED (Tight Financial Conditions)"
    else:
         plumbing_status = "FUNCTIONAL (Liquid)"
    
    # 5. Score
    score = last["MACRO_SCORE"]
    score_status = ""
    if score > 0.3: score_status = "RISK-ON (Bullish Regime)"
    elif score < -0.3: score_status = "RISK-OFF (Bearish Regime)"
    else: score_status = "NEUTRAL / CHOPPY"



    # Rotations
    def check_trend(col):
        return "LEADING" if last[col] > df[f"{col}_MA200"].iloc[-1] else "LAGGING"

    val_growth = check_trend("ROT_Value_Growth")
    cyc_def = check_trend("ROT_Cyc_Def_Sectors")
    small_large = check_trend("ROT_Small_Large")
    hibeta_lovol = check_trend("ROT_HiBeta_LoVol")

    # Playbook
    regime_name = ""
    if score > 0.5: regime_name = "AGGRESSIVE RISK-ON (Liquidity Fueled)"
    elif score > 0: regime_name = "MODERATE RISK-ON (Selective)"
    elif score > -0.5: regime_name = "CHOPPY / TRANSITIONAL (Caution)"
    else: regime_name = "DEFENSIVE / RISK-OFF (Capital Preservation)"
        
    longs = []
    shorts = []
    
    if btc_trend == "OPEN" and real_yield < 1.5:
        longs.append("Crypto (BTC/ETH)")
        longs.append("High Beta Tech (XLK/XBI)")
    elif real_yield > 2.0:
        shorts.append("Long Duration Assets (TLT, Gold)")
        longs.append("Cash (SHY)")
        
    if cyc_def == "LEADING":
        longs.append("Cyclicals (XLI, XLB)")
        shorts.append("Defensives (XLU, XLP)")
    else:
        longs.append("Quality/Defensives (NOBL, XLP)")
        shorts.append("Deep Cyclicals")
        
    if "UUP" in df.columns:
        usd_trend = "BULLISH" if last["UUP"] > df[f"UUP_MA200"].iloc[-1] else "BEARISH"
        if usd_trend == "BULLISH":
            shorts.append("Emerging Markets (EEM)")
            shorts.append("Commodities (Gold/Copper)")
        else:
            longs.append("Emerging Markets (EEM)")
            longs.append("Hard Assets (GLD/CPER)")

    invalidation_text = ""
    if regime_name.count("RISK-ON"):
        invalidation_text = "Thesis INVALIDATED if: 10Y Real Yields close > 2.0% OR Bitcoin closes below 200D MA."
    elif regime_name.count("RISK-OFF"):
        invalidation_text = "Thesis INVALIDATED if: NFCI drops below -0.6 (Easing) AND Discretionary (XLY) reclaims 200D MA."
    else:
        invalidation_text = "Thesis INVALIDATED if: VIX spikes > 25 (Breakout) OR SPY loses 200D MA support."

    # Anomalies
    anomalies = []
    z_cols = [c for c in df.columns if "_Z" in c]
    for col in z_cols:
        z_val = last[col]
        base_name = col.replace("_Z", "")
        if z_val > 2.0:
            anomalies.append(f"[!] {base_name} is OVEREXTENDED (+{z_val:.1f}σ). Prone to Mean Reversion.")
        elif z_val < -2.0:
            anomalies.append(f"[OK] {base_name} is OVERSOLD ({z_val:.1f}σ). Potential Bounce.")
    anomaly_text = "\n".join(anomalies) if anomalies else "None. No statistical extremes detected (>2 Sigma)."

    # Quant Lab
    quant_text = ""
    try:
        assets_to_analyze = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
        quant_text += "1. MULTI-ASSET QUANT REGIMES & OPTION PLAYBOOKS:\n"
        
        for ticker in assets_to_analyze:
            if ticker not in prices.columns: continue
            
            # Data Prep
            ret = prices[ticker].pct_change().infer_objects(copy=False).dropna()
            
            # A. Volatility Regime (GARCH)
            try:
                vol_garch, _ = quant_engine.fit_garch(ret.values * 100)
                if len(vol_garch) > 0:
                    curr_vol = vol_garch[-1] / 100 * np.sqrt(252)
                else:
                    curr_vol = ret.std() * np.sqrt(252)

                hist_vol = ret.rolling(21).std().iloc[-1] * np.sqrt(252)
                
                vol_status = "HIGH" if curr_vol > 0.20 else "LOW"
                vol_trend = "EXPANDING" if curr_vol > hist_vol * 1.1 else ("CONTRACTING" if curr_vol < hist_vol * 0.9 else "STABLE")
            except:
                vol_status = "N/A"
                vol_trend = "N/A"
                curr_vol = 0.0
                
            # B. Monte Carlo Bias
            S0 = prices[ticker].iloc[-1]
            mu = ret.mean() * 252
            sigma_sim = ret.std() * np.sqrt(252)
            # Short term simulation (1 Month)
            _, paths = quant_engine.simulate_gbm(S0, mu, sigma_sim, 30/252.0, 1/252.0, 100)
            exp_ret = (paths[:,-1].mean() / S0) - 1
            bias = "BULLISH" if exp_ret > 0.005 else ("BEARISH" if exp_ret < -0.005 else "NEUTRAL")
            
            # C. Generate Playbook
            strategy = ""
            if vol_status == "LOW" and vol_trend == "STABLE":
                if bias == "BULLISH": strategy = "Long Call / Debit Spread (Cheap Vol)"
                elif bias == "BEARISH": strategy = "Long Put / Debit Spread (Cheap Vol)"
                else: strategy = "Calendar Spread / Long Straddle"
            elif vol_status == "HIGH" or vol_trend == "EXPANDING":
                if bias == "BULLISH": strategy = "Short Put Spread / Cov. Call (Sell Vol)"
                elif bias == "BEARISH": strategy = "Short Call Spread / Collar"
                else: strategy = "Iron Condor / Short Straddle (Harvest Vol)"
            else:
                strategy = "Trend Following (Directional)"
                
            quant_text += f"   - {ticker}: {bias} Bias | Vol: {vol_status} ({vol_trend}) -> STRATEGY: {strategy}\n"

        if "SPY" in prices.columns:
            S0 = prices["SPY"].iloc[-1]
            sigma_spy = prices["SPY"].pct_change().infer_objects(copy=False).std() * np.sqrt(252)
            # ATM 30-Day Option
            greeks = quant_engine.calculate_greeks(S0, S0, 30/365.0, 0.045, sigma_spy, "call")
            
            quant_text += f"\n2. SPY ATM GREEKS (Risk Sensitivity - 30D Call):\n"
            quant_text += f"   - Delta: {greeks.get('Delta', 0):.2f} (Directional Risk)\n"
            quant_text += f"   - Gamma: {greeks.get('Gamma', 0):.4f} (Convexity/Acceleration)\n"
            quant_text += f"   - Vega:  {greeks.get('Vega', 0):.2f} (Vol Sensitivity)\n"
            quant_text += f"   - Theta: {greeks.get('Theta', 0):.2f} (Time Decay)\n"

    except Exception as e:
        quant_text += f"Error generating insights: {e}\n"

    # Global
    global_text = ""
    try:
        us_breakeven = df["10Y_Breakeven"].iloc[-1] if "10Y_Breakeven" in df.columns else 0.0
        diffs = []
        if "RateDiff_US_DE" in df.columns: diffs.append(f"US-Bund: {last['RateDiff_US_DE']:.2f}%")
        if "RateDiff_US_JP" in df.columns: diffs.append(f"US-JGB:  {last['RateDiff_US_JP']:.2f}%")
        diff_str = ", ".join(diffs)
        
        fx_playbook = []
        if "JPY=X" in prices.columns:
             jpy_trend = "BULLISH" if prices["JPY=X"].iloc[-1] > prices["JPY=X"].rolling(200).mean().iloc[-1] else "BEARISH"
             if jpy_trend == "BULLISH" and last.get("RateDiff_US_JP", 0) > 2.5:
                 fx_playbook.append("Long USD/JPY (Carry Trade)")
        
        global_text += f"1. RATE DIFFERENTIALS (Yield Advantage):\n   {diff_str}\n"
        global_text += f"2. FX REGIME & PLAYBOOK:\n"
        if fx_playbook:
            for play in fx_playbook: global_text += f"   - {play}\n"
        else:
            global_text += "   - Neutral / No clear divergence signals.\n"
    except Exception as e:
        global_text = str(e)

    # CTA Monitor
    cta_text = ""
    if cta_data is not None and not cta_data.empty:
        cta_text = "CTA TREND MONITOR (Multi-Timeframe Signals):\n"
        cta_text += f"   {'Asset':<10} | {'Short(20D)':<10} | {'Med(60D)':<10} | {'Long(200D)':<11} | {'Status':<15}\n"
        cta_text += "-" * 75 + "\n"
        for _, row in cta_data.iterrows():
            cta_text += f"   {row['Asset']:<10} | {row['Short (20D)']:<10} | {row['Med (60D)']:<10} | {row['Long (200D)']:<11} | {row['Signal']:<15}\n"
    else:
        cta_text = "CTA Data Invalid or Missing.\n"

    # Optimization Text
    opt_text = ""
    if opt_data and "max_sharpe" in opt_data:
        res = opt_data["max_sharpe"]["metrics"]
        w = opt_data["max_sharpe"]["weights"]
        
        # Check for Macro Adjustments
        score = opt_data.get("macro_score", 0.0)
        title_suffix = "(Macro-Adjusted)" if score != 0.0 else "(Historical)"
        
        opt_text = f"PORTFOLIO OPTIMIZATION {title_suffix}:\n"
        if score != 0.0:
            opt_text += f"   *Returns Adjusted for Regime Score: {score:.2f}*\n"
            
        opt_text += f"   Max Sharpe Portfolio (Sharpe: {res[2]:.2f} | Vol: {res[1]:.1%})\n"
        opt_text += "   Recommended Allocation:\n"
        for asset, weight in w.items():
            if weight > 0.01:
                opt_text += f"   - {asset:<10}: {weight:.1%}\n"
    else:
        opt_text = "Optimization Data Missing.\n"

    # Vol Text
    vol_text = ""
    try:
        vol_tickers = {"^VIX": "SP500 (VIX)", "^VXN": "Nasdaq (VXN)", "^GVZ": "Gold (GVZ)", "^MOVE": "Rates (MOVE)"}
        vol_text += "1. IMPLIED VOLATILITY RANK (1-Year Percentile):\n"
        for ticker, name in vol_tickers.items():
            if ticker in prices.columns:
                hist = prices[ticker].iloc[-252:]
                curr = hist.iloc[-1]
                low = hist.min()
                high = hist.max()
                rank = (curr - low) / (high - low) if high > low else 0.5
                regime = "FEAR" if rank > 0.8 else ("COMPLACENCY" if rank < 0.2 else "NORMAL")
                vol_text += f"   - {name}: {curr:.2f} (Rank: {rank:.0%}) -> {regime}\n"
        vol_text += "\n2. VOLATILITY INTERPRETATION:\n   - High Rank (>80%): Markets pricing extreme stress.\n   - Low Rank (<20%): Markets pricing perfection.\n"
    except: pass

    # Risk
    risk_text = ""
    try:
        _, curves = quant_engine.calculate_strategy_performance(df, prices)
        if "Core-Satellite (CCI)" in curves.columns:
            var95, cvar95 = quant_engine.calculate_portfolio_risk(curves["Core-Satellite (CCI)"])
            risk_text += "1. CORE-SATELLITE PORTFOLIO RISK (Daily):\n"
            risk_text += f"   - 95% VaR: {var95:.2%} (Max loss on 95% of days)\n"
            risk_text += f"   - 95% CVaR: {cvar95:.2%} (Avg loss on worst 5% of days)\n"
            risk_text += "   - Interpretation: " + ("ELEVATED RISK." if var95 < -0.015 else "NORMAL RISK PROFILE.\n")
    except: risk_text = "N/A"

    # Predictive Text
    pred_text = ""
    if pred_data:
        pred_text = "PREDICTIVE ANALYTICS MODEL (Forward Looking):\n"
        # Recession
        rec_prob = last.get("Recession_Prob", 0.0)
        rec_status = "HIGH RISK (>30%)" if rec_prob > 30 else "LOW RISK"
        pred_text += f"1. NY FED RECESSION PROBABILITY (12M): {rec_prob:.1f}% -> {rec_status}\n"
        
        # Internals
        pred_text += "2. MARKET INTERNALS (Leading Ratios):\n"
        internals = pred_data.get("internals", {})
        for name, data in internals.items():
            # Calculate simple Trend
            series = data["Series"].dropna()
            if not series.empty:
                curr = series.iloc[-1]
                ma50 = series.rolling(50).mean().iloc[-1]
                trend = "BULLISH" if curr > ma50 else "BEARISH"
                
                # Contextualize
                if name == "Defensive":
                     status = "DEFENSIVE (Caution)" if trend == "BULLISH" else "RISK-ON"
                elif name == "Credit_Risk":
                     status = "RISK-ON" if trend == "BULLISH" else "RISK-OFF (Stress)"
                else:
                     status = trend
                
                pred_text += f"   - {name:<25}: {status}\n"

    # Assemble Full Text
    date_str = dt.datetime.today().strftime("%Y-%m-%d")
    
    full_text = (
        f"SUPER MACRO DASHBOARD: EXECUTIVE SUMMARY ({date_str})\n"
        f"{'='*80}\n"
        f"{synthesis_text}\n"
        f"{'='*80}\n"
        f"THE VIEW (Legacy): {generate_narrative_summary(df, alpha_data)}\n"
        f"{'='*80}\n"
        f"MACRO REGIME SCORE: {score:.2f} -> {score_status}\n"
        f"LIQUIDITY VALVE: {btc_trend} (Real Yield: {real_yield:.2f}%)\n"
        f"MARKET FRAGILITY: {fragility_commentary}\n"
        f"CONSUMER HEALTH:  {cons_trend} ({cons_commentary})\n"
        f"PLUMBING STATUS:  {plumbing_status} (NFCI: {last['CF_NFCI']:.2f})\n\n"
        f"{valve_commentary}\n"
        f"{'='*80}\n"

        f"{strat_text}"
        f" EQUITY ROTATIONS (Trend vs 200DMA):\n"
        f"   - Value vs Growth: {val_growth}\n"
        f"   - Cyclical vs Def: {cyc_def}\n"
        f"   - Small vs Large:  {small_large}\n"
        f"   - HiBeta vs LoVol: {hibeta_lovol}\n\n"
        f" ACTIONABLE PLAYBOOK ({regime_name}):\n"
        f"   LONGS: {', '.join(longs)}\n"
        f"   SHORTS/AVOID: {', '.join(shorts)}\n\n"
        f" WATCH: Real Yields, USD Index, High Yield Spreads.\n"
        f" INVALIDATION: {invalidation_text}\n"
        f"{'-'*80}\n"
        f" STATISTICAL ANOMALIES (2-Sigma Extremes)\n"
        f"{anomaly_text}\n"
        f"{'-'*80}\n"
        f" PREDICTIVE ANALYTICS MODEL\n"
        f"{pred_text}\n"
        f"{'-'*80}\n"
        f" GLOBAL MACRO & FX REGIME\n"
        f"{global_text}\n"
        f"{'-'*80}\n"
        f" {cta_text}\n"
        f"{'-'*80}\n"
        f" {opt_text}\n"
        f"{'-'*80}\n"
        f" CROSS-ASSET VOLATILITY REGIME\n"
        f"{vol_text}\n"
        f"{'-'*80}\n"
        f" RISK MANAGEMENT (VaR)\n"
        f"{risk_text}\n"
        f"{'-'*80}\n"
        f" MACRO RISK WATCH\n"
        f"1. YIELD CURVE (10Y-2Y): {(last['10Y_Yield'] - last['2Y_Yield']):.2f}% -> {'INVERTED' if (last['10Y_Yield'] - last['2Y_Yield']) < 0 else 'NORMAL'}\n"
        f"2. CREDIT SPREADS (HY):  {last.get('HY_Spread', 0):.2f}% -> {'STRESSED' if last.get('HY_Spread', 0) > 5.0 else 'CALM'}\n\n"
        f"{'-'*80}\n"
        f" MONETARY PLUMBING & ECONOMY\n"
        f"1. LIQUIDITY IMPULSE (YoY):\n"
        f"   - M2 Money Supply: {last.get('M2_YoY', 0):.1%} -> {'EXPANSION' if last.get('M2_YoY', 0) > 0 else 'CONTRACTION'}\n"
        f"   - Fed Balance Sheet: {last.get('Fed_Assets_YoY', 0):.1%} (QT/QE Monitor)\n"
        f"2. ECONOMIC HEALTH:\n"
        f"   - CPI Inflation (YoY): {last.get('CPI_YoY', 0):.1%}\n"
        f"   - Unemployment Rate:   {last.get('Unemployment', 0):.1%}\n\n"
        f"{'-'*80}\n"

        f"{'-'*80}\n"
        f" QUANT LAB INSIGHTS\n"
        f"{quant_text}"
        f" QUANT & OPTIONS STRUCTURE\n"
        f"1. VIX TERM STRUCTURE: {'BACKWARDATION (Panic)' if last['CF_VIX'] > last['CF_VIX3M'] else 'NORMAL (Contango)'}\n"
        f"   (Spot VIX: {last['CF_VIX']:.2f} vs 3M VIX: {last['CF_VIX3M']:.2f})\n"
        f"{'-'*80}\n"
        f" FLOWS & POSITIONING (New Alpha Factors)\n"
        f"1. NET LIQUIDITY (Fed-TGA-RRP): ${alpha_data.get('net_liquidity', {}).get('latest', 0):.2f}T "
        f"-> {alpha_data.get('net_liquidity', {}).get('status', 'N/A')}\n"
        f"2. CRASH SIGNAL (VIX Term Struct): {alpha_data.get('vol_structure', {}).get('latest', 0):.2f} "
        f"-> {alpha_data.get('vol_structure', {}).get('signal', 'N/A')}\n"
        f"3. TAIL RISK (SKEW): {alpha_data.get('tail_risk', {}).get('latest', 0):.2f} "
        f"-> {alpha_data.get('tail_risk', {}).get('signal', 'N/A')}\n"
    )
    return full_text

def generate_pdf_report(df: pd.DataFrame, prices: pd.DataFrame, figures: Dict[str, plt.Figure], alpha_data: Dict = {}, cta_data: pd.DataFrame = None, opt_data: Dict = None, pred_data: Dict = None):
    """Generates the multi-page PDF report."""
    # Determine Output Path (Project Root)
    import os
    # Get directory of this file (plotting/report.py) -> go up one level to Project Root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_path = os.path.join(project_root, "Macro_Dashboard_Report.pdf")
    
    print(f"Generating PDF Report: {output_path}...")
    
    with PdfPages(output_path) as pdf:
        # Page 1+: Text Commentary (Paginated)
        
        full_text = generate_capital_flows_commentary(df, prices, alpha_data, cta_data, opt_data, pred_data)
        
        # 1. Wrap text horizontally first
        wrapped_lines = []
        for line in full_text.split('\n'):
            # Wrap at 95 chars to fit page width
            wrapped_lines.extend(textwrap.wrap(line, width=95, replace_whitespace=False, drop_whitespace=False))
            if not line: # Preserve empty lines
                wrapped_lines.append("")
        
        # 2. Paginate vertically
        lines_per_page = 46 # Reduced from 55 to prevent cut-off 
        
        for i in range(0, len(wrapped_lines), lines_per_page):
            chunk = wrapped_lines[i:i + lines_per_page]
            page_text = "\n".join(chunk)
            
            fig_text = plt.figure(figsize=(11, 8.5)) # Letter size
            fig_text.clf()
            
            # Add header for continuation pages
            if i > 0:
                page_text = f"--- REPORT CONTINUED (Page {i//lines_per_page + 1}) ---\n\n" + page_text
            else:
                # Add Title on First Page
                plt.text(0.5, 0.96, "MACRO ROTATIONS & LIQUIDITY DASHBOARD", transform=fig_text.transFigure, ha='center', fontsize=14, weight='bold')
            
            fig_text.text(0.05, 0.92, page_text, transform=fig_text.transFigure, size=9, ha="left", va="top", family="monospace")
            plt.axis('off')
            pdf.savefig(fig_text)
            plt.close(fig_text)
        
        # --- SECTION 1: GLOBAL MACRO (The Economy) ---
        


        # Page 3: Predictive Analytics (Recession & Internals)
        if "predictive" in figures:
            pdf.savefig(figures["predictive"])
            plt.close(figures["predictive"])

        # Page 4: Inflation Expectations (The Anchor)
        if "inflation_swaps" in figures:
            pdf.savefig(figures["inflation_swaps"])
            plt.close(figures["inflation_swaps"])

        # Page 5: Monetary Plumbing (Liquidity & Rates)
        if "plumbing" in figures:
            pdf.savefig(figures["plumbing"])
            plt.close(figures["plumbing"])

        # Page 5b: Global FX & Rates (New)
        if "global_fx" in figures:
            pdf.savefig(figures["global_fx"])
            plt.close(figures["global_fx"])

        # Page 6: Valuation & Real Rates (Cost of Capital)
        if "valuation" in figures:
            pdf.savefig(figures["valuation"])
            plt.close(figures["valuation"])

        # --- SECTION 2: MARKET RISK & STRUCTURE (The Market) ---

        # Page 7: Macro Risk Dashboard (Yields, Spreads)
        if "risk_macro" in figures:
            pdf.savefig(figures["risk_macro"])
            plt.close(figures["risk_macro"])

        # Page 8: Cross-Asset Correlation (Systemic Risk)
        if "cross_asset" in figures:
            pdf.savefig(figures["cross_asset"])
            plt.close(figures["cross_asset"])

        # Page 9: Forward Models (Regime, Rotation)
        if "forward" in figures:
            pdf.savefig(figures["forward"])
            plt.close(figures["forward"])

        # --- SECTION 3: INSTITUTIONAL FLOWS (The Players) ---

        # Page 10: Alpha Factors (Liquidity, Vol, SKEW)
        if "alpha" in figures:
            pdf.savefig(figures["alpha"])
            plt.close(figures["alpha"])

        # --- SECTION 4: MICRO QUANT LAB (The Asset - SPY) ---

        # Page 11: Quant Lab Dashboard
        if "quant" in figures:
            pdf.savefig(figures["quant"])
            plt.close(figures["quant"])

        # Page 12: Monte Carlo Cone
        if "monte_carlo" in figures:
            pdf.savefig(figures["monte_carlo"])
            plt.close(figures["monte_carlo"])

        # Page 13: Advanced Stochastic
        if "stochastic" in figures:
            pdf.savefig(figures["stochastic"])
            plt.close(figures["stochastic"])
            
        # Page 14: Mean Reversion
        if "mean_reversion" in figures:
            pdf.savefig(figures["mean_reversion"])
            plt.close(figures["mean_reversion"])
            
        # Page 15: Microstructure & Jumps
        if "microstructure" in figures:
            pdf.savefig(figures["microstructure"])
            plt.close(figures["microstructure"])
            
        # Page 16: Anti-Fragility
        if "antifragility" in figures:
            pdf.savefig(figures["antifragility"])
            plt.close(figures["antifragility"])

        # Page 17: Scenario Analysis
        if "scenarios" in figures:
            pdf.savefig(figures["scenarios"])
            plt.close(figures["scenarios"])

        # --- SECTION 5: PORTFOLIO CONSTRUCTION (The Solution) ---

        # Page 18: Efficient Frontier
        if "frontier" in figures:
            pdf.savefig(figures["frontier"])
            plt.close(figures["frontier"])
            
        # Backtest (Optional/Appendix)
        if "backtest" in figures:
            pdf.savefig(figures["backtest"])
            plt.close(figures["backtest"])

    print("PDF Report Saved Successfully!")
