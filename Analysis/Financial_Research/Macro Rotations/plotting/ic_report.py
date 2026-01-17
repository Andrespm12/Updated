"""
Investment Committee (IC) Daily Briefing Generator - "The Morning Reader" Edition.
Synthesizes quantitative models into a deep-dive narrative document.
"""
import textwrap
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os

# --- NARRATIVE ENGINES ---

def determine_stance_details(df: pd.DataFrame, alpha_data: dict) -> tuple[str, str, str]:
    """Returns (Stance Title, Stance Subtitle, Color)."""
    last = df.iloc[-1]
    score = last["MACRO_SCORE"]
    
    liq_status = alpha_data.get("net_liquidity", {}).get("status", "NEUTRAL")
    vol_sig = alpha_data.get("vol_structure", {}).get("signal", "NORMAL")
    
    if "CRASH" in vol_sig:
        return "MAXIMUM DEFENSIVE", "CRASH RISK IMMINENT", "darkred"
    elif score > 0.5 and "EXPANDING" in liq_status:
        return "AGGRESSIVE BULLISH", "Go for Leverage", "darkgreen"
    elif score > 0.2:
        return "BULLISH ACCUMULATION", "Buy the Dips", "forestgreen"
    elif score < -0.2:
        return "BEARISH DISTRIBUTION", "Sell Rallies", "firebrick"
    else:
        return "NEUTRAL / TACTICAL", "Stock Picker's Market", "chocolate"

def generate_macro_deep_dive(df: pd.DataFrame, prices: pd.DataFrame) -> str:
    """Generates the Main Story: A deep dive into the 'Why'."""
    last = df.iloc[-1]
    
    # Data points
    cpi = last.get("CPI_YoY", 0)
    unemp = last.get("Unemployment", 0)
    yield_10y = last.get("10Y_Yield", 0)
    score = last["MACRO_SCORE"]
    
    # Inflation & Fed Narrative
    inf_narrative = ""
    if cpi > 0.035:
        inf_narrative = (f"Inflation remains stubbornly high at {cpi:.1%}. This sticky inflation forces the Federal Reserve "
                         "to keep interest rates 'High for Longer,' acting as a gravity well on asset valuations. "
                         "Until we see a decisive break below 3%, any valuation expansion is likely temporary.")
    elif cpi < 0.02:
        inf_narrative = (f"Inflation has cooled significantly to {cpi:.1%}. This gives the Fed the 'Green Light' to cut rates "
                         "to support growth. In this environment, bad economic news is actually good for markets, as it accelerates "
                         "the pivot to easy money.")
    else:
        inf_narrative = (f"Inflation is stabilizing in the 'Goldilocks' zone ({cpi:.1%}). This suggests the economy is normalizing "
                         "without a recession, supporting a 'Soft Landing' thesis where earnings growth drives stock prices rather than Fed liquidity.")

    # Rates & Growth Interaction
    rate_narrative = ""
    if yield_10y > 4.5:
        rate_narrative = (f"However, the 10-Year Treasury Yield is pressing uncomfortably high at {yield_10y:.2f}%. "
                          "At this level, risk-free bonds offer a compelling alternative to stocks, compressing the Equity Risk Premium. "
                          "Tech and long-duration growth assets face significant headwinds here.")
    elif yield_10y < 3.5:
        rate_narrative = (f"The 10-Year Yield has relaxed to {yield_10y:.2f}%, providing breathing room for valuation multiples. "
                          "Lower rates essentially lower the discount rate for future cash flows, disproportionately benefiting Growth and Tech sectors.")
    else:
        rate_narrative = f"Rates are range-bound around {yield_10y:.2f}%, having a neutral impact on valuations for now."

    # Synthesis
    synthesis = ""
    if score > 0:
        synthesis = ("Overall, the macro data suggests the Bulls are in control. The combination of resilient growth "
                     "and manageable inflation creates a 'Risk-On' feedback loop. Investors should focus on riding momentum.")
    else:
        synthesis = ("Overall, the macro backdrop is deteriorating. The weight of tight financial conditions is starting "
                     "to crack the surface of the economy. Caution is warranted as the risk of a policy error is elevated.")

    full_text = f"{inf_narrative}\n\n{rate_narrative}\n\n{synthesis}"
    return full_text

def generate_liquidity_analysis(alpha_data: dict, df: pd.DataFrame) -> str:
    """Explains the Liquidity & Volatility picture."""
    # Liquidity
    net_liq = alpha_data.get("net_liquidity", {}).get("latest", 0)
    liq_trend = alpha_data.get("net_liquidity", {}).get("status", "FLAT")
    
    liq_text = (f"Global Liquidity is currently {liq_trend}. The 'Net Liquidity' (Fed Balance Sheet minus TGA/RRP) "
                f"stands at ${net_liq:.2f} Trillion. ")
    
    if "EXPANDING" in liq_trend:
        liq_text += ("This expansion acts as a rising tide that lifts all boats, particularly speculative assets like Crypto and Tech. "
                     "Don't fight the Fed's liquidity hose.")
    elif "CONTRACTING" in liq_trend:
        liq_text += ("Liquidity is being drained from the system. This removal of support often exposes weak hands and over-leveraged "
                     "positions. It is a headwind for P/E expansion.")
    else:
        liq_text += "Liquidity is stable, meaning market Alpha will come from earnings and idiosyncratic stories rather than a systemic push."

    # Volatility
    vix = df.iloc[-1].get("CF_VIX", 0)
    vol_text = f"\n\nVolatility (VIX) is at {vix:.2f}. "
    if vix < 13:
        vol_text += ("This extreme complacency is dangerous. Markets are priced for perfection, meaning any small shock could cause "
                     "a disproportionate sell-off (the 'Vol Minsky Moment'). Hedging is cheap and recommended.")
    elif vix > 25:
        vol_text += ("Fear is high. While uncomfortable, this is where long-term bottoms are formed. Look for capitulation signals to buy.")
    else:
        vol_text += "This is a normal volatility regime, conducive to standard trend-following strategies."

    return liq_text + vol_text

def generate_detailed_playbook(stance: str, df: pd.DataFrame, prices: pd.DataFrame) -> str:
    """Generates a detailed, reasoned playbook."""
    actions = []
    
    # Equity Strategy
    if "BULLISH" in stance:
        actions.append(("EQUITIES: OVERWEIGHT. Focus on 'High Beta' sectors (Technology, Discretionary) that benefit from the risk-on regime. "
                        "The trend is your friend; add to winners on pullbacks."))
    elif "BEARISH" in stance:
        actions.append(("EQUITIES: UNDERWEIGHT. It is time to play defense. Rotate into 'quality' factors and defensive sectors "
                        "(Staples, Healthcare, Utilities) which have stable cash flows."))
    else:
        actions.append(("EQUITIES: NEUTRAL. The index is likely to chop. Alpha will be found in stock selection/rotation rather than broad beta."))

    # Rate Strategy
    yield_10y = df["10Y_Yield"].iloc[-1]
    if yield_10y > 4.2:
        actions.append(("FIXED INCOME: BUY DURATION. With yields > 4.2%, bonds offer equity-like returns with lower risk. "
                        "We recommend building long positions in TLT/IEF."))
    else:
        actions.append(("FIXED INCOME: NEUTRAL. Yields are not yet attractive enough to warrant a major aggression given the inflation volatility."))

    # Hedging
    vix = df["CF_VIX"].iloc[-1]
    if vix < 14:
        actions.append(("HEDGING: AGGRESSIVE. Volatility is at multi-year lows. Buying long-dated Puts on SPY/QQQ is historically cheap insurance. "
                        "We recommend 'Risk Reversals' to fund the downside protection."))
    else:
        actions.append(("HEDGING: TACTICAL. Maintain standard stop-losses. No need for expensive tail hedges unless VIX term structure inverts."))

    return "\n\n".join([f"> {a}" for a in actions])

def generate_ic_report(df: pd.DataFrame, prices: pd.DataFrame, macro: pd.DataFrame, alpha_data: dict, pred_data: dict):
    """Main function to generate the IC Briefing PDF."""
    
    # 1. Synthesis
    date_str = dt.datetime.today().strftime("%A, %B %d, %Y")
    stance_title, stance_sub, stance_color = determine_stance_details(df, alpha_data)
    
    macro_story = generate_macro_deep_dive(df, prices)
    liq_risk_story = generate_liquidity_analysis(alpha_data, df)
    playbook = generate_detailed_playbook(stance_title, df, prices)
    
    # 2. PDF Generation
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_path = os.path.join(project_root, "Market_IC_Brief.pdf")
    
    print(f"Generating Enhanced IC Briefing: {output_path}...")
    
    with PdfPages(output_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11)) # Portrait
        
        # --- MASTHEAD ---
        plt.text(0.5, 0.96, "THE MORNING READER", ha='center', fontsize=26, weight='bold', fontname='serif', transform=fig.transFigure)
        plt.text(0.5, 0.93, "MACRO ROTATIONS INVESTMENT COMMITTEE", ha='center', fontsize=10, weight='bold', transform=fig.transFigure)
        plt.text(0.5, 0.91, f"INTTELLIGENCE BRIEFING  |  {date_str}  |  VOL. 1", ha='center', fontsize=9, style='italic', transform=fig.transFigure)
        
        # Divider
        plt.plot([0.05, 0.95], [0.89, 0.89], color='black', lw=2, transform=fig.transFigure, clip_on=False)
        plt.plot([0.05, 0.95], [0.885, 0.885], color='black', lw=0.5, transform=fig.transFigure, clip_on=False)

        # --- HEADLINE ---
        plt.text(0.05, 0.84, f"VERDICT: {stance_title}", fontsize=18, weight='heavy', color=stance_color, transform=fig.transFigure)
        plt.text(0.05, 0.81, f"SUB-THEME: {stance_sub}", fontsize=12, style='italic', color='black', transform=fig.transFigure)
        
        # --- COLUMN 1: THE MACRO STORY ---
        plt.text(0.05, 0.76, "THE MACRO DEEP DIVE", fontsize=12, weight='bold',  fontname='serif', style='italic', transform=fig.transFigure)
        
        # We use textwrap to simulate column
        wrap_width = 50 # Approx chars for half page
        wrapped_macro = "\n".join(textwrap.wrap(macro_story, width=wrap_width))
        
        plt.text(0.05, 0.74, wrapped_macro, fontsize=10, va='top', fontname='serif',  linespacing=1.4, transform=fig.transFigure)

        # --- COLUMN 2: LIQUIDITY & RISK ---
        plt.text(0.52, 0.76, "LIQUIDITY & RISK LAB", fontsize=12, weight='bold', fontname='serif',  style='italic', transform=fig.transFigure)
        
        wrapped_liq = "\n".join(textwrap.wrap(liq_risk_story, width=wrap_width))
        plt.text(0.52, 0.74, wrapped_liq, fontsize=10, va='top', fontname='serif', linespacing=1.4, transform=fig.transFigure)
        
        # --- DATA STRIP (Middle) ---
        plt.plot([0.05, 0.95], [0.45, 0.45], color='gray', lw=0.5, transform=fig.transFigure, clip_on=False)
        
        # Mini dashboard in the middle
        data_text = (f"MACRO SCORE: {df['MACRO_SCORE'].iloc[-1]:.2f}  |  "
                     f"VIX: {df['CF_VIX'].iloc[-1]:.2f}  |  "
                     f"10Y YIELD: {df['10Y_Yield'].iloc[-1]:.2f}%  |  "
                     f"CPI: {df['CPI_YoY'].iloc[-1]:.1%}  |  "
                     f"RECESSION PROB: {pred_data.get('recession_prob', pd.Series([0])).iloc[-1]:.1f}%")
        
        plt.text(0.5, 0.43, data_text, ha='center', fontsize=9, weight='bold', fontname='monospace', 
                 bbox=dict(facecolor='whitesmoke', edgecolor='none', pad=4), transform=fig.transFigure)

        plt.plot([0.05, 0.95], [0.41, 0.41], color='gray', lw=0.5, transform=fig.transFigure, clip_on=False)

        # --- BOTTOM SECTION: THE PLAYBOOK ---
        plt.text(0.05, 0.38, "THE STRATEGIC PLAYBOOK", fontsize=14, weight='bold', fontname='sans-serif', transform=fig.transFigure)
        
        wrapped_play = textwrap.fill(playbook, width=90) # Wider for bottom section
        plt.text(0.05, 0.35, wrapped_play, fontsize=11, va='top', fontname='serif', linespacing=1.5, transform=fig.transFigure)
        
        # --- FOOTER ---
        plt.text(0.5, 0.04, "Page 1 of 1  |  Generated by Macro Rotations AI  |  Confidential", ha='center', fontsize=8, color='gray', transform=fig.transFigure)

        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)
        
    print("Enhanced IC Briefing Saved Successfully!")
