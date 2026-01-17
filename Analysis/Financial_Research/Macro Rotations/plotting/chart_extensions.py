
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
