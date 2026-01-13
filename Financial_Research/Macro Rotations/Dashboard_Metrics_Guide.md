# Macro Rotations Dashboard: User Guide

This document explains the metrics, models, and data points presented in the Macro Rotations Dashboard Report (`Macro_Dashboard_Report.pdf`).

## Page 1: Executive Summary & Market Commentary

This section provides a high-level overview of the current macroeconomic environment using a "Capital Flows" framework.

### 1. Macro Regime Score
- **What it is**: A composite score ranging from -1.0 to +1.0 derived from 14 different trend and ratio indicators.
- **Interpretation**:
    - **> 0.3 (RISK-ON)**: Bullish. Growth is expanding, liquidity is ample.
    - **< -0.3 (RISK-OFF)**: Bearish. Defensive positioning is favored.
    - **Between -0.3 and 0.3**: Neutral/Choppy. No clear trend.

### 2. Liquidity Valve (BTC vs Gold)
- **What it is**: The ratio of Bitcoin (Risk/Speculative Liquidity) to Gold (Safety/Debasement).
- **Signal**:
    - **OPEN**: BTC outperforming Gold Trend (MA200). Signals that excess liquidity is flowing into risk assets.
    - **CLOSED**: BTC underperforming. Liquidity is drying up or fleeing to safety.
- **Context**: Checked against Real Yields. If Valve is OPEN but Real Yields are HIGH (>2%), it warns of a "Bull Trap" or unsustainable speculation.

### 3. Market Fragility (Breadth vs VIX)
- **What it is**: Compares Market Breadth (Eq Weight S&P 500 relative to Cap Weight) with the VIX (Fear Index).
- **Key Warning**: **"NARROW" Breadth + Low VIX (<15)** = **High Fragility**. This "Divergence" often precedes sharp market corrections (air pockets). It means the market is held up by a few mega-caps while investors are complacent.

### 4. Consumer Health (XLY vs XLP)
- **What it is**: The ratio of Consumer Discretionary (Cyclical) to Consumer Staples (Defensive).
- **Interpretation**:
    - **EXPANSION**: Consumers are spending on discretionary items. Economy is resilient.
    - **RETRENCHMENT**: Consumers are shifting to staples (needs). Signals tightening budgets and potential recession risk.

### 5. Macro Radar Chart
- **What it is**: A visual polygon representing the "Shape" of the macro regime across 6 dimensions: **Growth, Liquidity, Risk, Inflation, Rates, Sentiment**.
- **How to Read**:
    - **Expanded (Green)**: Risk-Friendly (High Growth, High Liquidity, Low Inflation).
    - **Contracted (Red)**: Restrictive (Low Growth, Tight Liquidity, High Inflation).
    - **Divergences**: Watch if the "Liquidity" axis collapses while "Growth" is still expanding (early warning).

### 6. Actionable Playbook
- **What it is**: Automated investment ideas based on the current regime.
- **Longs/Shorts**: Specific asset classes or sectors to Overweight (Long) or Underweight (Short).
- **Invalidation**: Critical levels that, if breached, prove the current thesis wrong (Stop Loss logic).

---

## Page 2: Backtest & 1-Year Projection

- **Equity Curves (Log Scale)**: Shows the historical performance of various strategies (e.g., "Macro Regime", "Liquidity Valve") vs "SPY (Hold)".
- **Median Path Projection**: A Monte Carlo simulation (1000 runs) projecting the likely future path of these strategies over the next year.
    - **Dotted Line**: The forecasted median outcome.
- **Table**:
    - **CAGR**: Compound Annual Growth Rate.
    - **Sharpe**: Risk-adjusted return (Higher is better).
    - **MaxDD**: Maximum Drawdown (Worst peak-to-trough loss).

---

## Page 3: Risk & Sector Rotation

### 1. Yield Curve (10Y - 2Y)
- **What it is**: Difference between 10-Year and 2-Year Treasury Yields.
- **Signal**: **Inversion (< 0)** is a statistically reliable predictor of an upcoming Recession (lag 12-18 months).

### 2. Credit Stress (HY Spreads)
- **What it is**: The yield premium demanded for High Yield (Junk) bonds vs Treasuries.
- **Signal**: Spikes above 5.0% indicate credit stress and often accompany equity sell-offs.

### 3. Bond Market Fear (MOVE Index)
- **What it is**: The VIX for Bonds. Measures implied volatility in Treasury options.
- **Signal**: **> 100** indicates Bond Market instability. High MOVE often leaks into Equity Volatility (VIX).

### 4. Sector Rotation Map (RRG Proxy)
- **What it is**: A scatter plot showing **Relative Strength (RS)** vs **Momentum**.
- **Quadrants**:
    - **LEADING (Top Right)**: Strong trend + Strong momentum. (Buy/Hold)
    - **WEAKENING (Bottom Right)**: Strong trend but losing momentum. (Trim)
    - **LAGGING (Bottom Left)**: Weak trend + Weak momentum. (Avoid/Short)
    - **IMPROVING (Top Left)**: Weak trend but gaining momentum. (Watch/Buy)

---

## Page 4: Monetary Plumbing & Economy

### 1. Liquidity Impulse
- **M2 Money Supply (YoY)**: The raw fuel for asset prices. >0% = Expansion.
- **Fed Balance Sheet (YoY)**: Measures Quantitative Tightening (QT) or Easing (QE).

### 2. Global Liquidity Wrecking Ball (DXY)
- **What it is**: US Dollar Index.
- **Signal**: A rapidly rising Dollar tightens global financial conditions, hurting Emerging Markets and Risk Assets (Inverse correlation with Stocks).

---

## Quant Lab Insights (New)

### Volatility Regimes (GARCH)
- **GARCH(1,1)**: A statistical model that estimates the "True" volatility, which tends to cluster (calm follows calm, storm follows storm).
- **Vol Trend**: Is volatility EXPANDING or CONTRACTING?
- **Playbook**:
    - **Low/Stable Vol**: Buy Options (Long Calls/Puts) as they are cheap.
    - **High/Expanding Vol**: Sell Options (Credit Spreads, Iron Condors) to harvest premium.

### Page 7: Quant Lab & Option Strategies
- **Multi-Asset Regime Analysis**:
  - **Bias**: Directional drift derived from Monte Carlo simulations.
  - **Vol Regime**: GARCH(1,1) model state (High/Low, Expanding/Contracting).
  - **Strategy**: Recommended option structure (e.g., "Long Call Spread", "Iron Condor") based on the Bias/Vol mix.
- **SPY ATM Greeks**: Theoretical sensitivity of a 30-day At-The-Money call option.
  - **Delta**: Directional risk.
  - **Gamma**: Acceleration risk.
  - **Vega**: Volatility exposure.
  - **Theta**: Time decay cost.

### Page 8: Institutional Alpha Factors (New)
This section tracks "smart money" flows and structural market imbalances.
- **1. Net Liquidity (The Howell/Dale Model)**:
  - **Formula**: `Fed Balance Sheet - Treasury General Account (TGA) - Reverse Repo (RRP)`.
  - **Concept**: Not all money printed by the Fed reaches the stock market. Money trapped in the TGA (Government Checking) or RRP (Money Market Parking) is **dead** liquidity. "Net Liquidity" is what remains to fuel asset prices.
  - **Interpretation**: If Net Liquidity is rising, risk assets (Stocks/Crypto) are supported. If falling, expect headwinds regardless of earnings.
- **2. Volatility Term Structure (Crash Signal)**:
  - **Metric**: `VIX (Spot) / VIX3M (3-Month Future)`.
  - **Logic**: In normal markets ("Contango"), immediate fear (VIX) is lower than future uncertainty (VIX3M). The ratio is < 1.0.
  - **Signal**: When the ratio flips > 1.0 ("Backwardation"), it means panic is immediate. This is a highly accurate predictor of market crashes or sharp corrections.
- **3. Tail Risk Monitor (Whale Positioning)**:
  - **Metric**: CBOE SKEW Index.
  - **Concept**: Measures the price of "disaster insurance" (deep out-of-the-money puts) relative to normal calls.
  - **Interpretation**: 
    - **> 135 (High Risk)**: Institutional whales are secretly terrified and buying crash protection, even if the market is going up (Bearish Divergence).
    - **< 115 (Complacency)**: No one expects a crash. Often precedes a "Rug Pull".

### Page 9: Cross-Asset Regime (Correlations)
This page checks if diversification is working or if macro drivers (Rates/Dollar) are breaking everything.
- **Top Panel: Correlation Matrix (30-Day)**:
  - **Heatmap**: Shows relationships between SPY, TLT (Bonds), GLD (Gold), UUP (Dollar), and BTC.
  - **Dark Red**: Moving together (High Risk if markets fall).
  - **Blue**: Moving opposite (Good diversification).
- **Bottom Panel: SPY vs TLT Rolling Correlation**:
  - **Green Zone (< -0.5)**: Healthy. Stocks Up, Bonds Down (Growth driven) OR Stocks Down, Bonds Up (Flight to Safety). Portfolios are protected.
  - **Red Zone (> 0.5)**: Danger. Stocks and Bonds move together. Usually driven by Inflation or Liquidity shocks. Bonds will NOT protect your stocks.

### CTA Trend Monitor (Text Report)
A systematic "Traffic Light" system for following trends, mimicking Commodity Trading Advisors (CTAs).
- **Short-Term (20D)**: Fast tactical moves.
- **Medium-Term (60D)**: The "meat" of the move.
- **Long-Term (200D)**: Structural Bull/Bear market.
- **Signal**:
  - **STRONG UPTREND**: Price > 20D, 60D, AND 200D Moving Averages.
  - **STRONG DOWNTREND**: Price < 20D, 60D, AND 200D.
  - **NEUTRAL**: Messy/Choppy signals.

### Page 10: Portfolio Optimization (Macro-Adjusted)
This page acts as a "Quant Advisor," simulating thousands of possible portfolios to find the mathematically optimal mix.
**Key Upgrade:** This is NOT just a historical optimization. It uses a **Black-Litterman Lite** approach, where the Expected Returns of assets are adjusted based on the current **Macro Regime Score**.
-   **Risk-On Regime**: The model assumes Stocks and Crypto will outperform Bonds.
-   **Risk-Off Regime**: The model assumes Bonds, Gold, and USD will outperform.
-   **Inflationary Regime**: The model boosts Hard Assets (Gold) and penalizes long-duration Bonds.

- **Efficient Frontier Chart**:
  - **Scatter Plot**: 5,000 random portfolios plotted by Risk (X-axis) vs Return (Y-axis).
  - **The "Frontier"**: The top-Left edge of the cloud represents the best possible returns for a given level of risk.
- **Optimal Portfolios**:
  - **Max Sharpe (Star)**: The "Goldilocks" portfolio. The best risk-adjusted return. Best for long-term growth.
  - **Min Volatility (Circle)**: The "safe haven". The portfolio with the absolute lowest calculated risk. Best for preservation.

### Page 11: Predictive Analytics Model (NY Fed)
This page focuses on forward-looking recession risks and leading economic indicators.
- **NY Fed Recession Probability**: The probability of a US recession in the next 12 months based on the Treasury Yield Curve.
  - **> 30%**: High Risk. Proceed with caution.
- **Leading Economic Indicators (LEI)**: Tracks the trend of key data points like Housing Starts, Manufacturing, and Consumer Sentiment.

---

## Page 12: Monte Carlo Simulation (Cone of Uncertainty)
- **What it is**: A stochastic simulation (1000 runs) of future price paths for SPY.
- **The "Cone"**:
  - **Dark Blue Line**: Median expected path.
  - **Shaded Areas**: Confidence intervals (50% and 95%).
## Page 2: Macro Regime Radar (State of the World)
**Goal:** Assess the current global macro environment at a glance.
- **Business Cycle**: Where are we? (Expansion, Slowdown, Recession, Recovery).
- **Inflation Regime**: Disinflation vs Reflation.
- **Policy**: Tight (Restrictive) vs Loose (Accommodative).

## Page 3: Predictive Analytics
**Goal:** Forward-looking risk assessment.
- **Recession Probability**: Derived from Yield Curve (10Y-3M).
- **Market Internals**: Cyclical vs Defensive rotation.

## Page 4: Inflation Expectations (The Anchor)
**Goal:** Track market-implied inflation expectations.
- **5Y Breakeven**: Medium-term market view.
- **5Y5Y Forward**: Long-term anchor. If this de-anchors > 2.5%, Fed must be aggressive.

## Page 5: Monetary Plumbing
**Goal:** Monitor the "pipes" of the financial system.
- **Net Liquidity**: Fed Balance Sheet - TGA - RRP.
- **SOFR Spread**: Stress in overnight lending markets.

## Page 6: Valuation & Real Rates
**Goal:** Assess the cost of capital and equity valuation.
- **Real Yield (10Y - Inflation)**: The true cost of money. >2% is restrictive.
- **ERP (Equity Risk Premium)**: Earnings Yield - 10Y Yield. Low ERP = Stocks are expensive relative to bonds.

## Page 7: Macro Risk Dashboard
**Goal:** High-level dashboard of systemic stress.
- **Credit Spreads (HYG/IEF)**: Is credit stress rising?
- **MOVE Index**: Bond market volatility (the "Vol of Vol").

## Page 8: Cross-Asset Correlation
**Goal:** Understand how asset classes are moving together.
- **Stock/Bond Correlation**:
  - Positive: Inflation fears dominate (Bad for 60/40 portfolios).
  - Negative: Growth fears dominate (Bonds hedge Stocks).

## Page 9: Forward Looking Models
**Goal:** Leading indicators for sector rotation.
- **Sector Rotation Model**: Which sectors benefit from the current regime? (e.g., Energy in Inflation, Tech in Growth).

## Page 10: Institutional Alpha Factors
**Goal:** Track what drives smart money flows.
- **VIX Term Structure**: Contango (Normal) vs Backwardation (Crash Risk).
- **SKEW Index**: Demand for tail-risk hedging. High SKEW = Nervousness.

## Page 11: Quant Lab Dashboard
**Goal:** Micro-structure and options analytics for SPY.
- **Gamma Exposure (GEX)**: Dealer hedging flows. Be careful at "Flip Points".
- **Vol Control**: Systematic fund leverage targets.

## Page 12: Monte Carlo Simulation (Cone)
**Goal:** Project future price paths based on volatility.
- **Cone Analysis**: Are we near the top (extended) or bottom (oversold) of the probable range?

## Page 13: Stochastic Models
**Goal:** Advanced volatility modeling.
- **Heston Volatility**: Is volatility itself trending up?
- **Regime Switching (HMM)**: Are we in a "Calm" or "Turbulent" state?

## Page 14: Mean Reversion
**Goal:** Identify overstretched moves.
- **Z-Scores**: Statistical distance from the mean. >2 Sigma = Reversion likely.

## Page 15: Market Microstructure
**Goal:** Assess liquidity and crash risk.
- **Merton Jumps**: Probability of sudden "gap" moves.
- **Hawkes Process**: Volatility clustering and feedback loops.

## Page 16: Anti-Fragility
**Goal:** Stress-test the portfolio structure.
- **Kurtosis**: Fat tail risk.
- **Taleb Ratio**: Upside volatility vs Downside volatility.

---

## Page 17: Scenario Analysis (Stress Testing)
- **What If?**: Simulates how the portfolio would perform in historical crises.
- **Scenarios**:
  - **2008 GFC**: Global Financial Crisis.
  - **2020 Covid**: Pandemic crash.
  - **2022 Inflation**: Rate hike cycle.
- **Goal**: Ensure the portfolio survives the "Worst Case" (Max Drawdown matches or beats the benchmark).

---

## Page 18: Portfolio Optimization (Efficient Frontier)
**Goal:** Portfolio Construction and Optimization.
- **Efficient Frontier Curve**: Shows the best possible return for a given level of risk.
- **Max Sharpe (Star)**: The "Goldilocks" portfolio. The best risk-adjusted return. Best for long-term growth.
- **Min Volatility (Circle)**: The "safe haven". The portfolio with the absolute lowest calculated risk. Best for preservation.
