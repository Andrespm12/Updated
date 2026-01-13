"""
Global Macro Equity Rotations Dashboard (Extended + FRED + Macro Score)
----------------------------------------------------------------------
Paste this into a Python cell in Google Antigravity.

What it does:
- Downloads ETF/stock data with yfinance
- Builds key macro rotation ratios:

    Core US equity internals
    ------------------------
    * XLY/XLP          -> Cyclicals vs Defensives (Growth expectations)
    * RPV/VONG         -> Value vs Growth (Rates / inflation / real yields)
    * SPHB/SPLV        -> High Beta vs Low Vol (Risk appetite)
    * IWM/QQQ          -> Small Caps vs Large Caps (Domestic breadth / risk)
    * NOBL/SPY         -> Dividend Aristocrats vs Market (Quality / defense)
    * XBI/SPY          -> Biotech vs Market (Speculative risk appetite)
    * (XLI+XLB)/(XLU+XLP) -> Cyclical sectors vs Defensive sectors
    * Staffing basket vs SPY (MAN + RHI + KELYA) / 3  -> Labor-cycle proxy

    Global & cross-asset extensions
    -------------------------------
    * SPY/ACWX        -> US vs Rest-of-World equities
    * EEM/VEA         -> EM vs DM (ex-US) rotation
    * RSP/SPY         -> Equal-weight vs cap-weight (breadth vs mega-cap)
    * CPER/GLD        -> Copper vs Gold (growth vs safety / inflation vs fear)
    * XLF/XLK         -> Financials vs Tech (rates/curve vs long-duration growth)

    Macro series from FRED
    ----------------------
    * DGS2            -> 2Y Treasury yield
    * DGS10           -> 10Y Treasury yield
    * YC_10Y2Y        -> 10Y - 2Y curve (steepening / inversion)
    * T10YIE          -> 10Y breakeven inflation
    * BAMLH0A0HYM2    -> HY OAS (high yield credit spread)

    Composite signal
    ----------------
    * MACRO_SCORE     -> Composite risk-on score from all ratios (range ~[-1, +1])

- Adds 50d & 200d MAs for each ratio
- Prints latest snapshot + heuristic regime messages
- Prints composite MACRO_SCORE interpretation
- Plots:
    * All ratios + 200d MA
    * FRED macro indicators
    * MACRO_SCORE over time

Requirements (inside Antigravity, if missing):
    pip install yfinance pandas_datareader
"""

import datetime as dt
import textwrap
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError(
        "You need to install yfinance first. "
        "In your Antigravity cell, run:  pip install yfinance"
    ) from e

try:
    from pandas_datareader import data as pdr
except ImportError as e:
    raise ImportError(
        "You need to install pandas_datareader for FRED data. "
        "In your Antigravity cell, run:  pip install pandas_datareader"
    ) from e


# ---------------------------- CONFIG ---------------------------------

CONFIG: Dict = {
    "start_date": "2010-01-01",   # Backtest start
    "end_date": None,             # None = today
    "tickers": [
        # Cyclicals vs Defensives
        "XLY", "XLP",
        # Value vs Growth
        "RPV", "VONG",
        # High Beta vs Low Vol
        "SPHB", "SPLV",
        # Small vs Large
        "IWM", "QQQ",
        # Dividend Aristocrats vs Market
        "NOBL", "SPY",
        # Biotech
        "XBI",
        # Cyclical vs Defensive sectors
        "XLI", "XLB", "XLU",
        # Staffing / HR basket
        "MAN", "RHI", "KELYA",
        # US vs Rest-of-World
        "ACWX",
        # EM vs DM (ex-US)
        "EEM", "VEA",
        # Equal-weight vs cap-weight
        "RSP",
        # Copper vs Gold
        "CPER", "GLD",
        # Financials vs Tech
        "XLF", "XLK",
    ],
    # Moving-average lengths for trend / alerts
    "ma_short": 50,
    "ma_long": 200,
    # Threshold (in %) for regime detection vs MA200
    "regime_threshold_pct": 1.0,
}


# ------------------------ DATA DOWNLOAD ------------------------------


def download_prices(
    tickers: List[str], start: str, end: Optional[str] = None
) -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers using yfinance.
    Returns a DataFrame with columns = tickers, index = Date.
    """
    if end is None:
        end = dt.date.today().strftime("%Y-%m-%d")

    data = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False
    )["Close"]

    # If only one ticker, yfinance returns a Series
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])

    data = data[tickers].dropna(how="all")
    return data


def download_fred(start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Download key macro series from FRED:
    - DGS2: 2Y yield
    - DGS10: 10Y yield
    - T10YIE: 10Y breakeven inflation
    - BAMLH0A0HYM2: HY OAS
    And derive:
    - YC_10Y2Y: 10Y - 2Y curve
    """
    fred_codes = ["DGS2", "DGS10", "T10YIE", "BAMLH0A0HYM2"]

    fred = pdr.DataReader(fred_codes, "fred", start, end)

    # Create curve
    fred["YC_10Y2Y"] = fred["DGS10"] - fred["DGS2"]

    # Some daily macro series may have NaNs / weekends; forward-fill
    fred = fred.ffill()

    return fred


# ------------------------ RATIO CONSTRUCTION -------------------------


def add_ratios(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a price DataFrame with all CONFIG tickers,
    return a new DataFrame with additional columns for the macro ratios.
    """

    df = price_df.copy()

    def safe_div(a, b):
        return np.where(b == 0, np.nan, a / b)

    # 1) Cyclicals vs Defensives: XLY/XLP
    df["XLY_XLP"] = safe_div(df["XLY"], df["XLP"])

    # 2) Value vs Growth: RPV/VONG
    df["RPV_VONG"] = safe_div(df["RPV"], df["VONG"])

    # 3) High Beta vs Low Vol: SPHB/SPLV
    df["SPHB_SPLV"] = safe_div(df["SPHB"], df["SPLV"])

    # 4) Small Caps vs Large Caps: IWM/QQQ
    df["IWM_QQQ"] = safe_div(df["IWM"], df["QQQ"])

    # 5) Dividend Aristocrats vs Market: NOBL/SPY
    df["NOBL_SPY"] = safe_div(df["NOBL"], df["SPY"])

    # 6) Biotech vs Market: XBI/SPY
    df["XBI_SPY"] = safe_div(df["XBI"], df["SPY"])

    # 7) Cyclical sectors vs Defensive sectors: (XLI + XLB) / (XLU + XLP)
    cyc = df["XLI"] + df["XLB"]
    defensives = df["XLU"] + df["XLP"]
    df["CYC_DEF_SECTORS"] = safe_div(cyc, defensives)

    # 8) Staffing basket vs SPY
    staffing_index = (df["MAN"] + df["RHI"] + df["KELYA"]) / 3.0
    df["STAFFING_SPY"] = safe_div(staffing_index, df["SPY"])

    # 9) US vs Rest-of-World: SPY/ACWX
    df["SPY_ACWX"] = safe_div(df["SPY"], df["ACWX"])

    # 10) EM vs DM (ex-US): EEM/VEA
    df["EEM_VEA"] = safe_div(df["EEM"], df["VEA"])

    # 11) Equal-weight vs cap-weight S&P: RSP/SPY
    df["RSP_SPY"] = safe_div(df["RSP"], df["SPY"])

    # 12) Copper vs Gold proxy: CPER/GLD
    df["CPER_GLD"] = safe_div(df["CPER"], df["GLD"])

    # 13) Financials vs Tech: XLF/XLK
    df["XLF_XLK"] = safe_div(df["XLF"], df["XLK"])

    return df


# ------------------------- ANALYTICS / ALERTS ------------------------


def add_moving_averages(
    df: pd.DataFrame,
    cols: List[str],
    ma_short: int = 50,
    ma_long: int = 200,
) -> pd.DataFrame:
    """
    Add short and long moving averages for each ratio column.
    """
    out = df.copy()
    for c in cols:
        out[f"{c}_MA{ma_short}"] = out[c].rolling(ma_short).mean()
        out[f"{c}_MA{ma_long}"] = out[c].rolling(ma_long).mean()
    return out


def latest_snapshot(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Returns a one-row DataFrame with the latest value and
    % distance to 200-day MA for each ratio.
    """
    last = df.iloc[-1].copy()
    snapshot_data = {}

    for c in cols:
        val = last[c]
        ma200 = last.get(f"{c}_MA200", np.nan)
        pct_above_ma = (val / ma200 - 1.0) * 100 if pd.notna(ma200) else np.nan
        snapshot_data[c] = {
            "Latest": val,
            "MA200": ma200,
            "% above MA200": pct_above_ma,
        }

    snap_df = pd.DataFrame(snapshot_data).T
    return snap_df.round(3)


def regime_message_for_ratio(
    name: str,
    latest: float,
    ma200: float,
    upside_threshold: float = 1.0,
    downside_threshold: float = -1.0,
) -> str:
    """
    Simple heuristic messages based on ratio relative to 200-day MA.
    upside_threshold / downside_threshold are % above/below MA200.
    """
    if np.isnan(latest) or np.isnan(ma200) or ma200 == 0:
        return f"{name}: not enough data for regime."

    pct = (latest / ma200 - 1.0) * 100

    # Direction vs MA
    if pct > upside_threshold:
        direction = "UP"
    elif pct < downside_threshold:
        direction = "DOWN"
    else:
        direction = "NEUTRAL"

    # --- Core ratios ---
    if name == "XLY_XLP":
        if direction == "UP":
            return "Cyclicals > Defensives: market leaning pro-growth / risk-on."
        elif direction == "DOWN":
            return "Defensives > Cyclicals: market bracing for slowdown or risk-off."
        else:
            return "Cyclicals vs Defensives: mixed / no strong growth signal yet."

    if name == "RPV_VONG":
        if direction == "UP":
            return "Value > Growth: market pricing higher rates/inflation or reflation."
        elif direction == "DOWN":
            return "Growth > Value: market favoring long-duration growth (lower rate / disinflation vibe)."
        else:
            return "Value vs Growth: balanced."

    if name == "SPHB_SPLV":
        if direction == "UP":
            return "High Beta > Low Vol: risk appetite strong, vol regime benign."
        elif direction == "DOWN":
            return "Low Vol > High Beta: defensive rotation, volatility or risk aversion rising."
        else:
            return "High Beta vs Low Vol: neutral / transitioning."

    if name == "IWM_QQQ":
        if direction == "UP":
            return "Small Caps > Large Growth: broad domestic risk-on, credit conditions supportive."
        elif direction == "DOWN":
            return "Large Growth > Small Caps: leadership narrow, sometimes late-cycle or risk-off."
        else:
            return "Small vs Large: no clear breadth signal."

    if name == "NOBL_SPY":
        if direction == "UP":
            return "Dividend Aristocrats > Market: quality/defensive tilt, possible risk-off or income focus."
        elif direction == "DOWN":
            return "Market > Aristocrats: investors chasing beta/growth over steady dividends."
        else:
            return "Quality vs Market: neutral."

    if name == "XBI_SPY":
        if direction == "UP":
            return "Biotech > Market: speculative risk appetite improving, funding/liquidity supportive."
        elif direction == "DOWN":
            return "Biotech < Market: market less willing to fund long-duration/speculative stories."
        else:
            return "Biotech vs Market: middle of the range."

    if name == "CYC_DEF_SECTORS":
        if direction == "UP":
            return "Cyclical sectors > Defensives: PMI / industrial cycle likely improving."
        elif direction == "DOWN":
            return "Defensive sectors > Cyclicals: industrial cycle under pressure."
        else:
            return "Cyclical vs Defensive sectors: mixed."

    if name == "STAFFING_SPY":
        if direction == "UP":
            return "Staffing > Market: labor demand healthy or recovering."
        elif direction == "DOWN":
            return "Staffing < Market: early warning of labor softening or recession risk."
        else:
            return "Staffing vs Market: neutral."

    # --- New global / cross-asset ratios ---

    if name == "SPY_ACWX":
        if direction == "UP":
            return "US > Rest-of-World: capital favoring US equities (growth/quality or strong USD)."
        elif direction == "DOWN":
            return "Rest-of-World > US: foreign markets leading, US losing relative momentum."
        else:
            return "US vs RoW: balanced leadership."

    if name == "EEM_VEA":
        if direction == "UP":
            return "EM > DM ex-US: risk appetite for EM, often weak USD / global reflation."
        elif direction == "DOWN":
            return "DM ex-US > EM: investors preferring safer developed markets over EM risk."
        else:
            return "EM vs DM: neutral."

    if name == "RSP_SPY":
        if direction == "UP":
            return "Equal-weight > Cap-weight: breadth healthy, gains broad across S&P constituents."
        elif direction == "DOWN":
            return "Cap-weight > Equal-weight: mega-caps dominating (narrow leadership / crowding)."
        else:
            return "Breadth vs mega-caps: neutral."

    if name == "CPER_GLD":
        if direction == "UP":
            return "Copper > Gold: pro-growth / reflation signal (industrial metals beating safe-haven gold)."
        elif direction == "DOWN":
            return "Gold > Copper: growth scare or risk-off, market preferring safety."
        else:
            return "Copper vs Gold: balanced."

    if name == "XLF_XLK":
        if direction == "UP":
            return "Financials > Tech: curve/rates backdrop favoring banks & value over long-duration growth."
        elif direction == "DOWN":
            return "Tech > Financials: market paying for secular growth, often lower rates or flattening curve."
        else:
            return "Financials vs Tech: neutral."

    # Fallback
    return f"{name}: direction {direction} ({pct:.2f}% vs MA200)."


def compute_macro_score(
    df: pd.DataFrame,
    ratio_cols: List[str],
    threshold_pct: float = 1.0,
) -> pd.DataFrame:
    """
    Build a composite risk-on score from all ratios.
    Idea:
        - For each ratio, compute % distance from MA200
        - If > +threshold_pct => +1
          If < -threshold_pct => -1
          Else 0
        - Flip sign for purely defensive ratio (NOBL_SPY)
        - Average across all ratios -> MACRO_SCORE in [-1, +1]
          (Positive = risk-on, negative = risk-off)
    """
    directions = {
        # +1 means higher ratio = more risk-on
        # -1 means higher ratio = more risk-off
        "XLY_XLP": 1,
        "RPV_VONG": 1,
        "SPHB_SPLV": 1,
        "IWM_QQQ": 1,
        "NOBL_SPY": -1,      # higher = more defensive
        "XBI_SPY": 1,
        "CYC_DEF_SECTORS": 1,
        "STAFFING_SPY": 1,
        "SPY_ACWX": 1,
        "EEM_VEA": 1,
        "RSP_SPY": 1,
        "CPER_GLD": 1,
        "XLF_XLK": 1,
    }

    scores = pd.Series(0.0, index=df.index)

    for name in ratio_cols:
        ma_col = f"{name}_MA200"
        if ma_col not in df.columns:
            continue

        z = (df[name] / df[ma_col] - 1.0) * 100  # % vs 200d MA
        raw = pd.Series(0.0, index=df.index)

        raw[z > threshold_pct] = 1.0
        raw[z < -threshold_pct] = -1.0

        direction = directions.get(name, 1)
        scores += raw * direction

    scores /= len(ratio_cols)
    out = df.copy()
    out["MACRO_SCORE"] = scores
    return out


def macro_score_text(score: float) -> str:
    """
    Turn the composite macro score into a short regime description.
    """
    if pd.isna(score):
        return "Macro composite score: n/a (insufficient data)."

    if score > 0.3:
        return (
            f"Macro composite score {score:.2f}: RISK-ON bias – "
            "multiple ratios aligned bullish (growth / cyclicals / beta)."
        )
    if score < -0.3:
        return (
            f"Macro composite score {score:.2f}: RISK-OFF bias – "
            "defensive / quality / safe-haven signals dominate."
        )

    return (
        f"Macro composite score {score:.2f}: MIXED / NEUTRAL regime – "
        "no strong consensus from the internal rotations."
    )


def print_regime_summary(snapshot_df: pd.DataFrame, macro_score: float) -> None:
    """
    Pretty-print summary of ratios and simple macro regime hints + macro score.
    """
    print("=" * 100)
    print("LATEST MACRO ROTATION SNAPSHOT")
    print("=" * 100)
    print(snapshot_df)
    print("\n--- Regime Interpretation (heuristic) ---")

    for name, row in snapshot_df.iterrows():
        msg = regime_message_for_ratio(
            name,
            latest=row["Latest"],
            ma200=row["MA200"],
            upside_threshold=CONFIG["regime_threshold_pct"],
            downside_threshold=-CONFIG["regime_threshold_pct"],
        )
        print("\n" + textwrap.fill(msg, width=100))

    print("\n" + "=" * 100)
    print("COMPOSITE MACRO RISK-ON SCORE")
    print("=" * 100)
    print(textwrap.fill(macro_score_text(macro_score), width=100))


# --------------------------- PLOTTING --------------------------------


def plot_ratios(df: pd.DataFrame, ratio_cols: List[str]) -> None:
    """
    Plot all ratio columns on a grid of subplots with 200-day MA.
    """
    n = len(ratio_cols)
    cols = 2
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for ax, ratio in zip(axes, ratio_cols):
        series = df[ratio]
        ma200 = df.get(f"{ratio}_MA200")

        ax.plot(series.index, series.values, label=ratio)
        if ma200 is not None:
            ax.plot(ma200.index, ma200.values, linestyle="--", label="MA200")

        ax.set_title(ratio)
        ax.grid(True, linestyle=":")
        ax.legend(fontsize=8)

    # Hide any unused axes
    for j in range(len(ratio_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Global Macro Equity Rotations (Extended)", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_fred(fred_df: pd.DataFrame) -> None:
    """
    Plot FRED macro series:
    - 2Y & 10Y yields
    - 10Y breakeven (T10YIE)
    - HY OAS (BAMLH0A0HYM2)
    - 10Y-2Y curve
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    axes = axes.flatten()

    # Panel 1: Yields
    axes[0].plot(fred_df.index, fred_df["DGS2"], label="2Y")
    axes[0].plot(fred_df.index, fred_df["DGS10"], label="10Y")
    axes[0].set_title("US Treasury Yields (2Y & 10Y)")
    axes[0].grid(True, linestyle=":")
    axes[0].legend(fontsize=8)

    # Panel 2: 10Y Breakeven
    axes[1].plot(fred_df.index, fred_df["T10YIE"], label="10Y Breakeven")
    axes[1].set_title("10Y Breakeven Inflation (T10YIE)")
    axes[1].grid(True, linestyle=":")
    axes[1].legend(fontsize=8)

    # Panel 3: HY OAS
    axes[2].plot(fred_df.index, fred_df["BAMLH0A0HYM2"], label="HY OAS")
    axes[2].set_title("High Yield Option-Adjusted Spread")
    axes[2].grid(True, linestyle=":")
    axes[2].legend(fontsize=8)

    # Panel 4: Yield Curve (10Y - 2Y)
    axes[3].plot(fred_df.index, fred_df["YC_10Y2Y"], label="10Y - 2Y")
    axes[3].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[3].set_title("Yield Curve (10Y - 2Y)")
    axes[3].grid(True, linestyle=":")
    axes[3].legend(fontsize=8)

    fig.suptitle("FRED Macro Indicators", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_macro_score(df: pd.DataFrame) -> None:
    """
    Plot the composite macro risk-on score over time.
    """
    if "MACRO_SCORE" not in df.columns:
        print("No MACRO_SCORE column found to plot.")
        return

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df["MACRO_SCORE"], label="MACRO_SCORE")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.axhline(0.3, linestyle=":", linewidth=1)
    plt.axhline(-0.3, linestyle=":", linewidth=1)
    plt.title("Composite Macro Risk-On Score")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()


# --------------------------- MAIN ------------------------------------


def main():
    print("Downloading price data from Yahoo Finance...")
    prices = download_prices(
        CONFIG["tickers"],
        CONFIG["start_date"],
        CONFIG["end_date"],
    )

    print(f"Data range: {prices.index.min().date()} -> {prices.index.max().date()}")
    print("Building equity/ETF ratios...")
    ratios_df = add_ratios(prices)

    ratio_cols = [
        "XLY_XLP",
        "RPV_VONG",
        "SPHB_SPLV",
        "IWM_QQQ",
        "NOBL_SPY",
        "XBI_SPY",
        "CYC_DEF_SECTORS",
        "STAFFING_SPY",
        "SPY_ACWX",
        "EEM_VEA",
        "RSP_SPY",
        "CPER_GLD",
        "XLF_XLK",
    ]

    print("Computing moving averages...")
    ratios_df = add_moving_averages(
        ratios_df,
        cols=ratio_cols,
        ma_short=CONFIG["ma_short"],
        ma_long=CONFIG["ma_long"],
    )

    # Drop initial NaNs from MAs
    ratios_df = ratios_df.dropna(
        subset=[f"{c}_MA{CONFIG['ma_long']}" for c in ratio_cols]
    )

    # Composite macro score
    print("Computing composite macro risk-on score...")
    ratios_df = compute_macro_score(
        ratios_df,
        ratio_cols=ratio_cols,
        threshold_pct=CONFIG["regime_threshold_pct"],
    )

    snap = latest_snapshot(ratios_df, ratio_cols)
    macro_score = ratios_df["MACRO_SCORE"].iloc[-1]
    print_regime_summary(snap, macro_score)

    print("\nGenerating ratio plots...")
    plot_ratios(ratios_df, ratio_cols)

    # FRED macro data
    try:
        print("\nDownloading FRED macro series...")
        fred = download_fred(
            ratios_df.index.min().date(),
            ratios_df.index.max().date(),
        )
        # Align FRED series to ratio dates
        fred = fred.reindex(ratios_df.index).ffill()
        print(f"Fetched FRED series: {list(fred.columns)}")
        plot_fred(fred)
    except Exception as e:
        print("\nWarning: could not download FRED series.")
        print("Install pandas_datareader and check internet access if needed.")
        print("Error:", e)

    print("\nPlotting composite macro score...")
    plot_macro_score(ratios_df)

    print("\nDone. Tweak CONFIG or extend add_ratios/ratio_cols / macro score logic as you like.")


if __name__ == "__main__":
    main()