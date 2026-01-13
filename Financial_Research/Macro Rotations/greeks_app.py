import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from datetime import datetime, timedelta
import time
import os
import sys

# ==========================================
# CONFIG & PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="Greeks Search Engine",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Premium" feel
st.markdown("""
<style>
    .metric-card {
        background-color: #0E1117;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #30333F;
        text-align: center;
    }
    .stPlotlyChart {
        background-color: #0E1117;
        border-radius: 10px;
        border: 1px solid #30333F;
        padding: 10px;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# QUANT ENGINE (BSM & Greeks)
# ==========================================
def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Calculates Option Price and Greeks using Black-Scholes-Merton.
    """
    try:
        # Handle expiration
        if T <= 1e-5:
            return {
                "Price": max(0, S - K) if option_type == "call" else max(0, K - S),
                "Delta": 0, "Gamma": 0, "Vega": 0, "Theta": 0, "Rho": 0
            }

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        N_prime = norm.pdf(d1)
        
        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            theta = (- (S * sigma * N_prime) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1
            theta = (- (S * sigma * N_prime) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
        gamma = N_prime / (S * sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * N_prime / 100
        
        return {
            "Price": price,
            "Delta": delta,
            "Gamma": gamma,
            "Vega": vega,
            "Theta": theta,
            "Rho": rho
        }
    except Exception:
        return {"Price": 0, "Delta": 0, "Gamma": 0, "Vega": 0, "Theta": 0, "Rho": 0}

@st.cache_data(ttl=900) # Cache for 15 mins to avoid spamming Yahoo
def get_stock_data(ticker_symbol):
    """Fetches stock info and history."""
    ticker = yf.Ticker(ticker_symbol)
    
    # Check if valid
    try:
        hist = ticker.history(period="1y")
        if hist.empty: return None
        
        info = ticker.info
        current_price = hist["Close"].iloc[-1]
        prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else current_price
        
        return {
            "current_price": current_price,
            "change": current_price - prev_close,
            "pct_change": (current_price - prev_close) / prev_close,
            "name": info.get("shortName", ticker_symbol),
            "sector": info.get("sector", "N/A"),
            "history": hist
        }
    except Exception as e:
        return None

@st.cache_data(ttl=300)
def get_full_option_chain(ticker_symbol, current_price, r=0.045):
    """Fetches and processes the full option chain."""
    try:
        _ticker_obj = yf.Ticker(ticker_symbol)
        exps = _ticker_obj.options
        if not exps: return pd.DataFrame()
        
        all_opts = []
        
        # Limit to first 6 expirations for performance/relevance
        target_exps = exps[:6]
        
        for date_str in target_exps:
            try:
                opt = _ticker_obj.option_chain(date_str)
                calls = opt.calls.copy()
                puts = opt.puts.copy()
                
                # Calculate T (Time to expiry in years)
                exp_date = datetime.strptime(date_str, "%Y-%m-%d")
                today = datetime.now()
                T = (exp_date - today).days / 365.25
                if T < 0.001: T = 0.001 # Avoid div by zero
                
                # Process Call Options
                calls['type'] = 'call'
                calls['T'] = T
                calls['expiration'] = date_str
                
                # Process Put Options
                puts['type'] = 'put'
                puts['T'] = T
                puts['expiration'] = date_str
                
                all_opts.append(calls)
                all_opts.append(puts)
                
            except Exception:
                continue
                
        if not all_opts: return pd.DataFrame()
        
        df = pd.concat(all_opts, ignore_index=True)
        
        # Calculate Greeks
        # Note: Yahoo provides 'impliedVolatility', we use that.
        # If IV is 0 or missing, Greeks will be garbage, but we filter later.
        
        greeks_list = []
        for idx, row in df.iterrows():
            sigma = row.get('impliedVolatility', 0)
            if sigma is None or np.isnan(sigma) or sigma == 0:
                sigma = 0.001 # Fallback
            
            g = calculate_greeks(current_price, row['strike'], row['T'], r, sigma, row['type'])
            greeks_list.append(g)
            
        greeks_df = pd.DataFrame(greeks_list)
        df = pd.concat([df.reset_index(drop=True), greeks_df], axis=1)
        
        # Filter junk
        df = df[df['impliedVolatility'] > 0.01]
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching options: {e}")
        return pd.DataFrame()

# ==========================================
# SIDEBAR & INPUTS
# ==========================================
with st.sidebar:
    st.header("Search Parameters")
    ticker_input = st.text_input("Ticker Symbol", value="NVDA").upper()
    risk_free_rate = st.number_input("Risk Free Rate (%)", value=4.5, step=0.1) / 100
    
    st.markdown("---")
    st.info("""
    **Guide:**
    - **Current Market**: View visual analysis of active options (Vol Surface, Skew, GEX).
    - **Chain Explorer**: Deep dive into specific contracts.
    - **Historical Simulator**: Estimate how Greeks *would have* evolved using past price/volatility.
    """)

# ==========================================
# MAIN APP LOGIC
# ==========================================

# 1. Fetch Data
stock_data = get_stock_data(ticker_input)

if stock_data:
    # --- Header ---
    col1, col2, col3, col4 = st.columns([2, 1, 1, 3])
    with col1:
        st.title(f"{stock_data['name']} ({ticker_input})")
        st.caption(f"Sector: {stock_data['sector']}")
    with col2:
        st.metric("Price", f"${stock_data['current_price']:.2f}", 
                 f"{stock_data['change']:.2f} ({stock_data['pct_change']:.2%})")
        
    st.markdown("---")
    
    # --- Fetch Options ---
    with st.spinner("Analyzing Option Chain..."):
        df_opt = get_full_option_chain(ticker_input, stock_data['current_price'], risk_free_rate)
    
    if not df_opt.empty:
        # TABS
        tab1, tab2, tab3 = st.tabs(["📊 Current Market Analysis", "🔢 Option Chain Explorer", "⏳ Historical Simulator"])
        
        # ==========================================
        # TAB 1: CURRENT MARKET (Visuals)
        # ==========================================
        with tab1:
            col_a, col_b = st.columns(2)
            
            # --- 1. Implied Volatility Surface / Smile ---
            with col_a:
                st.subheader("Volatility Smile")
                # Filter for "Near the Money" and relevant expiries
                near_money_range = 0.20 # +/- 20%
                strike_min = stock_data['current_price'] * (1 - near_money_range)
                strike_max = stock_data['current_price'] * (1 + near_money_range)
                
                df_smile = df_opt[(df_opt['strike'] >= strike_min) & 
                                  (df_opt['strike'] <= strike_max) & 
                                  (df_opt['volume'] > 10)].copy() # Filter for liquid options
                
                if not df_smile.empty:
                    fig_smile = px.line(df_smile, x='strike', y='impliedVolatility', color='expiration', 
                                        line_shape='spline', symbol='type', 
                                        title=f"IV Smile (Strikes +/- 20%)")
                    fig_smile.add_vline(x=stock_data['current_price'], line_dash="dash", line_color="white", annotation_text="Spot")
                    st.plotly_chart(fig_smile, use_container_width=True)
                else:
                    st.warning("Not enough liquid data for Smile Curve.")

            # --- 2. Net Gamma Exposure (GEX) ---
            with col_b:
                st.subheader("Net Gamma Exposure (GEX)")
                # GEX ~ Gamma * OpenInterest * NetPositions from Dealer perspective
                # Assumption: Dealers are Long Calls, Short Puts? Or Short both? 
                # Standard Approx: GEX = Gamma * OI * Spot * 100 * (1 for Call, -1 for Put)
                # This assumes dealers sold the calls (so they are short gamma) and sold the puts (long gamma? No.)
                # Correct Dealer Assumption typically: Dealers are Short Calls (Short Gamma) and Short Puts (Long Gamma to hedge?) 
                # Let's use the SpotGamma/SqueezeMetrics convention:
                # Call GEX = Gamma * OI (Positive contribution to stability/resistance? No, Dealer Short Call -> Short Gamma. Market moves up, dealer buys. Accentuates move.)
                
                # Standard Convention:
                # Dealers short calls -> Short Gamma (Negative GEX) creates volatility.
                # Dealers short puts -> Long Gamma (Positive GEX) dampens volatility.
                # Wait, standard GEX logic:
                # Dealer Long Call: Gamma > 0. Dealer Short Call: Gamma < 0.
                # Usually we assume customers BUY options. So Dealers are SHORT options.
                # Dealer Short Call: Gamma < 0. (Accels move)
                # Dealer Short Put: Gamma < 0. (Accels move)
                # Wait, let's use the "Net GEX" widely used:
                # GEX = Gamma * OI * Spot * (1 if Call else -1)
                # Positive GEX: Dealers Long Gamma (Suppress Vol). Negative GEX: Dealers Short Gamma (Amplify Vol).
                
                df_opt['GEX'] = df_opt['Gamma'] * df_opt['openInterest'] * stock_data['current_price'] * 100
                # Call contribution is positive, Put contribution is negative (Standard Model)
                df_opt.loc[df_opt['type'] == 'put', 'GEX'] = df_opt.loc[df_opt['type'] == 'put', 'GEX'] * -1
                
                gex_by_strike = df_opt.groupby('strike')['GEX'].sum().reset_index()
                gex_by_strike = gex_by_strike[(gex_by_strike['strike'] >= strike_min) & 
                                              (gex_by_strike['strike'] <= strike_max)]
                
                fig_gex = px.bar(gex_by_strike, x='strike', y='GEX', 
                                 title="Total Net GEX by Strike ($Billions approx)",
                                 color='GEX', color_continuous_scale="RdBu")
                fig_gex.add_vline(x=stock_data['current_price'], line_dash="dash", line_color="white")
                st.plotly_chart(fig_gex, use_container_width=True)
                
            # --- 3. 3D Volume/OI Surface ---
            st.subheader("Open Interest Heatmap")
            # Pivot for Heatmap
            try:
                # Focus on first expiration for clarity (or aggregate)
                # Let's do a heatmap of Strike vs Expiration for Total OI
                heatmap_data = df_opt.groupby(['strike', 'expiration'])['openInterest'].sum().reset_index()
                # Filter range again
                heatmap_data = heatmap_data[(heatmap_data['strike'] >= strike_min) & 
                                            (heatmap_data['strike'] <= strike_max)]
                
                fig_heat = px.density_heatmap(heatmap_data, x="strike", y="expiration", z="openInterest",
                                              nbinsx=30, nbinsy=len(target_exps),
                                              title="Open Interest Concentration",
                                              color_continuous_scale="Viridis")
                fig_heat.add_vline(x=stock_data['current_price'], line_dash="dash", line_color="white")
                st.plotly_chart(fig_heat, use_container_width=True)
            except:
                st.write("Insufficient data for heatmap.")

        # ==========================================
        # TAB 2: OPTION CHAIN EXPLORER
        # ==========================================
        with tab2:
            st.subheader("Filtered Option Chain")
            
            # Filters
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                selected_exp = st.selectbox("Select Expiration", options=df_opt['expiration'].unique())
            with col_f2:
                selected_type = st.radio("Option Type", ["call", "put"], horizontal=True)
                
            display_df = df_opt[(df_opt['expiration'] == selected_exp) & 
                                (df_opt['type'] == selected_type)].copy()
            
            # Formatting
            display_cols = ['strike', 'lastPrice', 'impliedVolatility', 'openInterest', 'volume', 
                            'Delta', 'Gamma', 'Vega', 'Theta']
            
            st.dataframe(display_df[display_cols].style.format({
                "lastPrice": "${:.2f}",
                "impliedVolatility": "{:.1%}",
                "Delta": "{:.3f}",
                "Gamma": "{:.4f}",
                "Vega": "{:.3f}",
                "Theta": "{:.3f}"
            }), use_container_width=True)
            
        # ==========================================
        # TAB 3: HISTORICAL SIMULATOR
        # ==========================================
        with tab3:
            st.subheader("Historical Greek Simulator")
            st.markdown("""
            **What if?**
            Since historical option data is not available, we **simulate** how a specific option contract structure (e.g., "30-Day ATM Call") 
            would have behaved over the last year given the stock's actual price and realized volatility history.
            """)
            
            col_sim1, col_sim2, col_sim3 = st.columns(3)
            with col_sim1:
                sim_days = st.slider("Days to Expiration (Fixed)", 1, 90, 30)
            with col_sim2:
                sim_moneyness = st.select_slider("Moneyness", options=[0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2], value=1.0, 
                                                 format_func=lambda x: f"{x:.0%} (OTM)" if x > 1 else (f"{x:.0%} (ITM)" if x < 1 else "ATM"))
            with col_sim3:
                sim_type = st.radio("Sim Option Type", ["call", "put"], horizontal=True)
                
            if st.button("Run Simulation"):
                hist_df = stock_data['history'].copy()
                
                # Calculate Realized Volatility (20-day rolling window)
                hist_df['log_ret'] = np.log(hist_df['Close'] / hist_df['Close'].shift(1))
                hist_df['realized_vol'] = hist_df['log_ret'].rolling(window=20).std() * np.sqrt(252)
                hist_df = hist_df.dropna()
                
                # Simulation Loop
                sim_results = []
                
                for date, row in hist_df.iterrows():
                    S_t = row['Close']
                    K_t = S_t * sim_moneyness # Floating Strike (e.g., always analyzing the ATM option of that day)
                    # Alternatively, fixed strike? No, "Structural" analysis usually means "How does an ATM option behave?"
                    # If we used fixed strike, it would go deep ITM/OTM and Greeks would vanish.
                    # Let's stick to "Rolling Fixed Moneyness" (e.g. Always analyzing the 100% Strike)
                    
                    sigma_t = row['realized_vol']
                    # Add a "Vol Premium" spread? Realized is usually lower than Implied.
                    # Let's add a flat 20% premium for simulation realism
                    sigma_t = sigma_t * 1.2 
                    
                    greeks = calculate_greeks(S_t, K_t, sim_days/365, risk_free_rate, sigma_t, sim_type)
                    greeks['Date'] = date
                    greeks['Close'] = S_t
                    greeks['Vol'] = sigma_t
                    sim_results.append(greeks)
                    
                sim_df = pd.DataFrame(sim_results).set_index("Date")
                
                # Visualize
                
                # 1. Price vs Delta
                fig_sim_delta = px.line(sim_df, x=sim_df.index, y="Delta", title="Simulated Delta History (Rolling ATM)", color_discrete_sequence=["cyan"])
                fig_sim_delta.add_trace(go.Scatter(x=sim_df.index, y=sim_df["Close"], name="Stock Price", yaxis="y2", line=dict(color="gray", dash="dot")))
                fig_sim_delta.update_layout(yaxis2=dict(overlaying="y", side="right", title="Stock Price"))
                st.plotly_chart(fig_sim_delta, use_container_width=True)
                
                # 2. Gamma vs Vol
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    fig_sim_gamma = px.line(sim_df, x=sim_df.index, y="Gamma", title="Simulated Gamma Exposure", color_discrete_sequence=["magenta"])
                    st.plotly_chart(fig_sim_gamma, use_container_width=True)
                with col_g2:
                    fig_sim_vol = px.line(sim_df, x=sim_df.index, y="Vol", title="Realized Volatility (Input)", color_discrete_sequence=["orange"])
                    st.plotly_chart(fig_sim_vol, use_container_width=True)
                    
                # 3. Theta Decay
                fig_sim_theta = px.line(sim_df, x=sim_df.index, y="Theta", title="Simulated Theta (Time Decay)", color_discrete_sequence=["red"])
                st.plotly_chart(fig_sim_theta, use_container_width=True)
                
    else:
        st.warning(f"No options data found for {ticker_input}. It might be an index or ETF with delayed data, or simply illiquid.")

else:
    st.error(f"Could not load data for {ticker_input}. Please check the ticker symbol.")

if __name__ == "__main__":
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())
