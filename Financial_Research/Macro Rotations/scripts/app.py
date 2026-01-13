import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import sys
import os

# Add 'Macro Rotations' folder to path so we can import the script
sys.path.append(os.path.join(os.path.dirname(__file__), "Macro Rotations"))

import macro_rotations as mr # Import the core logic
import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Macro Rotations Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR & SETTINGS ---
st.sidebar.title("⚙️ Settings")

# Date Range
start_date = st.sidebar.date_input("Start Date", datetime.date(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Update CONFIG in macro_rotations based on sidebar
mr.CONFIG["start_date"] = str(start_date)
mr.CONFIG["end_date"] = str(end_date)

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Export")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    with st.spinner("Fetching Data..."):
        # 1. Fetch Prices & Macro
        prices, macro = mr.download_data(mr.CONFIG)
        
        # 2. Build Analytics
        df = mr.build_analytics(prices, macro, mr.CONFIG)
        
        # 3. Add Trends & Score
        df = mr.add_trends_and_score(df, mr.CONFIG)
        
        return df, prices

# Initialize variables to avoid NameError if loading fails
df = None
prices = None

try:
    df, prices = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()
    sys.exit(1)

if df is None:
    st.warning("Data could not be loaded. Please check the logs.")
    st.stop()
    sys.exit(1)

# --- MAIN DASHBOARD ---
st.title("🌍 Global Macro Equity Rotations Dashboard")
st.markdown(f"**Date**: {datetime.date.today()} | **Regime**: {mr.CONFIG['start_date']} to {mr.CONFIG['end_date']}")

# Generate Commentary
report_text = mr.generate_capital_flows_commentary(df, prices)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Executive Summary", 
    "🌊 Capital Flows", 
    "🔄 Rotations", 
    "🧪 Backtest", 
    "📊 Raw Data"
])

# --- TAB 1: EXECUTIVE SUMMARY ---
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Actionable Playbook")
        st.text_area("Report", report_text, height=600, disabled=True)
        
    with col2:
        st.subheader("Key Metrics")
        
        # Macro Score Gauge
        curr_score = df["MACRO_SCORE"].iloc[-1]
        st.metric("Composite Macro Score", f"{curr_score:.2f}", 
                  delta="Risk On" if curr_score > 0 else "Risk Off",
                  delta_color="normal" if curr_score > 0 else "inverse")
        
        # VIX
        vix = df["CF_VIX"].iloc[-1]
        st.metric("VIX (Fear Index)", f"{vix:.2f}", delta=f"{vix - df['CF_VIX'].iloc[-2]:.2f}", delta_color="inverse")
        
        # Liquidity Valve
        valve = df["CF_Liquidity_Valve"].iloc[-1]
        valve_ma = df["CF_Liquidity_Valve_MA200"].iloc[-1]
        st.metric("Liquidity Valve (BTC)", f"{valve:.2f}", 
                  delta="OPEN" if valve > valve_ma else "CLOSED",
                  delta_color="normal" if valve > valve_ma else "inverse")

        # PDF Download
        if st.button("📄 Generate PDF Report"):
            with st.spinner("Generating PDF..."):
                mr.plot_super_dashboard(df, prices, report_text)
                with open("Macro_Dashboard_Report.pdf", "rb") as pdf_file:
                    st.download_button(
                        label="Download PDF",
                        data=pdf_file,
                        file_name="Macro_Dashboard_Report.pdf",
                        mime="application/pdf"
                    )

# --- TAB 2: CAPITAL FLOWS ---
with tab2:
    st.subheader("The 4 Pillars of Liquidity")
    
    col_a, col_b = st.columns(2)
    
    # Helper for Plotly Charts
    def plot_metric(data, col_name, ma_col, title, color_line, hlines=None):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data[col_name], mode='lines', name=title, line=dict(color=color_line)))
        if ma_col in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[ma_col], mode='lines', name="200D MA", line=dict(color='gray', dash='dash')))
        
        if hlines:
            for y, color, label in hlines:
                fig.add_hline(y=y, line_dash="dot", annotation_text=label, line_color=color)
                
        fig.update_layout(title=title, height=400, hovermode="x unified")
        return fig

    with col_a:
        # 1. Liquidity Valve
        st.plotly_chart(plot_metric(df, "CF_Liquidity_Valve", "CF_Liquidity_Valve_MA200", "1. Liquidity Valve (BTC/Gold)", "orange"), use_container_width=True)
        # 3. Breadth
        st.plotly_chart(plot_metric(df, "CF_Breadth", "CF_Breadth_MA200", "3. Market Breadth (RSP/SPY)", "blue"), use_container_width=True)
        
    with col_b:
        # 2. Real Yields
        st.plotly_chart(plot_metric(df, "CF_Real_Yield", "None", "2. Real Yields (10Y - Inflation)", "purple", 
                                    [(2.0, "red", "Restrictive"), (0.0, "green", "Accommodative")]), use_container_width=True)
        # 4. NFCI
        st.plotly_chart(plot_metric(df, "CF_NFCI", "None", "4. Financial Conditions (NFCI)", "gray", [(0.0, "black", "Avg")]), use_container_width=True)

# --- TAB 3: ROTATIONS ---
with tab3:
    st.subheader("Equity Rotations: Under the Hood")
    
    rotations = [
        ("ROT_Value_Growth", "Value vs Growth (RPV/VONG)"),
        ("ROT_Cyc_Def_Sectors", "Cyclicals vs Defensives"),
        ("ROT_Small_Large", "Small vs Large Cap (IWM/QQQ)"),
        ("ROT_HiBeta_LoVol", "High Beta vs Low Vol (SPHB/SPLV)")
    ]
    
    col_r1, col_r2 = st.columns(2)
    
    for i, (col, title) in enumerate(rotations):
        target_col = col_r1 if i % 2 == 0 else col_r2
        with target_col:
            st.plotly_chart(plot_metric(df, col, f"{col}_MA200", title, "black"), use_container_width=True)
            
    st.markdown("---")
    st.subheader("Cross-Asset Correlation Matrix (6-Month)")
    
    # Heatmap
    corr_cols = [c for c in df.columns if ("ROT_" in c or "CF_" in c) and "_Z" not in c and "_MA200" not in c]
    recent_df = df[corr_cols].iloc[-126:]
    corr_matrix = recent_df.corr()
    
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig_corr.update_layout(height=800)
    st.plotly_chart(fig_corr, use_container_width=True)

# --- TAB 4: BACKTEST ---
with tab4:
    st.subheader("Strategy Backtest")
    st.markdown("Comparing **Macro Regime** and **Liquidity Valve** strategies against SPY Buy & Hold.")
    
    if st.button("🧪 Run Backtest"):
        with st.spinner("Running Backtest..."):
            # Re-using the matplotlib figure from the script for now
            # Ideally we would refactor run_backtest to return data for Plotly
            fig_bt = mr.run_backtest(df, prices)
            if fig_bt:
                st.pyplot(fig_bt)
            else:
                st.error("Backtest failed. Check data availability.")

# --- TAB 5: RAW DATA ---
with tab5:
    st.subheader("Raw Dataframe")
    st.dataframe(df)

# --- TAB 6: QUANT LAB ---
import quant_engine as qe

with st.sidebar:
    st.markdown("### 🧮 Quant Lab Settings")
    quant_asset = st.selectbox("Select Asset for Analysis", prices.columns, index=0)
    quant_bench = st.selectbox("Select Benchmark", prices.columns, index=list(prices.columns).index("SPY") if "SPY" in prices.columns else 0)

tab_q1, tab_q2, tab_q3, tab_q4 = st.tabs(["Regression", "Options Lab", "Volatility", "Monte Carlo"])

# 1. REGRESSION
with tab_q1:
    st.subheader(f"Rolling Regression: {quant_asset} vs {quant_bench}")
    
    if st.button("Run Regression"):
        with st.spinner("Calculating Rolling Beta/Alpha..."):
            y = prices[quant_asset].pct_change().dropna()
            x = prices[quant_bench].pct_change().dropna()
            
            reg_df = qe.rolling_regression(y, x, window=60)
            
            fig_beta = px.line(reg_df, y="Beta", title="60-Day Rolling Beta")
            fig_beta.add_hline(y=1.0, line_dash="dot", line_color="gray")
            st.plotly_chart(fig_beta, use_container_width=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(px.line(reg_df, y="Alpha", title="Annualized Alpha"), use_container_width=True)
            with col_b:
                st.plotly_chart(px.line(reg_df, y="R2", title="R-Squared (Correlation Strength)"), use_container_width=True)

# 2. OPTIONS LAB
with tab_q2:
    st.subheader("Black-Scholes-Merton Calculator")
    
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        S = st.number_input("Spot Price ($)", value=float(prices[quant_asset].iloc[-1]), step=1.0)
        K = st.number_input("Strike Price ($)", value=float(prices[quant_asset].iloc[-1]), step=1.0)
        T_days = st.slider("Days to Expiration", 1, 365, 30)
        T = T_days / 365.0
    with col_opt2:
        sigma = st.slider("Implied Volatility (%)", 10.0, 200.0, 20.0) / 100.0
        r = st.number_input("Risk-Free Rate (%)", value=4.5) / 100.0
        opt_type = st.radio("Option Type", ["Call", "Put"])
        
    price = qe.black_scholes_merton(S, K, T, r, sigma, opt_type.lower())
    greeks = qe.calculate_greeks(S, K, T, r, sigma, opt_type.lower())
    
    st.metric(f"{opt_type} Price", f"${price:.2f}")
    
    # Visualize Greeks
    g_df = pd.DataFrame(greeks, index=["Value"]).T
    st.bar_chart(g_df)
    
    # 3D Surface Plot (Spot vs Time)
    st.markdown("#### Option Price Surface")
    s_range = np.linspace(S*0.5, S*1.5, 20)
    t_range = np.linspace(0.01, 1.0, 20)
    S_mesh, T_mesh = np.meshgrid(s_range, t_range)
    P_mesh = np.zeros_like(S_mesh)
    
    for i in range(20):
        for j in range(20):
            P_mesh[i,j] = qe.black_scholes_merton(S_mesh[i,j], K, T_mesh[i,j], r, sigma, opt_type.lower())
            
    fig_surf = go.Figure(data=[go.Surface(z=P_mesh, x=S_mesh, y=T_mesh)])
    fig_surf.update_layout(title="Price vs Spot & Time", scene=dict(xaxis_title="Spot", yaxis_title="Time", zaxis_title="Price"))
    st.plotly_chart(fig_surf, use_container_width=True)

# 3. VOLATILITY
with tab_q3:
    st.subheader("Volatility Modeling (GARCH vs Historical)")
    
    if st.button("Fit GARCH(1,1) Model"):
        with st.spinner("Fitting Model..."):
            ret = prices[quant_asset].pct_change().dropna().values
            # Scale returns for numerical stability
            ret_scaled = ret * 100 
            
            vol_scaled, params = qe.fit_garch(ret_scaled)
            vol_est = vol_scaled / 100 * np.sqrt(252) # Annualize
            
            # Historical Vol
            hist_vol = prices[quant_asset].pct_change().rolling(21).std() * np.sqrt(252)
            
            vol_df = pd.DataFrame({
                "GARCH(1,1) Est": vol_est,
                "Historical (21D)": hist_vol.values[-len(vol_est):]
            }, index=prices.index[-len(vol_est):])
            
            st.plotly_chart(px.line(vol_df, title=f"{quant_asset} Volatility Regime"), use_container_width=True)
            st.write(f"Estimated Parameters: Omega={params[0]:.4f}, Alpha={params[1]:.4f}, Beta={params[2]:.4f}")

# 4. MONTE CARLO
with tab_q4:
    st.subheader("Geometric Brownian Motion Simulation")
    
    n_paths = st.slider("Number of Paths", 10, 500, 50)
    sim_days = st.slider("Simulation Horizon (Days)", 30, 252, 90)
    
    if st.button("Run Simulation"):
        S0 = prices[quant_asset].iloc[-1]
        # Estimate drift and vol from last year
        ret = prices[quant_asset].pct_change().iloc[-252:]
        mu = ret.mean() * 252
        sigma_sim = ret.std() * np.sqrt(252)
        
        T_sim = sim_days / 252.0
        dt = 1/252.0
        
        time, paths = qe.simulate_gbm(S0, mu, sigma_sim, T_sim, dt, n_paths)
        
        # Plot
        fig_mc = go.Figure()
        for i in range(n_paths):
            fig_mc.add_trace(go.Scatter(x=time*252, y=paths[i], mode='lines', line=dict(width=1, color='rgba(0,255,255,0.3)'), showlegend=False))
            
        # Add Mean Path
        fig_mc.add_trace(go.Scatter(x=time*252, y=paths.mean(axis=0), mode='lines', line=dict(width=3, color='white'), name="Mean Path"))
        fig_mc.update_layout(title=f"Monte Carlo Simulation ({n_paths} Paths)", xaxis_title="Days", yaxis_title="Price")
        st.plotly_chart(fig_mc, use_container_width=True)
