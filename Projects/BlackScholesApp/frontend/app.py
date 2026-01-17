import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Configuration
API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Black-Scholes Calculator", layout="wide")

st.title("Black-Scholes Option Calculator")

# Sidebar - Ticker & Core Params
st.sidebar.header("Market Data")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()

if ticker:
    try:
        # Fetch Market Data
        market_data_res = requests.get(f"{API_BASE_URL}/market-data/{ticker}")
        if market_data_res.status_code == 200:
            market_data = market_data_res.json()
            current_price = market_data["current_price"]
            r_rate = market_data["risk_free_rate"]
            hist_vol = market_data["historical_volatility"]
            expirations = market_data["expirations"]

            st.sidebar.metric("Current Price", f"${current_price:.2f}")
            st.sidebar.metric("Risk-Free Rate", f"{r_rate:.2%}")
            st.sidebar.metric("Hist. Volatility", f"{hist_vol:.2%}")

            # Option Selection
            st.sidebar.subheader("Option Selection")
            selected_expiry = st.sidebar.selectbox("Expiration Date", expirations)
            
            option_type = st.sidebar.radio("Option Type", ["Call", "Put"])

            if selected_expiry:
                # Fetch Option Chain
                chain_res = requests.get(f"{API_BASE_URL}/option-chain/{ticker}", params={"date": selected_expiry})
                if chain_res.status_code == 200:
                    chain_data = chain_res.json()
                    options_list = chain_data["calls"] if option_type == "Call" else chain_data["puts"]
                    
                    if not options_list:
                        st.warning("No options found for this expiration.")
                    else:
                        df_chain = pd.DataFrame(options_list)
                        strikes = sorted(df_chain["strike"].unique())
                        
                        # Find closest strike to current price
                        closest_strike = min(strikes, key=lambda x: abs(x - current_price))
                        idx = strikes.index(closest_strike)
                        
                        selected_strike = st.sidebar.selectbox("Strike Price", strikes, index=idx)
                        
                        # Find specific option data
                        selected_option = df_chain[df_chain["strike"] == selected_strike].iloc[0]
                        market_price = selected_option.get("lastPrice", 0)
                        bid = selected_option.get("bid", 0)
                        ask = selected_option.get("ask", 0)
                        implied_vol = selected_option.get("impliedVolatility", 0)
                        
                        st.sidebar.markdown(f"**Market Price:** ${market_price} (Bid/Ask: {bid}/{ask})")
                        st.sidebar.markdown(f"**Implied Vol:** {implied_vol:.2%}")

                        # Inputs for Calculation
                        st.subheader("Calculation Parameters")
                        col1, col2 = st.columns(2)
                        with col1:
                            vol_input = st.number_input("Volatility (0.2 = 20%)", value=implied_vol if implied_vol > 0 else hist_vol, step=0.01, format="%.4f")
                        with col2:
                            rf_input = st.number_input("Risk-Free Rate", value=r_rate, step=0.001, format="%.4f")

                        if st.button("Calculate Theoretical Value"):
                            payload = {
                                "ticker": ticker,
                                "strike": float(selected_strike),
                                "expiration": selected_expiry,
                                "option_type": option_type.lower(),
                                "volatility": vol_input
                            }
                            
                            calc_res = requests.post(f"{API_BASE_URL}/calculate", json=payload)
                            if calc_res.status_code == 200:
                                result = calc_res.json()
                                theo_price = result["theoretical_price"]
                                greeks = result["greeks"]
                                
                                # Display Results
                                st.divider()
                                res_col1, res_col2 = st.columns(2)
                                
                                with res_col1:
                                    st.markdown("### Prices")
                                    st.metric("Theoretical Value", f"${theo_price:.2f}")
                                    st.metric("Market Price (Last)", f"${market_price:.2f}", delta=f"{theo_price - market_price:.2f}")
                                    
                                with res_col2:
                                    st.markdown("### Greeks")
                                    g_col1, g_col2 = st.columns(2)
                                    g_col1.metric("Delta", f"{greeks['delta']:.3f}")
                                    g_col1.metric("Gamma", f"{greeks['gamma']:.3f}")
                                    g_col1.metric("Theta", f"{greeks['theta']:.3f}")
                                    g_col2.metric("Vega", f"{greeks['vega']:.3f}")
                                    g_col2.metric("Rho", f"{greeks['rho']:.3f}")

                                # Visualization (Payoff)
                                st.divider()
                                st.subheader("Payoff Diagram (at Expiration)")
                                
                                spot_range = np.linspace(current_price * 0.5, current_price * 1.5, 100)
                                if option_type == "Call":
                                    payoff = np.maximum(spot_range - selected_strike, 0) - theo_price
                                else:
                                    payoff = np.maximum(selected_strike - spot_range, 0) - theo_price
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=spot_range, y=payoff, mode='lines', name='PnL'))
                                fig.add_vline(x=current_price, line_dash="dash", annotation_text="Current Price")
                                fig.add_vline(x=selected_strike, line_dash="solid", line_color="green", annotation_text="Strike")
                                fig.add_hline(y=0, line_color="gray")
                                fig.update_layout(title=f"PnL vs Spot Price for {ticker} {selected_strike} {option_type}", xaxis_title="Spot Price", yaxis_title="Profit/Loss")
                                st.plotly_chart(fig, use_container_width=True)

                            else:
                                st.error(f"Calculation Error: {calc_res.text}")

        else:
            st.error("Ticker not found or API error.")
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
