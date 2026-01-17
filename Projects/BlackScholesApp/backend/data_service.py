import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class DataService:
    @staticmethod
    def get_current_price(ticker: str):
        stock = yf.Ticker(ticker)
        # fast check for price
        hist = stock.history(period="1d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])

    @staticmethod
    def get_risk_free_rate():
        # Approximation: 10 Year Treasury Note Yield
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="1d")
        if hist.empty:
            return 0.045 # Default fallback
        # Yield is in percent, e.g., 4.5 -> 0.045
        return float(hist["Close"].iloc[-1]) / 100

    @staticmethod
    def get_historical_volatility(ticker: str, lookback_days=252):
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{lookback_days}d")
        hist['Log_Ret'] = np.log(hist['Close'] / hist['Close'].shift(1))
        # Annualized Volatility
        vol = hist['Log_Ret'].std() * np.sqrt(252)
        return float(vol)

    @staticmethod
    def get_option_chain(ticker: str, expiration_date: str):
        stock = yf.Ticker(ticker)
        try:
            opt = stock.option_chain(expiration_date)
            return {
                "calls": opt.calls.to_dict(orient="records"),
                "puts": opt.puts.to_dict(orient="records")
            }
        except Exception as e:
            print(f"Error fetching options: {e}")
            return None
            
    @staticmethod
    def get_expirations(ticker: str):
        stock = yf.Ticker(ticker)
        return stock.options
