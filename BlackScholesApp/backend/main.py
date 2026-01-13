from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from calculator import BlackScholes
from data_service import DataService

app = FastAPI(title="Black-Scholes API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CalculateRequest(BaseModel):
    ticker: str
    strike: float
    expiration: str # YYYY-MM-DD
    option_type: str # "call" or "put"
    volatility: Optional[float] = None # If None, use Historical Volatility

@app.get("/market-data/{ticker}")
def get_market_data(ticker: str):
    price = DataService.get_current_price(ticker)
    if price is None:
        raise HTTPException(status_code=404, detail="Ticker not found")
    
    risk_free = DataService.get_risk_free_rate()
    hv = DataService.get_historical_volatility(ticker)
    expirations = DataService.get_expirations(ticker)
    
    return {
        "current_price": price,
        "risk_free_rate": risk_free,
        "historical_volatility": hv,
        "expirations": expirations
    }

@app.get("/option-chain/{ticker}")
def get_option_chain(ticker: str, date: str):
    chain = DataService.get_option_chain(ticker, date)
    if chain is None:
        raise HTTPException(status_code=404, detail="Option chain not found")
    return chain

@app.post("/calculate")
def calculate_option(req: CalculateRequest):
    # 1. Get Stock Price
    S = DataService.get_current_price(req.ticker)
    if not S:
        raise HTTPException(status_code=404, detail="Ticker not found")

    # 2. Derive Time to Expiry (T)
    try:
        exp_date = datetime.strptime(req.expiration, "%Y-%m-%d")
        today = datetime.now()
        T = (exp_date - today).days / 365.0
        if T <= 0:
            T = 0.0001 # Minimal time if expired/today
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")

    # 3. Risk Free Rate
    r = DataService.get_risk_free_rate()

    # 4. Volatility (Use provided or Historical)
    sigma = req.volatility if req.volatility else DataService.get_historical_volatility(req.ticker)

    # 5. Calculate
    if req.option_type.lower() == "call":
        theoretical_price = BlackScholes.call_price(S, req.strike, T, r, sigma)
    else:
        theoretical_price = BlackScholes.put_price(S, req.strike, T, r, sigma)
        
    greeks = BlackScholes.calculate_greeks(S, req.strike, T, r, sigma, req.option_type.lower())

    return {
        "ticker": req.ticker,
        "current_stock_price": S,
        "strike": req.strike,
        "expiry": req.expiration,
        "time_to_expiry_years": T,
        "volatility_used": sigma,
        "risk_free_rate_used": r,
        "theoretical_price": theoretical_price,
        "greeks": greeks
    }
