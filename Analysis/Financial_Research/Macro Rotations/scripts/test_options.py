import yfinance as yf
import pandas as pd
import time

def test_options_data(ticker_symbol):
    print(f"Fetching options for {ticker_symbol}...")
    start_time = time.time()
    
    ticker = yf.Ticker(ticker_symbol)
    
    try:
        # Get expiration dates
        exps = ticker.options
        if not exps:
            print("No expirations found.")
            return

        print(f"Found {len(exps)} expirations. Fetching nearest: {exps[0]}")
        
        # Get option chain for nearest expiration
        opt = ticker.option_chain(exps[0])
        
        calls = opt.calls
        puts = opt.puts
        
        print(f"Calls: {len(calls)}, Puts: {len(puts)}")
        
        # Calculate Put/Call Volume Ratio for this expiration
        call_vol = calls['volume'].sum()
        put_vol = puts['volume'].sum()
        pcr = put_vol / call_vol if call_vol > 0 else 0
        
        print(f"Put Volume: {put_vol}")
        print(f"Call Volume: {call_vol}")
        print(f"P/C Ratio (Nearest Exp): {pcr:.2f}")
        
        # Check for Implied Volatility data
        if 'impliedVolatility' in calls.columns:
            avg_iv_call = calls['impliedVolatility'].mean()
            print(f"Avg Call IV: {avg_iv_call:.2%}")
            
    except Exception as e:
        print(f"Error: {e}")
        
    print(f"Time taken: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    test_options_data("SPY")
