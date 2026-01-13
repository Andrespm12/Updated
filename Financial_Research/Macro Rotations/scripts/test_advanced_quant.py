import yfinance as yf
import pandas as pd
import time

def test_advanced_quant(ticker_symbol="SPY"):
    print(f"--- Testing Advanced Quant for {ticker_symbol} ---")
    start_time = time.time()
    
    ticker = yf.Ticker(ticker_symbol)
    exps = ticker.options
    if not exps:
        print("No options found.")
        return

    # Fetch nearest expiration
    exp_date = exps[0]
    print(f"Expiration: {exp_date}")
    opt = ticker.option_chain(exp_date)
    
    calls = opt.calls
    puts = opt.puts
    
    # 1. Gamma Proxy (Net Open Interest)
    # Real GEX requires delta/gamma per strike. 
    # Proxy: (Call OI - Put OI) at each strike to see positioning.
    
    # Merge on strike
    df_opts = pd.merge(calls[['strike', 'openInterest']], puts[['strike', 'openInterest']], on='strike', suffixes=('_call', '_put'))
    df_opts['net_oi'] = df_opts['openInterest_call'] - df_opts['openInterest_put']
    
    # Find "Gamma Flip" proxy? (Strike where Net OI flips from positive to negative?)
    # Or just Total Net OI
    total_net_oi = df_opts['net_oi'].sum()
    print(f"Total Net Open Interest (Calls - Puts): {total_net_oi}")
    
    # Max Pain (Strike with lowest total value of options expiring)
    # Value = |Price - Strike| * OI
    # Need current price
    current_price = ticker.history(period='1d')['Close'].iloc[-1]
    print(f"Current Price: {current_price:.2f}")
    
    pain_data = []
    for index, row in df_opts.iterrows():
        strike = row['strike']
        call_val = max(0, current_price - strike) * row['openInterest_call']
        put_val = max(0, strike - current_price) * row['openInterest_put']
        pain_data.append({'strike': strike, 'pain': call_val + put_val})
        
    df_pain = pd.DataFrame(pain_data)
    max_pain_strike = df_pain.loc[df_pain['pain'].idxmin()]['strike']
    print(f"Max Pain Strike: {max_pain_strike}")
    
    print(f"Time taken: {time.time() - start_time:.2f}s")

def test_bond_vol():
    print("\n--- Testing Bond Volatility (TLT) ---")
    ticker = yf.Ticker("TLT")
    exps = ticker.options
    if exps:
        opt = ticker.option_chain(exps[0])
        calls = opt.calls
        if 'impliedVolatility' in calls.columns:
            avg_iv = calls['impliedVolatility'].mean()
            print(f"TLT Implied Volatility (Avg Call): {avg_iv:.2%}")

if __name__ == "__main__":
    test_advanced_quant("SPY")
    test_bond_vol()
