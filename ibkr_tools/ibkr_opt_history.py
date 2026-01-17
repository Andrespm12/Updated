import asyncio
import numpy as np
from scipy.stats import norm
from ib_insync import *
from datetime import datetime, timedelta
import pandas as pd
from rich.console import Console
from rich.table import Table

# Configuration
HOST = '127.0.0.1'
PORT = 7496
CLIENT_ID = 5
SYMBOL = 'SPY'
STRIKE = 690  # Adjust based on current market (approx ATM)
RIGHT = 'C'   # Call
EXPIRY_LOOKAHEAD_DAYS = 30 # Approx next monthly

def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Greeks using Black-Scholes.
    S: Spot Price
    K: Strike Price
    T: Time to Expiry (in years)
    r: Risk-free rate
    sigma: Implied Volatility
    """
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            delta = norm.cdf(d1)
            theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            delta = norm.cdf(d1) - 1
            theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)

        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Divide by 100 for standard interpretation
        
        # Theta usually typically expressed per day
        theta = theta / 365.0

        return delta, gamma, theta, vega
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

async def main():
    ib = IB()
    console = Console()
    
    try:
        console.print(f"[bold yellow]Connecting to {HOST}:{PORT}...[/bold yellow]")
        await ib.connectAsync(HOST, PORT, clientId=CLIENT_ID, timeout=15)
        console.print("[bold green]Connected![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Connection failed: {repr(e)}[/bold red]")
        return

    # 1. Find a Contract (Next Monthly Expiry)
    console.print(f"Searching for {SYMBOL} Options around strike {STRIKE}...")
    spx = Stock(SYMBOL, 'SMART', 'USD')
    await ib.qualifyContractsAsync(spx)
    
    chains = await ib.reqSecDefOptParamsAsync(spx.symbol, '', spx.secType, spx.conId)
    
    # Pick the 'SMART' chain
    smart_chain = next((c for c in chains if c.exchange == 'SMART'), chains[0])
    
    # Find expiry ~30 days out
    target_date = datetime.now() + timedelta(days=EXPIRY_LOOKAHEAD_DAYS)
    # Sort expirations and find closest
    expirations = sorted([datetime.strptime(e, '%Y%m%d') for e in smart_chain.expirations])
    expiry = min(expirations, key=lambda x: abs(x - target_date)).strftime('%Y%m%d')
    
    console.print(f"Selected Expiry: [bold cyan]{expiry}[/bold cyan]")
    
    # Use valid contract discovery
    console.print("Fetching valid contracts for this expiry...")
    temp_contract = Option(SYMBOL, expiry, exchange='SMART')
    details = await ib.reqContractDetailsAsync(temp_contract)
    
    if not details:
        console.print("[red]No contracts found![/red]")
        return
        
    calls = [d.contract for d in details if d.contract.right == 'C']
    
    # Current SPX price?
    [ticker] = await ib.reqTickersAsync(spx)
    spot_price = ticker.marketPrice() or ticker.close or 690
    
    contract = min(calls, key=lambda c: abs(c.strike - spot_price))
    console.print(f"Spot: {spot_price:.2f} -> Selected Strike: [bold cyan]{contract.strike}[/bold cyan]")
    
    await ib.qualifyContractsAsync(contract)
    console.print(f"Resolving: {contract.localSymbol}")
    
    # Enable Delayed Data just in case
    ib.reqMarketDataType(3)

    # 2. Fetch Historical Data
    # We need Spot Price history and Option IV history.
    # Actually, simpler to fetch Option Price history and Underlying Price history, 
    # then calculate Implied Vol (or use provided IV)
    
    # Let's fetch Option Price (Midpoint) and Option IV (from IBKR)
    end_time = ''
    duration = '1 W' # Try shorter duration
    bar_size = '1 hour'
    
    console.print("Fetching Historical Prices and IV (1 Week, 1 Hour bars)...")
    
    # A. Option Price
    # Try TRADES instead of MIDPOINT
    bars_price = await ib.reqHistoricalDataAsync(
        contract, end_time, duration, bar_size, 'TRADES', 1, 1, False
    )
    
    if not bars_price:
        console.print("[bold red]No historical price data found for this option.[/bold red]")
        return

    # B. Option IV
    bars_iv = await ib.reqHistoricalDataAsync(
        contract, end_time, duration, bar_size, 'OPTION_IMPLIED_VOLATILITY', 1, 1, False
    )
    
    # C. Underlying Price (for Spot)
    bars_underlying = await ib.reqHistoricalDataAsync(
        spx, end_time, duration, bar_size, 'MIDPOINT', 1, 1, False
    )
    
    # Merge Data
    df_price = util.df(bars_price).set_index('date')['close'].rename('opt_price')
    
    if bars_iv:
         df_iv = util.df(bars_iv).set_index('date')['close'].rename('iv')
    else:
         console.print("[red]Warning: No IV data found. Using constant 0.20[/red]")
         df_iv = pd.Series(0.20, index=df_price.index, name='iv')

    if bars_underlying:
         df_spot = util.df(bars_underlying).set_index('date')['close'].rename('spot')
    else:
         # Fallback to last known spot?
         df_spot = pd.Series(spot_price, index=df_price.index, name='spot')
    
    df = pd.concat([df_price, df_iv, df_spot], axis=1).dropna()
    
    # 3. Calculate Greeks
    greeks_list = []
    r_rate = 0.045 # Approx risk free rate 4.5%
    expire_dt = datetime.strptime(expiry, '%Y%m%d').date()
    
    for date, row in df.iterrows():
        # T in years
        T = (expire_dt - date).days / 365.0
        if T <= 0: T = 0.0001
        
        d, g, t, v = black_scholes_greeks(
            row['spot'], 
            STRIKE, 
            T, 
            r_rate, 
            row['iv']
        )
        greeks_list.append({'delta': d, 'gamma': g, 'theta': t, 'vega': v})
        
    df_greeks = pd.DataFrame(greeks_list, index=df.index)
    final_df = pd.concat([df, df_greeks], axis=1)
    
    # 4. Display
    console.print(f"\n[bold white underline]HISTORICAL GREEKS: {contract.localSymbol}[/bold white underline]\n")
    
    table = Table(box=box.ROUNDED)
    table.add_column("Date", style="dim")
    table.add_column("Spot", justify="right")
    table.add_column("Opt Price", justify="right")
    table.add_column("IV", justify="right", style="cyan")
    table.add_column("Delta", justify="right", style="green")
    table.add_column("Gamma", justify="right")
    table.add_column("Theta", justify="right")
    table.add_column("Vega", justify="right")
    
    for date, row in final_df.tail(10).iterrows():
        table.add_row(
            str(date),
            f"{row['spot']:.2f}",
            f"{row['opt_price']:.2f}",
            f"{row['iv']:.3f}",
            f"{row['delta']:.3f}",
            f"{row['gamma']:.3f}",
            f"{row['theta']:.3f}",
            f"{row['vega']:.3f}",
        )
        
    console.print(table)
    console.print(f"\n[dim]Complete history saved to {contract.localSymbol}_greeks.csv[/dim]")
    final_df.to_csv(f"{contract.localSymbol}_greeks.csv")
    
    ib.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
