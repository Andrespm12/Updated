import asyncio
from ib_insync import *
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich import box
from datetime import datetime, timedelta

# Configuration
HOST = '127.0.0.1'
PORT = 7496
CLIENT_ID = 6
SYMBOL = 'SPY'
EXPIRY_LOOKAHEAD_DAYS = 30

def generate_greeks_table(ticker):
    """Generate a table showing live greeks for the option."""
    table = Table(title=f"LIVE OPTIONS CHAIN: {ticker.contract.localSymbol}", box=box.ROUNDED)
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right", style="bold")
    
    # Prices
    table.add_row("Bid", f"{ticker.bid:.2f}")
    table.add_row("Ask", f"{ticker.ask:.2f}")
    table.add_row("Last", f"{ticker.last:.2f}")
    
    # Model Greeks (tickOptionComputation)
    # modelGreeks is an object: OptionComputation(impliedVol, delta, optPrice, pvDividend, gamma, vega, theta, undPrice)
    greeks = ticker.modelGreeks
    
    if greeks:
        iv = greeks.impliedVol
        delta = greeks.delta
        gamma = greeks.gamma
        vega = greeks.vega
        theta = greeks.theta
        
        # Formatting handles None
        table.add_row("IV", f"{iv:.2%}" if iv else "-")
        table.add_row("Delta", f"{delta:.3f}" if delta else "-", style="green")
        table.add_row("Gamma", f"{gamma:.3f}" if gamma else "-")
        table.add_row("Vega", f"{vega:.3f}" if vega else "-")
        table.add_row("Theta", f"{theta:.3f}" if theta else "-", style="red")
        table.add_row("Und Price", f"{greeks.undPrice:.2f}" if greeks.undPrice else "-")
    else:
        table.add_row("Greeks", "Waiting for data...")

    return table

async def main():
    ib = IB()
    try:
        print(f"Connecting to {HOST}:{PORT}...")
        await ib.connectAsync(HOST, PORT, clientId=CLIENT_ID, timeout=15)
        print("Connected.")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # 1. Find a Contract
    print(f"Locating ATM {SYMBOL} Call...")
    spx = Stock(SYMBOL, 'SMART', 'USD')
    await ib.qualifyContractsAsync(spx)
    
    # Get Market Price for ATM
    [ticker] = await ib.reqTickersAsync(spx)
    ref_price = ticker.marketPrice() or ticker.close or 690
    strike = round(ref_price / 5) * 5 # Round to nearest 5
    
    chains = await ib.reqSecDefOptParamsAsync(spx.symbol, '', spx.secType, spx.conId)
    smart_chain = next((c for c in chains if c.exchange == 'SMART'), chains[0])
    
    target_date = datetime.now() + timedelta(days=EXPIRY_LOOKAHEAD_DAYS)
    expirations = sorted([datetime.strptime(e, '%Y%m%d') for e in smart_chain.expirations])
    expiry = min(expirations, key=lambda x: abs(x - target_date)).strftime('%Y%m%d')
    
    print(f"Selected Expiry: {expiry}")
    
    # Use reqContractDetails to find actual valid contracts for this expiry
    # This avoids guessing strikes that don't exist
    print("Fetching valid contracts for this expiry...")
    temp_contract = Option(SYMBOL, expiry, exchange='SMART')
    details = await ib.reqContractDetailsAsync(temp_contract)
    
    if not details:
        print("No contracts found for this expiry!")
        return

    # Filter for Calls
    calls = [d.contract for d in details if d.contract.right == 'C']
    if not calls:
        print("No calls found!")
        return
        
    # Find closest to strike
    contract = min(calls, key=lambda c: abs(c.strike - strike))
    print(f"Locked on: {contract.localSymbol} (Strike: {contract.strike})")
    
    await ib.qualifyContractsAsync(contract)
    
    # Enable Delayed Data
    ib.reqMarketDataType(3)
    
    # 2. Request Data
    # GenericTickList 100 (Option Volume), 101 (Open Interest), 104 (Hist Vol), 106 (Implied Vol)
    ib.reqMktData(contract, '100,101,104,106', False, False)
    
    ticker = ib.ticker(contract)
    
    # 3. Live Loop
    print(f"Streaming data for {contract.localSymbol}...")
    
    with Live(generate_greeks_table(ticker), refresh_per_second=4) as live:
        while ib.isConnected():
            await asyncio.sleep(0.2)
            live.update(generate_greeks_table(ticker))

    ib.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
