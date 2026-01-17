import asyncio
from ib_insync import *
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich import box

# Configuration
HOST = '127.0.0.1'
PORT = 7496
CLIENT_ID = 3

# Watchlist of High Volatility / Popular Stocks to scan client-side
WATCHLIST = [
    'NVDA', 'TSLA', 'AMD', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NFLX', 
    'SPY', 'QQQ', 'IWM', 'COIN', 'MARA', 'PLTR', 'SOFI', 'UBER', 'RIVN', 'GME', 'AMC'
]

async def main():
    ib = IB()
    console = Console()
    
    try:
        console.print(f"[bold yellow]Connecting to {HOST}:{PORT}...[/bold yellow]")
        ib.RequestTimeout = 15
        await ib.connectAsync(HOST, PORT, clientId=CLIENT_ID, timeout=15)
        console.print("[bold green]Connected![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Connection failed: {repr(e)}[/bold red]")
        return

    # Create contracts
    contracts = [Stock(s, 'SMART', 'USD') for s in WATCHLIST]
    
    # Qualify
    with console.status("[bold green]Qualifying contracts...[/bold green]"):
        await ib.qualifyContractsAsync(*contracts)
    
    # Request DELAYED Market Data (Type 3)
    ib.reqMarketDataType(3)

    # Request Market Data
    for c in contracts:
        ib.reqMktData(c, '', False, False)
        
    console.print("[bold yellow]Scanning market data... (waiting 5s)[/bold yellow]")
    await asyncio.sleep(5) # Wait for ticks to arrive
    
    # Collect Data
    data = []
    for c in contracts:
        t = ib.ticker(c)
        if t.marketPrice():
            price = t.marketPrice()
            # If close is missing, use open or price
            close = t.close if t.close else price
            change_pct = ((price - close) / close) * 100 if close else 0.0
            volume = t.volume or 0
            
            data.append({
                'symbol': c.symbol,
                'price': price,
                'change_pct': change_pct,
                'volume': volume,
                'volatility': abs(change_pct) # Simple proxy
            })
            
    # Sort for scanners
    top_gainers = sorted(data, key=lambda x: x['change_pct'], reverse=True)[:10]
    most_active = sorted(data, key=lambda x: x['volume'], reverse=True)[:10]
    
    # Display Results
    console.print("\n[bold white underline]HIGH VOLATILITY CLIENT SCANNER[/bold white underline]\n")
    
    layout = Layout()
    layout.split_row(
        Layout(name="left"),
        Layout(name="right")
    )
    
    # Gainers Table
    table_g = Table(title="TOP MOVERS (%)", box=box.ROUNDED)
    table_g.add_column("Symbol", style="cyan")
    table_g.add_column("Price", justify="right")
    table_g.add_column("% Chg", justify="right", style="bold")
    
    for d in top_gainers:
        color = "green" if d['change_pct'] >= 0 else "red"
        table_g.add_row(
            d['symbol'],
            f"{d['price']:.2f}",
            f"[{color}]{d['change_pct']:+.2f}%[/{color}]"
        )
        
    # Active Table
    table_v = Table(title="MOST ACTIVE (Vol)", box=box.ROUNDED)
    table_v.add_column("Symbol", style="cyan")
    table_v.add_column("Volume", justify="right", style="magenta")
    table_v.add_column("Price", justify="right")

    for d in most_active:
        table_v.add_row(
            d['symbol'],
            f"{d['volume']:,}",
            f"{d['price']:.2f}"
        )

    layout["left"].update(table_g)
    layout["right"].update(table_v)
    
    console.print(layout)
    
    ib.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
