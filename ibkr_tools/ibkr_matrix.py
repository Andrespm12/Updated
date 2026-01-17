import asyncio
from ib_insync import *
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich import box
from datetime import datetime

# Configuration
HOST = '127.0.0.1'
PORT = 7496
CLIENT_ID = 2
SYMBOLS = ['SPY', 'QQQ', 'IWM', 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMD']

def generate_table(tickers):
    """Generate the main market data table."""
    table = Table(title="MARKET MATRIX", box=box.ROUNDED, style="cyan")
    table.add_column("Symbol", style="bold white")
    table.add_column("Last", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("% Chg", justify="right")
    table.add_column("Bid", justify="right")
    table.add_column("Ask", justify="right")
    table.add_column("Volume", justify="right")

    for ticker in tickers:
        try:
            market_price = ticker.marketPrice() or 0.0
            close_price = ticker.close or market_price
            
            # If we don't have a valid close yet (e.g. pre-market), use yesterday's close if available
            # Note: ib_insync fills 'close' with previous day close usually.
            
            change = market_price - close_price
            pct_change = (change / close_price) * 100 if close_price else 0.0
            
            # Color coding
            color = "green" if change >= 0 else "red"
            emoji = "▲" if change >= 0 else "▼"
            
            
            table.add_row(
                ticker.contract.symbol,
                f"{market_price:.2f}",
                f"[{color}]{change:+.2f}[/{color}]",
                f"[{color}]{emoji} {pct_change:+.2f}%[/{color}]",
                f"{ticker.bid:.2f}",
                f"{ticker.ask:.2f}",
                f"{ticker.volume:,}"
            )
        except Exception:
            table.add_row(ticker.contract.symbol, "N/A", "-", "-", "-", "-", "-")
            
    return table

def generate_log_panel(log_messages):
    """Generate a panel for streaming logs/trades."""
    log_text = "\n".join(log_messages[-15:]) # Show last 15 messages
    return Panel(
        Text(log_text, style="green"),
        title="[bold yellow]LIVE FEED - NVDA",
        border_style="yellow",
        box=box.ROUNDED
    )

async def main():
    ib = IB()
    try:
        print(f"Connecting to {HOST}:{PORT}...")
        # Increase timeout for connection
        ib.RequestTimeout = 15
        await ib.connectAsync(HOST, PORT, clientId=CLIENT_ID, timeout=15)
        print("Connected.")
    except Exception as e:
        print(f"Connection failed: {repr(e)}")
        return

    # Create contracts
    contracts = [Stock(s, 'SMART', 'USD') for s in SYMBOLS]
    
    # Qualify contracts (resolves conId, etc.)
    await ib.qualifyContractsAsync(*contracts)
    
    # Request market data
    for contract in contracts:
        ib.reqMktData(contract, '', False, False)

    # Specific focus on NVDA for the 'tape'
    nvda_contract = next((c for c in contracts if c.symbol == 'NVDA'), contracts[0])
    
    # Just reusing the 'contracts' list as our ticker objects? 
    # Wait, ib.reqMktData returns a Ticker object or we get it from ib.tickers()
    # Let's map contracts to Tickers
    tickers = [ib.ticker(c) for c in contracts]
    
    log_messages = []

    # Callback for updates on NVDA
    def on_ticker_update(t):
        if t.contract.symbol == 'NVDA':
            timestamp = datetime.now().strftime("%H:%M:%S")
            # Simple trade tape simulation using last trade
            if t.last:
                log_messages.append(f"[{timestamp}] TRADE: {t.last:.2f} x {t.lastSize or '?'}")

    ib.pendingTickersEvent += on_ticker_update

    # Live Loop
    with Live(refresh_per_second=4, screen=True) as live:
        while ib.isConnected():
            await asyncio.sleep(0.2)
            
            # Create Layout
            layout = Layout()
            layout.split_row(
                Layout(name="left", ratio=2),
                Layout(name="right", ratio=1)
            )
            
            layout["left"].update(generate_table(tickers))
            layout["right"].update(generate_log_panel(log_messages))
            
            live.update(layout)

    ib.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
