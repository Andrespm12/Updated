import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import os

class PortfolioReplayer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.trades = None
        self.stocks_data = {} # Cache for yfinance data
        
    def load_data(self):
        """Parses the IBKR HTML Activity Statement."""
        print(f"Loading data from: {self.filepath}")
        try:
            # Read all tables from HTML
            tables = pd.read_html(self.filepath)
            print(f"Found {len(tables)} tables.")
            
            # Find the 'Trades' table. 
            # Strategy: Look for specific columns usually found in IBKR Trades section.
            # Columns often include: 'Symbol', 'Date/Time', 'Quantity', 'T. Price', 'Comm/Fee'
            
            trades_df = pd.DataFrame()
            found = False
            
            best_df = None
            
            for i, df in enumerate(tables):
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    cols = ['_'.join(map(str, c)).strip() for c in df.columns]
                else:
                    cols = [str(c) for c in df.columns]
                
                cols_lower = [c.lower() for c in cols]
                print(f"Table {i} Columns: {cols_lower}")
                
                # Heuristic: Trades table usually has Date/Time, Symbol, Quantity, Price, Proceeds
                # We want to avoid 'Value Date' tables (Borrow Fees)
                if any('symbol' in c for c in cols_lower) and \
                   any('date/time' in c for c in cols_lower) and \
                   any('quantity' in c for c in cols_lower) and \
                   any('proceeds' in c for c in cols_lower):
                    print(f"!!! TRADES table found at index {i} !!!")
                    best_df = df.copy()
                    
                    # Clean the messy IBKR headers (e.g. symbol_symbol_symbol)
                    new_cols = []
                    for c in cols_lower:
                        # Take the first meaningful part. The format is usually "Symbol_Symbol_..."
                        # We can just check what the column *contains* and map it to a standard name.
                        if 'symbol' in c: new_cols.append('Symbol')
                        elif 'date/time' in c: new_cols.append('Date')
                        elif 'quantity' in c: new_cols.append('Quantity')
                        elif 't. price' in c: new_cols.append('Price')
                        elif 'comm' in c: new_cols.append('Commission')
                        elif 'basis' in c: new_cols.append('Basis')
                        elif 'proceeds' in c: new_cols.append('Proceeds')
                        else: new_cols.append(c) # Keep original if unknown matches
                        
                    best_df.columns = new_cols
                    break # Stop once we find the *real* trades table
            
            if best_df is None:
                 print("WARNING: No suitable Trades table found via heuristic.")
                 return None
            
            self.trades = self._clean_trades_data(best_df)
            print("Trades loaded and cleaned successfully.")
            return self.trades
            
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return None

    def _clean_trades_data(self, df):
        """Cleans raw HTML table data."""
        # IBKR HTML tables often have aggregation rows (Total, etc.) or Asset Class headers inside.
        # We need to filter for Asset Class == 'Stocks' if possible, or just cleaning headers.
        
        # 1. Standardize Columns
        # Sometimes header is row 0. Let's inspect first few rows if columns are integers.
        if isinstance(df.columns[0], int):
             # Try to find header row
             pass # Logic to be refined based on actual output inspection
             
        # Basic cleanup assuming headers matched 'symbol', 'quantity' etc.
        # Rename columns to standard internal names
        # Map: Symbol -> ticker, Date/Time -> date, Quantity -> qty, T. Price -> price, Comm/Fee -> comm
        
        # Case insensitive mapping
        col_map = {}
        for c in df.columns:
            cl = str(c).lower()
            if 'symbol' in cl: col_map[c] = 'Symbol'
            elif 'date/time' in cl: col_map[c] = 'Date'
            elif 'quantity' in cl: col_map[c] = 'Quantity'
            elif 't. price' in cl: col_map[c] = 'Price'
            elif 'comm/fee' in cl: col_map[c] = 'Commission'
            elif 'basis' in cl: col_map[c] = 'Basis'
            elif 'proceeds' in cl: col_map[c] = 'Proceeds'
            elif 'asset class' in cl: col_map[c] = 'AssetClass'
            
        df = df.rename(columns=col_map)
        
        # Filter for rows that actually look like trades (have a Date)
        # Drop rows where 'Date' is NaN or 'Total'
        if 'Date' in df.columns:
             # Force parsing with dayfirst=False (Assuming MDY which is typical for US, or ISO)
             # IBKR usually uses YYYY-MM-DD or similar. Let's try flexible parsing.
            df = df[pd.to_datetime(df['Date'], errors='coerce').notna()]
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Filter for Stocks if AssetClass column exists, otherwise assume mixed and we might need to rely on Symbol format
        if 'AssetClass' in df.columns:
            df = df[df['AssetClass'].astype(str).str.contains('Stock|STK', case=False, na=False)]
            
        # Convert numeric
        cols_to_num = ['Quantity', 'Price', 'Commission', 'Proceeds']
        for c in cols_to_num:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
                
        return df

    def generate_trade_log(self):
        """Matches trades using FIFO to generate a P&L log."""
        if self.trades is None: return pd.DataFrame()
        
        # Sort chronologically
        trades = self.trades.sort_values('Date').reset_index(drop=True)
        
        # Storage for Open Lots per ticker
        # Dictionary of Lists: { 'AAPL': [ {'date': ..., 'qty': ..., 'price': ...}, ... ] }
        open_lots = {}
        
        # Results
        closed_trades = []
        
        for _, trade in trades.iterrows():
            ticker = trade['Symbol']
            date = trade['Date']
            qty = trade['Quantity']
            price = trade['Price']
            comm = trade['Commission'] # We might allocate comms, but simplified request asked for Price PnL.
            # Let's subtract comms from PnL if we can match them?
            # For simplicity per prompt, we focus on Price matching first.
            
            if ticker not in open_lots:
                open_lots[ticker] = []
            
            lots = open_lots[ticker]
            
            # Determine if this trade OPENS or CLOSES risk
            # This is tricky if we flip from Long to Short.
            # Simplified assumption: 
            # If we are Long (lots > 0) and we Sell -> Close
            # If we are Short (lots < 0) and we Buy -> Close
            # Else -> Open
            
            current_net_pos = sum(l['qty'] for l in lots)
            
            is_closing = False
            if current_net_pos > 0 and qty < 0:
                is_closing = True
            elif current_net_pos < 0 and qty > 0:
                is_closing = True
                
            if is_closing:
                qty_to_close = abs(qty)
                
                # Match against lots (FIFO - from index 0)
                while qty_to_close > 1e-9 and lots:
                    match_lot = lots[0] # FIFO
                    
                    # Determine how much of this lot we can close
                    lot_qty_abs = abs(match_lot['qty'])
                    
                    if lot_qty_abs > qty_to_close:
                        # Partial Close of this Lot
                        closed_qty = qty_to_close
                        # Update lot remaining
                        remaining = lot_qty_abs - qty_to_close
                        # Restore sign
                        match_lot['qty'] = remaining * (1 if match_lot['qty'] > 0 else -1)
                        qty_to_close = 0
                    else:
                        # Full Close of this Lot
                        closed_qty = lot_qty_abs
                        qty_to_close -= closed_qty
                        lots.pop(0) # Remove empty lot
                        
                    # Calculate PnL
                    # If Lot was Buy (Pos), Trade is Sell. PnL = (SellPrice - BuyPrice) * Qty
                    # If Lot was Sell (Neg), Trade is Buy. PnL = (SellPrice - BuyPrice) * Qty (Sign handled via logic)
                    
                    # Standard PnL: Sales Proceeds - Cost Basis
                    if match_lot['qty'] > 0: # We are closing a Long
                         open_price = match_lot['price']
                         close_price = price
                         pnl = (close_price - open_price) * closed_qty
                         # Direction for report
                         direction = "Long"
                         q_bought = closed_qty
                         q_sold = closed_qty
                         p_bought = open_price
                         p_sold = close_price
                         d_open = match_lot['date']
                         d_close = date
                         
                    else: # We are closing a Short (Lot qty is negative, but we used abs above)
                         # We sold at match_lot['price'], buying back at 'price'
                         open_price = match_lot['price']
                         close_price = price
                         pnl = (open_price - close_price) * closed_qty # Short PnL
                         
                         direction = "Short"
                         # User asked for: qty bought, price bought...
                         # For Short: Initial was Sell, Closing was Buy.
                         # We map it to the logical columns.
                         d_open = match_lot['date']
                         d_close = date
                         # To fit "quantity bought" / "quantity sold" headers literal meaning:
                         q_bought = closed_qty # Closing trade
                         p_bought = close_price
                         q_sold = closed_qty # Opening trade
                         p_sold = open_price
                    
                    closed_trades.append({
                        "Ticker": ticker,
                        "Initial Trade Date": d_open,
                        "Quantity Bought": q_bought if direction == "Long" else 0, # Or just 'Quantity'
                        "Avg Price Bought": p_bought if direction == "Long" else 0,
                        "Date of Closing": d_close,
                        "Avg Selling Price": p_sold if direction == "Long" else 0,
                        "Quantity Sold": q_sold if direction == "Long" else 0,
                        
                        # Note: The user columns are specific. Let's try to adapt strictly:
                        # "initial trade date, quantity bought, avg price bought, date of closing, avg selling price, quantity sold"
                        # This schema fits Long trades perfectly. For shorts, it's weird.
                        # I will populate them mapping "Bought" to the Buy side of the pair, regardless of order.
                        
                        "Refined_InitDate": d_open,
                        "Refined_Qty": closed_qty,
                        "Refined_BuyPrice": p_bought if direction == "Long" else close_price,
                        "Refined_SellPrice": close_price if direction == "Long" else open_price,
                        "Refined_PnL": pnl
                    })
                
                # If we closed all lots and still have quantity (Flip position),
                # The remaining `qty_to_close` (in original sign logic) becomes a new Open Lot.
                # But here `qty_to_close` is 0 or positive remainder.
                # The original `qty` was the full trade.
                # If we exhausted lots and still have trade left, that part is an OPEN.
                if qty_to_close > 1e-9:
                    # Logic for flipping is complex. 
                    # Simpler approach: Process split trades? 
                    # If we ran out of lots, the remainder is a new lot in the *new* direction.
                    # Current loop handles the 'closing' part.
                    # The remainder should be added to lots.
                    
                    # Original trade was `qty` (e.g. -100). We closed 50 (lots). Remainder -50.
                    # We add -50 to lots.
                    rem_signed = qty_to_close * (-1 if qty < 0 else 1)
                    lots.append({
                        'date': date,
                        'qty': rem_signed,
                        'price': price
                    })
            
            else:
                # Open new Lot
                lots.append({
                    'date': date,
                    'qty': qty,
                    'price': price
                })
        
        # Format DataFrame
        df = pd.DataFrame(closed_trades)
        
        if df.empty:
            return df
            
        # Select and Rename to match User Prompt Exactly
        # "initial trade date, quantity bought, avg price bought, date of closing, avg selling price, quantity sold, PnL"
        
        # My intermediate computed columns:
        # Refined_BuyPrice -> Price of the BUY leg
        # Refined_SellPrice -> Price of the SELL leg
        
        final_df = pd.DataFrame()
        final_df['Initial Trade Date'] = df['Refined_InitDate']
        final_df['Quantity Bought'] = df['Refined_Qty'] # Assuming Bought = Sold amount in a matched close
        final_df['Avg Price Bought'] = df['Refined_BuyPrice']
        final_df['Date of Closing'] = df['Date of Closing']
        final_df['Avg Selling Price'] = df['Refined_SellPrice']
        final_df['Quantity Sold'] = df['Refined_Qty']
        final_df['PnL'] = df['Refined_PnL']
        final_df['Ticker'] = df['Ticker'] # Adding Ticker as it's essential
        
        # Reorder
        cols = ['Ticker', 'Initial Trade Date', 'Quantity Bought', 'Avg Price Bought', 'Date of Closing', 'Avg Selling Price', 'Quantity Sold', 'PnL']
        return final_df[cols]

if __name__ == "__main__":
    # Hardcoded path for testing
    HTML_PATH = "/Users/andrespena/Desktop/I7656131_U7656133_20250101_20251231_AS_Fv2_6e7c0263b63d09764a454ac744e8cd84.html"
    
    replayer = PortfolioReplayer(HTML_PATH)
    replayer.load_data()
    
    print(f"\n--- Generating Trade Log (FIFO) ---")
    
    df_log = replayer.generate_trade_log()
    
    if not df_log.empty:
        # Sort by Closing Date
        df_log = df_log.sort_values('Date of Closing')
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        print(df_log.to_string())
        print("-" * 50)
        print(f"Total Realized P&L: ${df_log['PnL'].sum():,.2f}")
        
        # Save to CSV
        output_file = "IBKR_Trade_Log_2025.csv"
        df_log.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")
    else:
        print("No closed trades found.")
