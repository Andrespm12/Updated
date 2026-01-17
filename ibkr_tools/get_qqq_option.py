import asyncio
from ib_insync import *

async def main():
    ib = IB()
    try:
        # Common ports: 7496 (TWS Live), 7497 (TWS Paper), 4001 (Gateway Live), 4002 (Gateway Paper)
        connected = False
        for port in [7496, 7497, 4001, 4002]:
            try:
                # print(f"Attempting to connect to port {port}...")
                await ib.connectAsync('127.0.0.1', port, clientId=98)
                print(f"Connected to port {port}")
                connected = True
                break
            except Exception:
                pass
        
        if not connected:
            print("Could not connect to IBKR TWS/Gateway.")
            return

        # 1. Define QQQ Stock
        qqq = Stock('QQQ', 'SMART', 'USD')
        await ib.qualifyContractsAsync(qqq)
        
        # 2. Construct the contract manually for Jan 20, 2023 (Standard Expiration)
        # Strike 442.0, Call.
        # Note: 'SMART' might fail for expired, but let's try. Sometimes we need 'CBOE' etc.
        # But 'SMART' is usually best for data.
        target_contract = Option('QQQ', '20230120', 442.0, 'C', 'SMART', 'USD')
        # Do NOT qualify or reqDetails, as it fails for expired options often. 
        # Just ask for data with what we have.
        
        # 3. Request Historical Data for this Option on Jan 3, 2022
        print(f"Requesting historical data for guessed contract: {target_contract} on 2022-01-03...")
        
        bars = await ib.reqHistoricalDataAsync(
            target_contract,
            endDateTime='20220104 23:59:59',
            durationStr='1 W',
            barSizeSetting='1 day',
            whatToShow='MIDPOINT',
            useRTH=True,
            formatDate=1
        )
        
        target_date = datetime.date(2022, 1, 3)
        target_bar = None
        for bar in bars:
            if bar.date == target_date:
                target_bar = bar
                break
        
        if target_bar:
            print("-" * 30)
            print(f"Option Price on {target_date} (Strike 442, Exp Jan 20 2023):")
            print(f"Close: {target_bar.close}")
            print(f"Open:  {target_bar.open}")
            print(f"High:  {target_bar.high}")
            print(f"Low:   {target_bar.low}") 
            print("-" * 30)
        else:
            print(f"No data found specifically for {target_date}.")
            if bars:
                print(f"Available dates: {[b.date for b in bars]}")
            else:
                print("No bars returned. Contract might be invalid or data unavailable.")

        ib.disconnect()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    import datetime
    asyncio.run(main())
