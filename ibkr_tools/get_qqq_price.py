import asyncio
from ib_insync import *
import datetime

async def main():
    ib = IB()
    try:
        # Common ports: 7496 (TWS Live), 7497 (TWS Paper), 4001 (Gateway Live), 4002 (Gateway Paper)
        connected = False
        for port in [7496, 7497, 4001, 4002]:
            try:
                print(f"Attempting to connect to port {port}...")
                await ib.connectAsync('127.0.0.1', port, clientId=99)
                print(f"Connected to port {port}")
                connected = True
                break
            except Exception as e:
                # print(f"Failed to connect to {port}: {e}")
                pass
        
        if not connected:
            print("Could not connect to IBKR TWS/Gateway on any common port.")
            return

        qqq = Stock('QQQ', 'SMART', 'USD')
        
        print("Requesting historical data for QQQ around Jan 2022...")
        # Request data ending Jan 10, 2022, look back 1 month to ensure we cover the start of the year
        # Note: times in IBKR are often UTC or exchange time. Daily bars date is usually date object.
        bars = await ib.reqHistoricalDataAsync(
            qqq,
            endDateTime='20220110 23:59:59',
            durationStr='1 M',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        if not bars:
            print("No historical data returned.")
            ib.disconnect()
            return

        # Filters for bars in 2022
        bars_2022 = [b for b in bars if b.date.year == 2022]
        bars_2022.sort(key=lambda x: x.date)
        
        if bars_2022:
            first_day = bars_2022[0]
            print("-" * 30)
            print(f"First Trading Day of 2022 found: {first_day.date}")
            print("-" * 30)
            print(f"Open:  {first_day.open}")
            print(f"High:  {first_day.high}")
            print(f"Low:   {first_day.low}")
            print(f"Close: {first_day.close}")
            print(f"Volume:{first_day.volume}")
            print("-" * 30)
        else:
            print("No data found for Jan 2022 in the requested range.")

        ib.disconnect()
        print("Done.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    asyncio.run(main())
