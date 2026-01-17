import asyncio
from ib_insync import *

async def main():
    ib = IB()
    try:
        # Connect to TWS/Gateway
        # Common ports: 7496 (TWS Live), 7497 (TWS Paper), 4001 (Gateway Live), 4002 (Gateway Paper)
        # We'll try 7497 first as it's common for dev/paper, but user said "App currently opened" so likely TWS.
        # User didn't specify paper/live, so we can try a few standard ports.
        
        print("Attempting to connect to IBKR TWS/Gateway...")
        
        # Try TWS Live first (7496)
        try:
            await ib.connectAsync('127.0.0.1', 7496, clientId=1)
            print("Connected to port 7496 (TWS Live)")
        except Exception:
            try:
                # Try TWS Paper (7497)
                await ib.connectAsync('127.0.0.1', 7497, clientId=1)
                print("Connected to port 7497 (TWS Paper)")
            except Exception:
                 # Try Gateway Live (4001)
                try:
                    await ib.connectAsync('127.0.0.1', 4001, clientId=1)
                    print("Connected to port 4001 (Gateway Live)")
                except Exception:
                    # Try Gateway Paper (4002)
                    await ib.connectAsync('127.0.0.1', 4002, clientId=1)
                    print("Connected to port 4002 (Gateway Paper)")

        print("Connection established successfully!")
        
        # Get account summary to prove it works
        account_summary = ib.accountSummary()
        if account_summary:
            print(f"Account Summary retrieved. Found {len(account_summary)} items.")
        else:
            print("Connected, but no account summary data available (check TWS permissions/subscription).")

        # Keeping it simple for now, just close.
        ib.disconnect()
        print("Disconnected.")

    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == '__main__':
    asyncio.run(main())
