import asyncio
import pandas as pd
import subprocess
import os
import sys

# Ensure we can import ib_service from current dir
sys.path.append(os.getcwd())
from ib_service import ib_service

# Configuration
GCLOUD_BIN = '/Users/andrespena/google-cloud-sdk/bin/gcloud'
BQ_BIN = '/Users/andrespena/google-cloud-sdk/bin/bq'
DATASET_ID = "market_data"
TABLE_ID = "historical_prices"

# Add SDK bin to PATH for bq dependency on gcloud
os.environ["PATH"] += os.pathsep + '/Users/andrespena/google-cloud-sdk/bin'

async def main():
    print("Getting Project ID...")
    try:
        project_id = subprocess.check_output([GCLOUD_BIN, 'config', 'get-value', 'project']).decode().strip()
        print(f"Using Project ID: {project_id}")
    except Exception as e:
        print(f"Error getting project ID: {e}")
        return

    print("Connecting to IBKR...")
    connected = await ib_service.connect()
    if not connected:
        print("Could not connect to IBKR. Ensure Trader Workstation/Gateway is open.")
        return
    
    symbol = "SPY"
    print(f"Fetching historical data for {symbol}...")
    # Fetch 1 year of data
    df = await ib_service.get_historical_data(symbol, duration='1 Y', bar_size='1 day')
    
    if df is not None and not df.empty:
        print(f"Retrieved {len(df)} records.")
        
        # Prepare for BQ
        # Convert date to string for CSV
        df['date'] = df['date'].astype(str)
        # Add symbol column
        df['symbol'] = symbol
        
        csv_file = f"{symbol}_history.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved to {csv_file}")
        
        # Create dataset if not exists
        print("Creating dataset if needed...")
        # Ignore error if exists
        subprocess.run([BQ_BIN, '--location=US', 'mk', '--dataset', f'{project_id}:{DATASET_ID}'], capture_output=True)
        
        # Load to BQ
        print("Uploading to BigQuery...")
        # Use autodetect for schema
        cmd = [
            BQ_BIN, 'load', 
            '--source_format=CSV', 
            '--autodetect', 
            '--replace', # Replace table content
            f'{project_id}:{DATASET_ID}.{TABLE_ID}', 
            csv_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("Upload successful!")
            print(result.stdout)
        else:
            print("Upload failed.")
            print(result.stdout)
            print(result.stderr)
            
    else:
        print("No data found or empty dataframe.")
    
    ib_service.disconnect()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
