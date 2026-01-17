import asyncio
from ib_insync import *
from datetime import datetime, timedelta
import pandas as pd
from scipy.stats import norm
import numpy as np

# Use Singleton pattern for IB connection
class IBService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(IBService, cls).__new__(cls)
            cls._instance.ib = None # Delayed init to capture correct loop
            cls._instance.host = '127.0.0.1'
            cls._instance.port = 7496
            cls._instance.client_id = 10
            cls._instance.connected = False
        return cls._instance

    async def connect(self):
        # Instantiate IB here to ensure it uses the current asyncio loop (Uvicorn's loop)
        if self.ib is None:
            self.ib = IB()
            
        if not self.ib.isConnected():
            try:
                print(f"Connecting to {self.host}:{self.port}...")
                await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
                self.ib.reqMarketDataType(3) # Delayed data
                self.connected = True
                print("Web App Connected to IBKR")
            except Exception as e:
                print(f"Connection error: {e}")
                self.connected = False
        return self.connected

    def disconnect(self):
        if self.ib.isConnected():
            self.ib.disconnect()
            self.connected = False

    async def search_options_chain(self, symbol: str, lookahead_days: int = 30):
        if not self.connected: await self.connect()
        
        contract = Stock(symbol, 'SMART', 'USD')
        await self.ib.qualifyContractsAsync(contract)
        
        chains = await self.ib.reqSecDefOptParamsAsync(contract.symbol, '', contract.secType, contract.conId)
        
        # Filter for SMART exchange
        smart_chain = next((c for c in chains if c.exchange == 'SMART'), None)
        if not smart_chain:
            return None
            
        # Parse expirations and find closest to target
        target_date = datetime.now() + timedelta(days=lookahead_days)
        exp_dates = sorted([datetime.strptime(e, '%Y%m%d') for e in smart_chain.expirations])
        
        # Return simplified struct
        return {
            'symbol': symbol,
            'expirations': [e.strftime('%Y%m%d') for e in exp_dates],
            'strikes': sorted(list(smart_chain.strikes)),
            'recommended_expiry': min(exp_dates, key=lambda x: abs(x - target_date)).strftime('%Y%m%d')
        }

    async def get_details(self, symbol, expiry, right):
         # Helper to find valid contracts
         contract = Option(symbol, expiry, exchange='SMART', right=right)
         details = await self.ib.reqContractDetailsAsync(contract)
         return details

    async def get_market_data_snapshot(self, symbol, expiry, strike, right):
        if not self.connected: await self.connect()
        
        # Build contract
        contract = Option(symbol, expiry, float(strike), right, 'SMART')
        await self.ib.qualifyContractsAsync(contract)
        
        # Request Snapshot
        # 100=OptVol, 101=OI, 104=HistVol, 106=ImpVol
        self.ib.reqMktData(contract, '100,101,104,106', True, False) # Snapshot=True
        
        # Wait a moment for data
        await asyncio.sleep(0.5)
        
        ticker = self.ib.ticker(contract)
        
        greeks = ticker.modelGreeks
        return {
            'symbol': contract.localSymbol,
            'price': ticker.last if not np.isnan(ticker.last) else (ticker.close if not np.isnan(ticker.close) else 0),
            'bid': ticker.bid if not np.isnan(ticker.bid) else 0,
            'ask': ticker.ask if not np.isnan(ticker.ask) else 0,
            'iv': greeks.impliedVol if greeks else 0,
            'delta': greeks.delta if greeks else 0,
            'gamma': greeks.gamma if greeks else 0,
            'vega': greeks.vega if greeks else 0,
            'theta': greeks.theta if greeks else 0,
            'updated': datetime.now().isoformat()
        }

ib_service = IBService()
