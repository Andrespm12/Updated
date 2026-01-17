from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
from web_app.ib_service import ib_service

app = FastAPI()
templates = Jinja2Templates(directory="web_app/templates")

# Models source
class OptionRequest(BaseModel):
    symbol: str
    expiry: str
    strike: float
    right: str

@app.on_event("startup")
async def startup_event():
    # Force ib_insync (which might call get_event_loop) to see the running loop
    loop = asyncio.get_running_loop()
    asyncio.set_event_loop(loop)
    
    # Attempt to connect on startup
    await ib_service.connect()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/connect")
async def connect():
    status = await ib_service.connect()
    return {"connected": status}

@app.get("/api/chain/{symbol}")
async def get_chain(symbol: str):
    data = await ib_service.search_options_chain(symbol.upper())
    if not data:
        return {"error": "Chain not found"}
    return data

@app.get("/api/contracts/{symbol}/{expiry}")
async def get_contracts(symbol: str, expiry: str):
    # Returns valid strikes for this specific expiry
    # This matches the logic we fixed in the CLI scripts
    try:
        details = await ib_service.get_details(symbol.upper(), expiry, 'C') # Check Calls
        strikes = sorted(list(set([d.contract.strike for d in details])))
        return {"strikes": strikes}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/greeks/snapshot")
async def get_greeks(req: OptionRequest):
    try:
        data = await ib_service.get_market_data_snapshot(
            req.symbol.upper(), req.expiry, req.strike, req.right
        )
        return data
    except Exception as e:
        return {"error": str(e)}
