#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FastAPI server for the MCP Forex Trading Bot"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Import custom modules
from main import ForexTradingBot

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize the trading bot
trading_bot = ForexTradingBot()

# Initialize FastAPI app
app = FastAPI(
    title="MCP Forex Trading Bot API",
    description="API for the MCP Forex Trading Bot",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for API access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define API models
class SignalResponse(BaseModel):
    pair: str
    action: str
    confidence: float
    reason: str
    timestamp: datetime

class TradeResponse(BaseModel):
    ticket: int
    symbol: str
    type: str
    volume: float
    open_price: float
    current_price: float
    profit: float
    open_time: str

class PerformanceResponse(BaseModel):
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_profit: float
    average_profit: float
    average_loss: float
    largest_win: float
    largest_loss: float

# API routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api")
async def api_root():
    return {"message": "MCP Forex Trading Bot API"}

@app.get("/api/market-status")
async def get_market_status():
    """Get the current forex market status"""
    is_open = trading_bot.market_status.is_market_open()
    market_hours_text = trading_bot.market_status.get_market_hours_text()
    
    return {
        "is_open": is_open,
        "market_hours_text": market_hours_text
    }

@app.get("/api/global-markets")
async def get_global_markets():
    """Get the status of global forex markets"""
    centers_status_text = trading_bot.global_markets.get_centers_status_text()
    all_centers_status = trading_bot.global_markets.get_all_centers_status()
    
    # Format the centers data for the frontend
    centers_status = {}
    for center_name, is_open in all_centers_status.items():
        # Get local time for the center
        local_time = trading_bot.global_markets.get_center_local_time(center_name)
        
        centers_status[center_name] = {
            "is_open": is_open,
            "local_time": local_time.strftime("%Y-%m-%d %H:%M:%S") if local_time else None
        }
    
    return {
        "centers_status_text": centers_status_text,
        "centers": centers_status
    }

@app.get("/api/signals")
async def get_signals():
    """Get the latest trading signals"""
    signals = []
    
    for pair, signal in trading_bot.latest_signals.items():
        if signal:
            signals.append({
                "pair": pair,
                "action": signal.get("action", "HOLD"),
                "confidence": signal.get("confidence", 0),
                "reason": signal.get("reason", ""),
                "timestamp": signal.get("timestamp", datetime.now()).isoformat()
            })
    
    return {"signals": signals}

@app.get("/api/pairs")
async def get_pairs():
    """Get the currency pairs being monitored"""
    return {"pairs": trading_bot.currency_pairs}

@app.post("/api/add-pair")
async def add_pair(pair: str):
    """Add a currency pair to monitor"""
    if not pair or len(pair) != 6:
        raise HTTPException(status_code=400, detail="Invalid currency pair format. Must be 6 characters (e.g., EURUSD)")
    
    # Convert to uppercase
    pair = pair.upper()
    
    # Check if already monitoring this pair
    if pair in trading_bot.currency_pairs:
        return {"success": False, "message": f"Already monitoring {pair}"}
    
    # Add the pair
    trading_bot.currency_pairs.append(pair)
    
    return {"success": True, "message": f"Added {pair} to monitored pairs", "pairs": trading_bot.currency_pairs}

@app.post("/api/remove-pair")
async def remove_pair(pair: str):
    """Remove a currency pair from monitoring"""
    if not pair or len(pair) != 6:
        raise HTTPException(status_code=400, detail="Invalid currency pair format. Must be 6 characters (e.g., EURUSD)")
    
    # Convert to uppercase
    pair = pair.upper()
    
    # Check if monitoring this pair
    if pair not in trading_bot.currency_pairs:
        return {"success": False, "message": f"Not monitoring {pair}"}
    
    # Remove the pair
    trading_bot.currency_pairs.remove(pair)
    
    return {"success": True, "message": f"Removed {pair} from monitored pairs", "pairs": trading_bot.currency_pairs}

@app.get("/api/trading-status")
async def get_trading_status():
    """Get the MT5 trading status"""
    response = {
        "mt5_enabled": trading_bot.mt5_enabled,
        "auto_trading": trading_bot.auto_trading,
        "account_info": None,
        "active_trades": {}
    }
    
    # Get account info if MT5 is enabled
    if trading_bot.mt5_enabled:
        account_info = trading_bot.get_mt5_account_info()
        if account_info:
            response["account_info"] = {
                "name": account_info.get("name", "Unknown"),
                "login": account_info.get("login", "Unknown"),
                "server": account_info.get("server", "Unknown"),
                "currency": account_info.get("currency", "USD"),
                "balance": round(account_info.get("balance", 0), 2),
                "equity": round(account_info.get("equity", 0), 2),
                "margin": round(account_info.get("margin", 0), 2),
                "free_margin": round(account_info.get("margin_free", 0), 2),
                "leverage": account_info.get("leverage", 1)
            }
        
        # Get active trades
        active_trades = trading_bot.get_active_trades()
        
        # Format active trades for display
        formatted_trades = {}
        for pair, trade_info in active_trades.items():
            # Convert datetime to string for JSON serialization
            trade_data = trade_info.copy()
            if "open_time" in trade_data and isinstance(trade_data["open_time"], datetime):
                trade_data["open_time"] = trade_data["open_time"].strftime("%Y-%m-%d %H:%M:%S")
            
            # Remove signals to avoid circular references
            if "signals" in trade_data:
                del trade_data["signals"]
                
            formatted_trades[pair] = trade_data
            
        response["active_trades"] = formatted_trades
    
    return response

@app.post("/api/execute-trade")
async def execute_trade(pair: str, action: str, lot_size: float):
    """Manually execute a trade"""
    if not trading_bot.mt5_enabled:
        return {"success": False, "message": "MT5 trading is not enabled."}
    
    if action not in ["BUY", "SELL"]:
        return {"success": False, "message": "Invalid action. Must be BUY or SELL."}
    
    if lot_size <= 0:
        return {"success": False, "message": "Invalid lot size. Must be greater than 0."}
    
    # Create a simple signal for the trade
    signals = {
        "pair": pair,
        "action": action,
        "confidence": 1.0,  # Manual trade has 100% confidence
        "timestamp": datetime.now(),
        "reason": "Manual trade execution"
    }
    
    # Execute the trade
    if hasattr(trading_bot, "_execute_trade"):
        # Use the trading bot's execute trade method
        result = trading_bot._execute_trade(pair, signals)
        
        if result:
            return {"success": True, "message": f"{action} trade for {pair} executed successfully."}
        else:
            return {"success": False, "message": f"Failed to execute {action} trade for {pair}."}
    else:
        return {"success": False, "message": "Trading functionality not available."}

@app.post("/api/close-all-trades")
async def close_all_trades():
    """Close all active trades"""
    if not trading_bot.mt5_enabled:
        return {"success": False, "message": "MT5 trading is not enabled."}
    
    # Close all trades
    if hasattr(trading_bot, "close_all_trades"):
        closed_count = trading_bot.close_all_trades()
        
        return {"success": True, "message": f"Closed {closed_count} trades successfully."}
    else:
        return {"success": False, "message": "Trading functionality not available."}

@app.get("/api/trade-history")
async def get_trade_history(limit: int = 50):
    """Get trade history"""
    response = {"trades": []}
    
    # Check if database manager is available
    if hasattr(trading_bot, "db_manager") and trading_bot.db_manager:
        # Get trade history from database
        trades = trading_bot.db_manager.get_trade_history(limit=limit)
        response["trades"] = trades
    
    return response

@app.get("/api/performance")
async def get_performance():
    """Get performance metrics"""
    response = {"performance": {}}
    
    # Check if database manager is available
    if hasattr(trading_bot, "db_manager") and trading_bot.db_manager:
        # Get performance summary from database
        performance = trading_bot.db_manager.get_performance_summary()
        response["performance"] = performance
    
    return response

# Start the trading bot when the API server starts
@app.on_event("startup")
async def startup_event():
    """Start the trading bot when the API server starts"""
    # Initialize MT5 if credentials are available
    if trading_bot.mt5_server or os.getenv("MT5_CREATE_DEMO", "false").lower() == "true":
        trading_bot._initialize_mt5()
    
    # Start the trading bot in a separate thread
    import threading
    trading_thread = threading.Thread(target=trading_bot._analysis_loop, name="TradingThread")
    trading_thread.daemon = True
    trading_thread.start()
    
    logger.info("Trading bot started in background thread")

# Shutdown the trading bot when the API server stops
@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown the trading bot when the API server stops"""
    trading_bot.stop()
    logger.info("Trading bot stopped")

# Run the server if executed directly
if __name__ == "__main__":
    import uvicorn
    trading_bot.running = True
    uvicorn.run(app, host="0.0.0.0", port=8070)
