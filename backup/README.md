# MCP Forex Trading Bot

A sophisticated algorithmic trading bot that analyzes forex currency pairs using technical indicators and institutional trading methods. It connects directly to MetaTrader 5 for real-time data and trade execution.

## Features

- MetaTrader 5 (MT5) integration for real-time trading
- Advanced technical analysis with ICT (Inner Circle Trader) methodology
- REST API for monitoring and controlling the trading system
- Real-time market status monitoring for global forex centers
- MongoDB integration for trade tracking and performance metrics
- Optional Telegram integration for signal delivery
- Optional news analysis using OpenAI API

## Setup Instructions

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure your credentials in the `.env` file:
   ```
   MT5_SERVER=MetaQuotes-Demo
   MT5_LOGIN=5035363260
   MT5_PASSWORD=_xQyH8Kz
   MT5_AUTO_TRADING=true
   RISK_PERCENT=1.0
   MIN_CONFIDENCE=0.6
   MONGODB_URI=mongodb+srv://your_mongodb_connection_string
   ```

3. Run the API server:
   ```
   python api_server.py
   ```

## API Endpoints

- `/api/market-status` - Check if forex market is open
- `/api/global-markets` - Status of global trading centers
- `/api/signals` - Get current trading signals
- `/api/pairs` - List monitored currency pairs
- `/api/add-pair` - Add currency pair to monitor
- `/api/remove-pair` - Remove currency pair from monitoring
- `/api/trading-status` - MT5 connection and account status
- `/api/execute-trade` - Manually execute a trade
- `/api/close-all-trades` - Close all active trades
- `/api/trade-history` - View trade history
- `/api/performance` - View performance metrics

## Technical Analysis Methods

- Inner Circle Trader (ICT) methodology
- Moving Average Convergence Divergence (MACD)
- Relative Strength Index (RSI)
- Bollinger Bands
- Moving Averages (SMA, EMA)
- Stochastic Oscillator

## Core Components

- `main.py`: Core trading bot implementation
- `api_server.py`: FastAPI server for web interface
- `mt5_trader.py`: MetaTrader 5 integration
- `technical_analysis.py`: Technical indicators and analysis
- `ict_model.py`: ICT trading methodology implementation
- `db_manager.py`: MongoDB database operations
- `market_status.py`: Market open/close detection
- `global_markets.py`: Global trading centers monitoring

## License

MIT
