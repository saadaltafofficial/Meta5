# ICT Forex Trading Bot

A sophisticated algorithmic trading bot that implements the Inner Circle Trader (ICT) methodology for forex trading. It connects directly to MetaTrader 5 for real-time data and trade execution, with a focus on high-probability setups during specific market killzones.

## Features

- **ICT Methodology Implementation**: Order blocks, fair value gaps, breaker blocks, and liquidity pools
- **Multi-Timeframe Analysis**: Short-term and long-term trade setups with dynamic exit strategies
- **Precision Risk Management**: 1.5% account risk with multiple take profit levels (1.5R, 2.75R, 4.75R)
- **Killzone Trading**: Optimized for Pakistan nighttime trading hours and high-liquidity sessions
- **MetaTrader 5 Integration**: Direct connection for real-time data and trade execution
- **Telegram Reporting**: Concise performance reports every 12 hours
- **VPS Optimized**: Designed to run 24/7 on a Windows VPS with minimal storage usage

## Project Structure

```
/
├── config/            # Configuration files
│   ├── .env.example   # Example environment variables template
│   └── config.json    # Trading parameters and settings
├── data/              # Data storage
├── scripts/           # Batch scripts for startup
│   └── start_ict_bot.bat  # VPS startup script
├── src/               # Source code
│   ├── analysis/      # Performance analysis modules
│   ├── core/          # Core trading functionality
│   ├── ict/           # ICT methodology implementation
│   ├── reporting/     # Telegram reporting modules
│   └── utils/         # Utility functions and helpers
├── tests/             # Test scripts
└── main.py           # Main entry point
```

## Installation

### Prerequisites

- Python 3.8 or higher
- MetaTrader 5 installed on your system
- Telegram bot (for reporting)

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ict-forex-bot.git
   cd ict-forex-bot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create your environment file:
   ```
   cp config/.env.example config/.env
   ```

4. Edit the `config/.env` file with your credentials:
   - MT5 server, login, and password
   - Telegram bot token and chat ID
   - OpenAI API key (optional for advanced reports)

## Usage

### Running the Bot

```
python main.py
```

### VPS Deployment

For 24/7 operation on a Windows VPS:

```
scripts/start_ict_bot.bat
```

This script will start both MT5 and the trading bot, and automatically restart the bot if it crashes.

## ICT Trading Methodology

This bot implements the Inner Circle Trader (ICT) methodology from the 2022/2024 mentorship, including:

1. **Order Blocks**: Last red candle before a strong bullish move (bullish OB) or last green candle before a strong bearish move (bearish OB)

2. **Fair Value Gaps (FVG)**: When low price > previous high price (bullish FVG) or high price < previous low price (bearish FVG)

3. **Market Structure**: Identifying market structure shifts (MSS), breaker blocks, and liquidity pools

4. **Killzones**: Trading during specific high-probability time windows, optimized for Pakistan time:
   - London-NY Overlap (7-12pm PKT)
   - Super Prime Hours (7-9pm PKT)
   - Asian Session (1-5am PKT)

5. **Risk Management**:
   - Risk percentage: 1.5% of account per trade
   - Take profit levels at multiple R:R ratios (1.5, 2.75, 4.75)
   - Dynamic position sizing based on account balance and volatility

## Telegram Reporting

The bot sends concise performance reports to your Telegram chat every 12 hours, including:

- Account balance and equity with percentage change
- Open positions with profit/loss indicators and R-multiple values
- Closed positions from the last 24 hours
- Current trading session information (e.g., Super Prime, London-NY Overlap)

To test the Telegram reporting:
```
python -m src.reporting.telegram_reporter --test
```

## Maintenance

To truncate the balance history file if it grows too large:
```
python -m src.utils.truncate_balance_history
```

## License

MIT

## Disclaimer

Trading forex carries a high level of risk and may not be suitable for all investors. Before deciding to trade foreign exchange, you should carefully consider your investment objectives, level of experience, and risk appetite. The possibility exists that you could sustain a loss of some or all of your initial investment and therefore you should not invest money that you cannot afford to lose.
