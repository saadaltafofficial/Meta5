#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Focused trading script for EURUSD and GBPUSD with enhanced signal generation
"""

import logging
import time
import os
from main import ForexTradingBot
from datetime import datetime

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Create the trading bot instance
bot = ForexTradingBot()

# Focus only on EURUSD and GBPUSD
bot.currency_pairs = ['EURUSD', 'GBPUSD']

# Initialize MT5 connection
bot._initialize_mt5()

# Enable auto-trading
bot.auto_trading = True

# Lower the confidence threshold
bot.min_confidence = 0.4

# Enhance ICT model parameters for better signal detection
bot.ict_model.fair_value_gap_threshold = 0.001  # More sensitive to fair value gaps
bot.ict_model.liquidity_threshold = 0.0015  # More sensitive to liquidity levels

# Function to analyze and generate signals
def analyze_pairs():
    print("\n==== ANALYZING CURRENCY PAIRS ====\n")
    for pair in bot.currency_pairs:
        print(f"\nAnalyzing {pair}...")
        
        # Get forex data
        data = bot.data_provider.get_forex_data(pair, interval='1h')
        if data is None or data.empty:
            print(f"  No data available for {pair}")
            continue
            
        # Perform technical analysis
        technical_analysis = bot.analyzer.analyze(data, pair)
        
        # Perform ICT model analysis
        ict_analysis = bot.ict_model.analyze(data, pair)
        
        # Get news for the pair
        news = None
        if bot.news_analyzer:
            news = bot.news_analyzer.get_forex_news(pair)
        
        # Generate signals
        signals = bot._generate_signals(pair, data, technical_analysis, ict_analysis, news)
        
        # Store the signals
        bot.latest_signals[pair] = signals
        
        # Print signal information
        print(f"  Action: {signals.get('action', 'UNKNOWN')}")
        print(f"  Confidence: {signals.get('confidence', 0):.2f}")
        print(f"  Reason: {signals.get('reason', 'No reason provided')}")
        
        # Print technical analysis details
        tech_action = technical_analysis.get('action', 'HOLD')
        tech_confidence = technical_analysis.get('confidence', 0)
        print(f"  Technical Analysis: {tech_action} (Confidence: {tech_confidence:.2f})")
        
        # Print ICT analysis details
        ict_confidence = ict_analysis.get('confidence', 0)
        market_structure = ict_analysis.get('market_structure', {}).get('trend', 'UNKNOWN')
        print(f"  ICT Analysis: Confidence {ict_confidence:.2f}, Market Structure: {market_structure}")
        
        # Execute trade if auto-trading is enabled and signal confidence is high enough
        if bot.auto_trading and signals:
            confidence = signals.get('confidence', 0)
            action = signals.get('action', 'HOLD')
            
            if confidence >= bot.min_confidence and action in ['BUY', 'SELL']:
                print(f"  Executing {action} trade for {pair} with confidence {confidence:.2f}...")
                trade_result = bot._execute_trade(pair, signals)
                if trade_result:
                    print(f"  Trade executed successfully!")
                else:
                    print(f"  Failed to execute trade.")
            else:
                print(f"  No trade executed: {'Low confidence' if confidence < bot.min_confidence else 'No actionable signal'}")


# Analyze pairs once
analyze_pairs()

# Start the terminal trader
print("\n==== STARTING TERMINAL TRADER ====\n")
print("Press Ctrl+C to exit")

try:
    # Start the bot
    bot.running = True
    
    # Display the terminal interface
    while True:
        # Clear screen (Windows)
        print("\033[H\033[J", end="")
        
        # Display market status
        is_market_open = bot.market_status.is_market_open()
        market_status = "OPEN 🟢" if is_market_open else "CLOSED 🔴"
        print("="*80)
        print(f"📊 FOREX MARKET STATUS: {market_status}")
        print("="*80)
        
        # Display global markets status
        print("🌎 GLOBAL FOREX MARKETS")
        print("="*80)
        # Custom code to display global markets status
        print("🌎 *GLOBAL FOREX MARKETS STATUS*\n")
        
        # Check status of major trading centers
        sydney_open = bot.global_markets.is_center_open('Sydney')
        tokyo_open = bot.global_markets.is_center_open('Tokyo')
        london_open = bot.global_markets.is_center_open('London')
        newyork_open = bot.global_markets.is_center_open('New York')
        
        # Get local times for each center
        sydney_time = bot.global_markets.get_center_local_time('Sydney')
        tokyo_time = bot.global_markets.get_center_local_time('Tokyo')
        london_time = bot.global_markets.get_center_local_time('London')
        newyork_time = bot.global_markets.get_center_local_time('New York')
        
        # Display status with emoji indicators
        print(f"{'✅' if sydney_open else '❌'} *Sydney*: {'OPEN' if sydney_open else 'CLOSED'} (Local time: {sydney_time.strftime('%H:%M %Z')})")
        print(f"{'✅' if tokyo_open else '❌'} *Tokyo*: {'OPEN' if tokyo_open else 'CLOSED'} (Local time: {tokyo_time.strftime('%H:%M %Z')})")
        print(f"{'✅' if london_open else '❌'} *London*: {'OPEN' if london_open else 'CLOSED'} (Local time: {london_time.strftime('%H:%M %Z')})")
        print(f"{'✅' if newyork_open else '❌'} *New York*: {'OPEN' if newyork_open else 'CLOSED'} (Local time: {newyork_time.strftime('%H:%M %Z')})")
        
        # Overall status
        if sydney_open or tokyo_open or london_open or newyork_open:
            print("\n🟢 At least one major market is open. Forex trading is available.")
        else:
            print("\n🔴 All major markets are closed. Limited forex trading available.")
        print()
        
        # Display MT5 account info
        if bot.mt5_enabled and bot.mt5_trader:
            account_info = bot.mt5_trader.get_account_info()
            print("="*80)
            print("✅ MT5 TRADING: CONNECTED")
            print("="*80)
            print(f"Server: {bot.mt5_server}")
            print(f"Login: {bot.mt5_login}")
            print(f"Name: {account_info.get('name', 'Unknown')}")
            print(f"Balance: ${account_info.get('balance', 0):.2f}")
            print(f"Equity: ${account_info.get('equity', 0):.2f}")
            print(f"Profit: ${account_info.get('profit', 0):.2f}")
            print(f"Auto Trading: {'Enabled' if bot.auto_trading else 'Disabled'}")
        else:
            print("="*80)
            print("❌ MT5 TRADING: NOT CONNECTED")
            print("="*80)
        
        # Display monitored currency pairs
        print("="*80)
        print("💱 MONITORED CURRENCY PAIRS")
        print("="*80)
        for pair in bot.currency_pairs:
            print(f"- {pair}")

        
        # Display trading signals
        print("="*80)
        print(f"🔍 TRADING SIGNALS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        if not bot.latest_signals:
            print("No trading signals available.")
        else:
            for pair, signal in bot.latest_signals.items():
                if signal:
                    action = signal.get('action', 'HOLD')
                    confidence = signal.get('confidence', 0)
                    reason = signal.get('reason', 'No reason provided')
                    
                    # Display signal without color coding (for compatibility)
                    print(f"{pair}: {action} (Confidence: {confidence:.2f})")
                    print(f"  Reason: {reason}")
                    print(f"  Execute: {'Yes' if confidence >= bot.min_confidence else 'No (below threshold)'}")
                    print()
        
        # Display active trades
        print("="*80)
        print("📈 ACTIVE TRADES")
        print("="*80)
        
        active_trades = bot.get_active_trades()
        if not active_trades:
            print("No active trades.")
        else:
            for pair, trade in active_trades.items():
                print(f"{pair}: {trade['type']} @ {trade['open_price']}")
                print(f"  Open Time: {trade['open_time']}")
                print(f"  Lot Size: {trade['lot_size']}")
                print(f"  Stop Loss: {trade['stop_loss']}")
                print(f"  Take Profit: {trade['take_profit']}")
                print()
        
        # Re-analyze pairs every 5 minutes
        if datetime.now().minute % 5 == 0 and datetime.now().second < 10:
            print("\nRe-analyzing currency pairs...")
            analyze_pairs()
        
        # Sleep for a few seconds
        time.sleep(5)
        
except KeyboardInterrupt:
    print("\nExiting terminal trader...")
except Exception as e:
    logger.error(f"Error in terminal trader: {e}", exc_info=True)
