#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug script to investigate signal generation with lower confidence threshold
"""

import logging
import time
from main import ForexTradingBot

# Configure detailed logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

# Create the trading bot instance
bot = ForexTradingBot()

# Initialize MT5 connection
bot._initialize_mt5()

# Print current configuration
print(f"\n==== CURRENT CONFIGURATION ====\n")
print(f"Minimum confidence threshold: {bot.min_confidence}")
print(f"Risk percentage: {bot.risk_percent}%")
print(f"Currency pairs: {', '.join(bot.currency_pairs)}")
print(f"MT5 auto trading: {'Enabled' if bot.auto_trading else 'Disabled'}")

# Lower the confidence threshold to see more signals
original_threshold = bot.min_confidence
bot.min_confidence = 0.3  # Lower threshold to see more signals
print(f"\nLowered min_confidence threshold to: {bot.min_confidence}")

def detailed_analysis(pair, data):
    """Perform detailed analysis on a currency pair and print results"""
    print(f"\n==== DETAILED ANALYSIS FOR {pair} ====\n")
    
    # Get technical analysis
    print("TECHNICAL ANALYSIS:")
    technical_analysis = bot.analyzer.analyze(data, pair)
    print(f"  Action: {technical_analysis.get('action', 'UNKNOWN')}")
    print(f"  Confidence: {technical_analysis.get('confidence', 0):.2f}")
    print(f"  Reason: {technical_analysis.get('reason', 'No reason provided')}")
    
    # Print technical indicators
    indicators = technical_analysis.get('indicators', {})
    print("\nTECHNICAL INDICATORS:")
    for indicator_name, values in indicators.items():
        if indicator_name != 'ict':
            print(f"  {indicator_name.upper()}:")
            if isinstance(values, dict):
                for k, v in values.items():
                    print(f"    {k}: {v}")
            else:
                print(f"    {values}")
    
    # Get ICT model analysis
    print("\nICT MODEL ANALYSIS:")
    ict_analysis = bot.ict_model.analyze(data, pair)
    print(f"  Confidence: {ict_analysis.get('confidence', 0):.2f}")
    
    # Print market structure
    market_structure = ict_analysis.get('market_structure', {})
    print("\n  MARKET STRUCTURE:")
    for k, v in market_structure.items():
        print(f"    {k}: {v}")
    
    # Print key levels
    key_levels = ict_analysis.get('key_levels', {})
    print("\n  KEY LEVELS:")
    for level_type, levels in key_levels.items():
        print(f"    {level_type}:")
        if isinstance(levels, list) and levels:
            print(f"      {len(levels)} items found")
            if len(levels) > 0 and isinstance(levels[0], dict):
                for i, level in enumerate(levels[:3]):  # Show first 3 only
                    print(f"      Item {i+1}: {level}")
        elif isinstance(levels, dict):
            for k, v in levels.items():
                print(f"      {k}: {v}")
    
    # Get news
    print("\nNEWS ANALYSIS:")
    news = None
    if bot.news_analyzer:
        news = bot.news_analyzer.get_forex_news(pair)
        if news:
            print(f"  {len(news)} news items found")
            for i, item in enumerate(news[:3]):  # Show first 3 only
                print(f"  Item {i+1}:")
                print(f"    Title: {item.get('title', 'Unknown')}")
                print(f"    Impact: {item.get('impact', 'Unknown')}")
                print(f"    Direction: {item.get('impact_direction', 'Unknown')}")
        else:
            print("  No news available")
    else:
        print("  News analyzer not enabled")
    
    # Generate combined signals
    print("\nCOMBINED SIGNAL:")
    signals = bot._generate_signals(pair, data, technical_analysis, ict_analysis, news)
    print(f"  Action: {signals.get('action', 'UNKNOWN')}")
    print(f"  Confidence: {signals.get('confidence', 0):.2f}")
    print(f"  Reason: {signals.get('reason', 'No reason provided')}")
    print(f"  Would execute: {'Yes' if signals.get('confidence', 0) >= original_threshold else 'No (below original threshold)'}")
    
    return signals

# Run detailed analysis for each pair
print("\n==== RUNNING DETAILED ANALYSIS ====\n")
for pair in bot.currency_pairs:
    try:
        # Get forex data
        data = bot.data_provider.get_forex_data(pair, interval='1h')
        if data is None or data.empty:
            print(f"No data available for {pair}")
            continue
            
        # Perform detailed analysis
        signals = detailed_analysis(pair, data)
        
        # Store the signals
        bot.latest_signals[pair] = signals
        
    except Exception as e:
        logger.error(f"Error analyzing {pair}: {e}", exc_info=True)

# Restore original threshold
bot.min_confidence = original_threshold
print(f"\nRestored min_confidence threshold to: {bot.min_confidence}")

# Run the terminal trader to display the signals
print("\n==== RUNNING TERMINAL TRADER WITH GENERATED SIGNALS ====\n")
print("Press Ctrl+C to exit")

try:
    # Start the bot with our pre-generated signals
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
        bot.global_markets.print_status()
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
        import datetime
        print(f"🔍 TRADING SIGNALS - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        if not bot.latest_signals:
            print("No trading signals available.")
        else:
            for pair, signal in bot.latest_signals.items():
                if signal:
                    action = signal.get('action', 'HOLD')
                    confidence = signal.get('confidence', 0)
                    reason = signal.get('reason', 'No reason provided')
                    
                    # Color coding based on action
                    if action == 'BUY':
                        action_colored = "\033[92mBUY\033[0m"  # Green
                    elif action == 'SELL':
                        action_colored = "\033[91mSELL\033[0m"  # Red
                    else:
                        action_colored = "\033[93mHOLD\033[0m"  # Yellow
                    
                    # Display signal
                    print(f"{pair}: {action_colored} (Confidence: {confidence:.2f})")
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
        
        # Sleep for a few seconds
        time.sleep(5)
        
except KeyboardInterrupt:
    print("\nExiting terminal trader...")
except Exception as e:
    logger.error(f"Error in terminal trader: {e}", exc_info=True)
