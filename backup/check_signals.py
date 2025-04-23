#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check current trading signals and their confidence levels
"""

from main import ForexTradingBot

# Create the trading bot instance
bot = ForexTradingBot()

# Initialize MT5 connection
bot._initialize_mt5()

# Print current configuration
print(f"\nCurrent min_confidence threshold: {bot.min_confidence}")

# Perform analysis for each currency pair
for pair in bot.currency_pairs:
    print(f"\nAnalyzing {pair}...")
    
    # Get data for the pair
    if hasattr(bot.data_provider, 'get_data'):
        data = bot.data_provider.get_data(pair)
    else:
        # ForexDataProviderMT5 uses get_forex_data instead
        data = bot.data_provider.get_forex_data(pair)
    
    # Perform technical analysis
    technical_analysis = bot.analyzer.analyze(pair, data)
    
    # Perform ICT model analysis
    ict_analysis = bot.ict_model.analyze(pair, data)
    
    # Generate signals
    signals = bot._generate_signals(pair, data, technical_analysis, ict_analysis)
    
    # Print the signals
    print(f"Action: {signals.get('action', 'UNKNOWN')}")
    print(f"Confidence: {signals.get('confidence', 0):.2f}")
    print(f"Reason: {signals.get('reason', 'No reason provided')}")

# Print latest signals stored in the bot
print("\nLatest signals stored in the bot:")
for pair, signal in bot.latest_signals.items():
    if signal:
        print(f"{pair}: {signal.get('action', 'UNKNOWN')} with confidence {signal.get('confidence', 0):.2f}")
    else:
        print(f"{pair}: No signal")
