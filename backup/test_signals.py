#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to check if trading signals are being generated properly
"""

import logging
from main import ForexTradingBot

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Create the trading bot instance
bot = ForexTradingBot()

# Initialize MT5 connection
bot._initialize_mt5()

# Print current configuration
print(f"\nCurrent min_confidence threshold: {bot.min_confidence}")

# Set different confidence thresholds to test
thresholds = [0.6, 0.4, 0.2]

for threshold in thresholds:
    print(f"\n\n==== Testing with confidence threshold: {threshold} ====\n")
    bot.min_confidence = threshold
    
    # Run the analysis for each currency pair
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
        
        # Print signal information
        print(f"  Action: {signals.get('action', 'UNKNOWN')}")
        print(f"  Confidence: {signals.get('confidence', 0):.2f}")
        print(f"  Reason: {signals.get('reason', 'No reason provided')}")
        print(f"  Would execute: {'Yes' if signals.get('confidence', 0) >= 0.6 else 'No (below original threshold)'}")

# Restore original threshold
bot.min_confidence = 0.6
print(f"\nRestored min_confidence threshold to: {bot.min_confidence}")
