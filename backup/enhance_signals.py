#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhance signal generation and confidence levels
"""

import logging
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

# Initialize MT5 connection
bot._initialize_mt5()

# Update currency pairs to only include EURUSD and GBPUSD
bot.currency_pairs = ['EURUSD', 'GBPUSD']

# Print current configuration
print(f"\n==== CURRENT CONFIGURATION ====\n")
print(f"Minimum confidence threshold: {bot.min_confidence}")
print(f"Risk percentage: {bot.risk_percent}%")
print(f"Currency pairs: {', '.join(bot.currency_pairs)}")

# Enhance ICT model confidence
print("\n==== ENHANCING ICT MODEL CONFIDENCE ====\n")
bot.ict_model.fair_value_gap_threshold = 0.001  # More sensitive to fair value gaps
bot.ict_model.liquidity_threshold = 0.0015  # More sensitive to liquidity levels
print("ICT model parameters adjusted for higher confidence signals")

# Lower the confidence threshold temporarily
original_threshold = bot.min_confidence
bot.min_confidence = 0.4  # Lower threshold to see more signals
print(f"Lowered min_confidence threshold to: {bot.min_confidence}")

# Run analysis for each pair
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
    
    # Print if signal would be executed
    print(f"  Would execute: {'Yes' if signals.get('confidence', 0) >= original_threshold else 'No (below original threshold)'}")

# Update the .env file with the new confidence threshold if desired
update_env = input("\nWould you like to permanently update the minimum confidence threshold to 0.4? (y/n): ")
if update_env.lower() == 'y':
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    with open(env_path, 'r') as file:
        env_content = file.read()
    
    # Replace the MIN_CONFIDENCE value
    if 'MIN_CONFIDENCE=' in env_content:
        env_content = env_content.replace(
            f"MIN_CONFIDENCE={original_threshold}", 
            f"MIN_CONFIDENCE=0.4"
        )
    else:
        env_content += f"\nMIN_CONFIDENCE=0.4"
    
    # Write the updated content back to the .env file
    with open(env_path, 'w') as file:
        file.write(env_content)
    
    print("The .env file has been updated with the new confidence threshold.")
    print("Please restart the trading bot for the changes to take effect.")
else:
    # Restore original threshold
    bot.min_confidence = original_threshold
    print(f"\nRestored min_confidence threshold to: {bot.min_confidence}")

# Run the terminal trader with the enhanced signals
run_terminal = input("\nWould you like to run the terminal trader with the enhanced signals? (y/n): ")
if run_terminal.lower() == 'y':
    print("\nStarting terminal trader with enhanced signals...")
    print("Press Ctrl+C to exit")
    
    # Import and run the terminal trader
    import subprocess
    subprocess.run(["python", "terminal_trader.py"])
else:
    print("\nExiting without running terminal trader.")
