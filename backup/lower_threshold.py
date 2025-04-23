#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lower the confidence threshold to see more trading signals
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

# Lower the confidence threshold to see more signals
original_threshold = bot.min_confidence
bot.min_confidence = 0.3  # Lower threshold to see more signals
print(f"Lowered min_confidence threshold to: {bot.min_confidence}")

# Run the analysis loop once to generate signals
bot.running = True
bot._analysis_loop(single_run=True)  # Modified to run just once

# Print the generated signals
print("\nTrading signals with lower threshold:")
for pair, signal in bot.latest_signals.items():
    if signal:
        print(f"\n{pair}:")
        print(f"  Action: {signal.get('action', 'UNKNOWN')}")
        print(f"  Confidence: {signal.get('confidence', 0):.2f}")
        print(f"  Reason: {signal.get('reason', 'No reason provided')}")
        print(f"  Would execute: {'Yes' if signal.get('confidence', 0) >= original_threshold else 'No (below original threshold)'}")
    else:
        print(f"\n{pair}: No signal")

# Restore original threshold
bot.min_confidence = original_threshold
print(f"\nRestored min_confidence threshold to: {bot.min_confidence}")
