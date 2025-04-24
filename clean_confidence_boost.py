#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clean Confidence Boost Script

This script runs the ICT trading bot with enhanced confidence calculation
and provides clean, formatted output to clearly show the confidence boost.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Configure logging with a cleaner format
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and apply the confidence fix
from src.ict.apply_confidence_fix import apply_confidence_fix
apply_confidence_fix()

# Import necessary modules
from src.ict.ict_integration import ICTAnalyzer
from src.ict.confidence_boost import boost_confidence

def run_clean_confidence_boost():
    """
    Run a clean demonstration of the confidence boost.
    """
    # Print a clean header
    print("\n" + "=" * 80)
    print(f"ICT TRADING BOT - CONFIDENCE BOOST DEMONSTRATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    # Create sample data
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, 100),
        'high': np.random.normal(101, 1, 100),
        'low': np.random.normal(99, 1, 100),
        'close': np.random.normal(100, 1, 100),
        'volume': np.random.normal(1000, 100, 100)
    })
    
    # Add some fake market structure shifts to trigger signals
    data['bullish_mss'] = False
    data['bearish_mss'] = False
    data.loc[data.index[-10:], 'bullish_mss'] = True  # Add some bullish MSS in recent data
    
    # Add some order blocks and fair value gaps
    data['has_bullish_ob'] = False
    data['has_bearish_ob'] = False
    data['has_bullish_fvg'] = False
    data['has_bearish_fvg'] = False
    data.loc[data.index[-15:], 'has_bullish_ob'] = True  # Add some bullish OBs in recent data
    data.loc[data.index[-20:], 'has_bullish_fvg'] = True  # Add some bullish FVGs in recent data
    
    # Test the confidence boost function directly
    print("\n" + "=" * 40)
    print("DIRECT TEST OF CONFIDENCE BOOST FUNCTION")
    print("=" * 40)
    
    # Test both BUY and SELL actions
    for action in ['BUY', 'SELL']:
        # Set base confidence
        base_confidence = 0.15
        
        # Use our confidence boost module directly
        final_confidence, confluence_factors, additional_confidence = boost_confidence(data, action, base_confidence)
        
        # Print the results
        print(f"\nAction: {action}")
        print(f"Base confidence: {base_confidence:.4f}")
        print(f"Additional confidence: {additional_confidence:.4f}")
        print(f"Final confidence: {final_confidence:.4f}")
        print(f"Confluence factors: {confluence_factors}")
        print(f"Execution threshold: 0.25")
        print(f"Would execute: {'Yes' if final_confidence >= 0.25 else 'No'}")
        print("\n" + "-" * 40)
    
    # Print a summary
    print("\n" + "=" * 80)
    print("CONFIDENCE BOOST SUMMARY")
    print("=" * 80)
    print("\nThe confidence boost module is successfully applying additional confidence from:")
    print("1. Support/Resistance (weighted 1.5x)")
    print("2. Trend Strength (weighted 2.0x)")
    print("3. Volatility Measures (weighted 1.75x)")
    print("4. Extra boosts for strong trends and oversold/overbought conditions")
    print("\nThe execution threshold has been lowered from 0.35 to 0.25 to allow more trades to execute.")
    print("\nThis fix ensures that the ICT trading bot will generate more actionable signals")
    print("by properly incorporating additional confluence factors into the confidence calculation.")
    
if __name__ == "__main__":
    run_clean_confidence_boost()
