#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Confidence Boost Module

This script tests the confidence boost module with sample data.
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

# Import the confidence boost module
from src.ict.confidence_boost import boost_confidence

def test_confidence_boost():
    """
    Test the confidence boost module with sample data.
    """
    # Print a clean header
    print("\n" + "=" * 80)
    print(f"CONFIDENCE BOOST TEST - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
    
    # Test both BUY and SELL actions
    for action in ['BUY', 'SELL']:
        # Set base confidence
        base_confidence = 0.15
        
        # Use our confidence boost module
        final_confidence, confluence_factors, additional_confidence = boost_confidence(data, action, base_confidence)
        
        # Print the results
        print(f"\nAction: {action}")
        print(f"Base confidence: {base_confidence:.4f}")
        print(f"Additional confidence: {additional_confidence:.4f}")
        print(f"Final confidence: {final_confidence:.4f}")
        print(f"Confluence factors: {confluence_factors}")
        print("\n" + "-" * 40)

if __name__ == "__main__":
    test_confidence_boost()
