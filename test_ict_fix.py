#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test ICT Fix

This script tests the fixed ICT integration with confidence boost.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from src.ict.ict_mechanics import *
from src.ict.ict_integration import ICTAnalyzer
from src.ict.confidence_boost import boost_confidence

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ict_fix():
    """
    Test the ICT integration with confidence boost.
    """
    # Create an instance of ICTAnalyzer
    analyzer = ICTAnalyzer()
    
    # Monkey patch the generate_ict_signal method with our fixed version
    from src.ict.ict_integration_fix import generate_ict_signal_fixed
    analyzer.generate_ict_signal = generate_ict_signal_fixed.__get__(analyzer, ICTAnalyzer)
    
    # Create a sample DataFrame with OHLC data
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, 100),
        'high': np.random.normal(101, 1, 100),
        'low': np.random.normal(99, 1, 100),
        'close': np.random.normal(100, 1, 100),
        'volume': np.random.normal(1000, 100, 100)
    })
    
    # Ensure data is sorted by index
    data = data.sort_index()
    
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
    
    # Generate an ICT signal
    signal = analyzer.generate_ict_signal(data, pair="EURUSD")
    
    # Print the signal details
    logger.info(f"Action: {signal['action']}")
    logger.info(f"Confidence: {signal['confidence']}")
    logger.info(f"Reason: {signal['reason']}")
    logger.info(f"Base confidence: {signal['details']['base_confidence']}")
    logger.info(f"Additional confidence: {signal['details']['additional_confidence']}")
    logger.info(f"Confluence factors: {signal['details']['confluence_factors']}")
    
    return signal

if __name__ == "__main__":
    test_ict_fix()
