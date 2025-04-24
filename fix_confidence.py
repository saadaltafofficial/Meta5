#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Confidence Boost Fix Script

This script demonstrates how to fix the confidence calculation in the ICT trading bot.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from src.ict.ict_mechanics import *
from src.ict.support_resistance import detect_support_resistance, calculate_sr_confidence
from src.ict.trend_indicators import calculate_adx, calculate_ma_trend, calculate_trend_confidence
from src.ict.volatility_measures import calculate_atr, calculate_bollinger_bands, calculate_volatility_confidence
from src.ict.confidence_boost import boost_confidence

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_confidence_calculation():
    """
    Demonstrate how to fix the confidence calculation in the ICT trading bot.
    """
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
    
    # Calculate support and resistance levels
    sr_levels = detect_support_resistance(data, lookback=50, strength_threshold=2)
    
    # Calculate trend indicators
    data = calculate_adx(data, period=14)
    data = calculate_ma_trend(data, fast_period=9, slow_period=21, trend_period=50)
    trend_conf = calculate_trend_confidence(data, lookback=20)
    
    # Calculate volatility measures
    data = calculate_atr(data, period=14)
    data = calculate_bollinger_bands(data, period=20, std_dev=2)
    vol_conf = calculate_volatility_confidence(data, lookback=20)
    
    # Calculate SR confidence based on current price
    current_price = data['close'].iloc[-1]
    sr_conf = calculate_sr_confidence(current_price, sr_levels)
    
    # Create additional factors dictionary
    additional_factors = {
        'support_resistance': {
            'levels': sr_levels,
            'confidence': sr_conf
        },
        'trend': {
            'adx': data['adx'].iloc[-1] if 'adx' in data.columns else None,
            'plus_di': data['+di'].iloc[-1] if '+di' in data.columns else None,
            'minus_di': data['-di'].iloc[-1] if '-di' in data.columns else None,
            'buy_confidence': trend_conf['buy_confidence'],
            'sell_confidence': trend_conf['sell_confidence']
        },
        'volatility': {
            'atr': data['atr'].iloc[-1] if 'atr' in data.columns else None,
            'atr_percent': data['atr_percent'].iloc[-1] if 'atr_percent' in data.columns else None,
            'bb_bandwidth': data['bb_bandwidth'].iloc[-1] if 'bb_bandwidth' in data.columns else None,
            'bb_percent_b': data['bb_percent_b'].iloc[-1] if 'bb_percent_b' in data.columns else None,
            'buy_confidence': vol_conf['buy_confidence'],
            'sell_confidence': vol_conf['sell_confidence']
        }
    }
    
    # Log the additional factors
    logger.info(f"Additional factors calculated: SR={additional_factors['support_resistance'].get('confidence', {})}, "
              f"Trend Buy={additional_factors['trend'].get('buy_confidence', 0)}, Trend Sell={additional_factors['trend'].get('sell_confidence', 0)}, "
              f"Vol Buy={additional_factors['volatility'].get('buy_confidence', 0)}, Vol Sell={additional_factors['volatility'].get('sell_confidence', 0)}")
    
    # Test both BUY and SELL actions
    for action in ['BUY', 'SELL']:
        # Set base confidence
        base_confidence = 0.1
        
        # Initialize additional confidence and confluence factors
        additional_confidence = 0.0
        confluence_factors = []
        
        # 1. Support/Resistance confidence
        sr_conf = additional_factors['support_resistance'].get('confidence', {})
        logger.info(f"SR confidence: {sr_conf}")
        
        # Handle both dictionary and float formats for SR confidence
        if isinstance(sr_conf, dict):
            if action == 'BUY' and sr_conf.get('buy_confidence', 0) > 0:
                sr_boost = sr_conf.get('buy_confidence', 0)
                additional_confidence += sr_boost
                confluence_factors.append("Near Support Level")
                logger.info(f"Added SR buy confidence: {sr_boost:.4f}")
            elif action == 'SELL' and sr_conf.get('sell_confidence', 0) > 0:
                sr_boost = sr_conf.get('sell_confidence', 0)
                additional_confidence += sr_boost
                confluence_factors.append("Near Resistance Level")
                logger.info(f"Added SR sell confidence: {sr_boost:.4f}")
        elif isinstance(sr_conf, (int, float)) and sr_conf > 0:
            # If it's a single value, apply it to both buy and sell
            additional_confidence += sr_conf
            if action == 'BUY':
                confluence_factors.append("Near Support Level")
            elif action == 'SELL':
                confluence_factors.append("Near Resistance Level")
            logger.info(f"Added SR confidence: {sr_conf:.4f}")
        
        # 2. Trend confidence
        if action == 'BUY' and additional_factors['trend'].get('buy_confidence', 0) > 0:
            trend_boost = additional_factors['trend']['buy_confidence']
            additional_confidence += trend_boost
            logger.info(f"Added trend buy confidence: {trend_boost:.4f}")
            # Check if ADX indicates strong trend
            if additional_factors['trend'].get('adx') and additional_factors['trend'].get('adx') > 25:
                confluence_factors.append("Strong Bullish Trend")
            else:
                confluence_factors.append("Bullish Trend Alignment")
        elif action == 'SELL' and additional_factors['trend'].get('sell_confidence', 0) > 0:
            trend_boost = additional_factors['trend']['sell_confidence']
            additional_confidence += trend_boost
            logger.info(f"Added trend sell confidence: {trend_boost:.4f}")
            # Check if ADX indicates strong trend
            if additional_factors['trend'].get('adx') and additional_factors['trend'].get('adx') > 25:
                confluence_factors.append("Strong Bearish Trend")
            else:
                confluence_factors.append("Bearish Trend Alignment")
        
        # 3. Volatility confidence
        if action == 'BUY' and additional_factors['volatility'].get('buy_confidence', 0) > 0:
            vol_boost = additional_factors['volatility']['buy_confidence']
            additional_confidence += vol_boost
            logger.info(f"Added volatility buy confidence: {vol_boost:.4f}")
            # Check if price is near lower Bollinger Band
            if additional_factors['volatility'].get('bb_percent_b') and additional_factors['volatility'].get('bb_percent_b') < 0.2:
                confluence_factors.append("Oversold Condition")
            else:
                confluence_factors.append("Favorable Volatility")
        elif action == 'SELL' and additional_factors['volatility'].get('sell_confidence', 0) > 0:
            vol_boost = additional_factors['volatility']['sell_confidence']
            additional_confidence += vol_boost
            logger.info(f"Added volatility sell confidence: {vol_boost:.4f}")
            # Check if price is near upper Bollinger Band
            if additional_factors['volatility'].get('bb_percent_b') and additional_factors['volatility'].get('bb_percent_b') > 0.8:
                confluence_factors.append("Overbought Condition")
            else:
                confluence_factors.append("Favorable Volatility")
        
        # Calculate final confidence (cap at 0.95 to avoid overconfidence)
        final_confidence = min(base_confidence + additional_confidence, 0.95)
        
        # Log the confidence calculation for debugging
        logger.info(f"Confidence calculation for {action}: {base_confidence:.4f} (base) + {additional_confidence:.4f} (additional) = {final_confidence:.4f} (final)")
        logger.info(f"Confluence factors: {confluence_factors}")
        
        # Compare with our confidence boost module
        boost_final_confidence, boost_confluence_factors, boost_additional_confidence = boost_confidence(data, action, base_confidence)
        logger.info(f"Confidence boost module calculation for {action}: {base_confidence:.4f} (base) + {boost_additional_confidence:.4f} (additional) = {boost_final_confidence:.4f} (final)")
        logger.info(f"Boost module confluence factors: {boost_confluence_factors}")
        logger.info("---")

if __name__ == "__main__":
    fix_confidence_calculation()
