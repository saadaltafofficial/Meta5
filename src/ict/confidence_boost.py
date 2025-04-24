#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Confidence Boost Module

This module provides a simple way to boost confidence levels based on support/resistance,
trend strength, and volatility measures.
"""

import logging
import pandas as pd
from src.ict.support_resistance import detect_support_resistance, calculate_sr_confidence
from src.ict.trend_indicators import calculate_adx, calculate_ma_trend, calculate_trend_confidence
from src.ict.volatility_measures import calculate_atr, calculate_bollinger_bands, calculate_volatility_confidence

# Configure logging
logger = logging.getLogger(__name__)

def boost_confidence(df, action, base_confidence):
    """
    Boost confidence based on support/resistance, trend, and volatility
    Can potentially convert HOLD signals to BUY or SELL based on strong indicators
    
    Args:
        df (pd.DataFrame): Price data with OHLC values
        action (str): 'BUY', 'SELL', or 'HOLD'
        base_confidence (float): Base confidence from ICT setup
        
    Returns:
        tuple: (final_confidence, confluence_factors, additional_confidence)
    """
    additional_confidence = 0.0
    confluence_factors = []
    
    # If action is HOLD, determine if we should convert to BUY or SELL based on indicators
    if action == 'HOLD':
        # Calculate trend and volatility to determine potential action
        trend_buy, trend_sell = calculate_trend_confidence(df)
        vol_buy, vol_sell = calculate_volatility_confidence(df)
        sr_confidence = 0.0
        
        # Ensure trend_buy and trend_sell are floats
        try:
            trend_buy = float(trend_buy)
            trend_sell = float(trend_sell)
            vol_buy = float(vol_buy)
            vol_sell = float(vol_sell)
        except (ValueError, TypeError):
            # If conversion fails, set to default values
            trend_buy = 0.0
            trend_sell = 0.0
            vol_buy = 0.0
            vol_sell = 0.0
            logger.warning(f"Error converting trend/volatility values to float")
        
        # Get RSI if available
        try:
            rsi = float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else 50.0  # Default to neutral if no RSI
        except (ValueError, TypeError, IndexError):
            rsi = 50.0  # Default to neutral if error
            logger.warning(f"Error getting RSI value, using default 50.0")
        
        # Convert to BUY if any of these conditions are met
        if (trend_buy > 0.1) or (vol_buy > 0.05) or (rsi < 30.0):
            action = 'BUY'
            confluence_factors.append("Bullish Indicators")
            # Add more specific factors
            if trend_buy > 0.1:
                confluence_factors.append("Bullish Trend")
            if vol_buy > 0.05:
                confluence_factors.append("Favorable Volatility")
            if rsi < 30.0:
                confluence_factors.append("Oversold Condition")
            logger.info(f"Converting HOLD to BUY due to bullish indicators (trend={trend_buy:.2f}, vol={vol_buy:.2f}, rsi={rsi:.1f})")
        # Convert to SELL if any of these conditions are met
        elif (trend_sell > 0.1) or (vol_sell > 0.05) or (rsi > 70.0):
            action = 'SELL'
            confluence_factors.append("Bearish Indicators")
            # Add more specific factors
            if trend_sell > 0.1:
                confluence_factors.append("Bearish Trend")
            if vol_sell > 0.05:
                confluence_factors.append("Favorable Volatility")
            if rsi > 70.0:
                confluence_factors.append("Overbought Condition")
            logger.info(f"Converting HOLD to SELL due to bearish indicators (trend={trend_sell:.2f}, vol={vol_sell:.2f}, rsi={rsi:.1f})")
        
        # If we still have HOLD, return early with minimal boost
        if action == 'HOLD':
            # Even for HOLD, add a small confidence boost
            return base_confidence + 0.1, ["Neutral Market"], 0.1
    
    # Only calculate additional confidence for actionable signals
    if action not in ['BUY', 'SELL']:
        return base_confidence, confluence_factors, additional_confidence
    
    try:
        # 1. Support/Resistance confidence
        try:
            # Get the current price
            current_price = df['close'].iloc[-1]
            
            # Detect support and resistance levels
            sr_levels = detect_support_resistance(df, lookback=100, strength_threshold=2)
            sr_conf = calculate_sr_confidence(current_price, sr_levels)
            
            # Add support/resistance confidence with increased weight (1.5x)
            if action == 'BUY' and sr_conf.get('buy_confidence', 0) > 0:
                sr_boost = sr_conf.get('buy_confidence', 0) * 1.5  # Increase weight by 50%
                additional_confidence += sr_boost
                confluence_factors.append("Near Support Level")
                logger.info(f"Added SR buy confidence: {sr_boost:.4f} (weighted)")
            elif action == 'SELL' and sr_conf.get('sell_confidence', 0) > 0:
                sr_boost = sr_conf.get('sell_confidence', 0) * 1.5  # Increase weight by 50%
                additional_confidence += sr_boost
                confluence_factors.append("Near Resistance Level")
                logger.info(f"Added SR sell confidence: {sr_boost:.4f} (weighted)")
        except Exception as e:
            logger.warning(f"Error calculating support/resistance confidence: {e}")
        
        # 2. Trend confidence
        try:
            # Calculate trend indicators
            trend_df = calculate_adx(df, period=14)
            trend_df = calculate_ma_trend(trend_df, fast_period=9, slow_period=21, trend_period=50)
            trend_conf = calculate_trend_confidence(trend_df, lookback=20)
            
            # Add trend confidence with increased weight (2x)
            if action == 'BUY' and trend_conf.get('buy_confidence', 0) > 0:
                trend_boost = trend_conf['buy_confidence'] * 2.0  # Double the weight
                additional_confidence += trend_boost
                logger.info(f"Added trend buy confidence: {trend_boost:.4f} (weighted)")
                # Check if ADX indicates strong trend
                if 'adx' in trend_df.columns and trend_df['adx'].iloc[-1] > 25:
                    confluence_factors.append("Strong Bullish Trend")
                    # Add extra confidence for strong trend
                    additional_confidence += 0.05
                    logger.info(f"Added extra confidence for strong trend: 0.0500")
                else:
                    confluence_factors.append("Bullish Trend Alignment")
            elif action == 'SELL' and trend_conf.get('sell_confidence', 0) > 0:
                trend_boost = trend_conf['sell_confidence'] * 2.0  # Double the weight
                additional_confidence += trend_boost
                logger.info(f"Added trend sell confidence: {trend_boost:.4f} (weighted)")
                # Check if ADX indicates strong trend
                if 'adx' in trend_df.columns and trend_df['adx'].iloc[-1] > 25:
                    confluence_factors.append("Strong Bearish Trend")
                    # Add extra confidence for strong trend
                    additional_confidence += 0.05
                    logger.info(f"Added extra confidence for strong trend: 0.0500")
                else:
                    confluence_factors.append("Bearish Trend Alignment")
        except Exception as e:
            logger.warning(f"Error calculating trend confidence: {e}")
        
        # 3. Volatility confidence
        try:
            # Calculate volatility measures
            vol_df = calculate_atr(df, period=14)
            vol_df = calculate_bollinger_bands(vol_df, period=20, std_dev=2)
            vol_conf = calculate_volatility_confidence(vol_df, lookback=20)
            
            # Add volatility confidence with increased weight (1.75x)
            if action == 'BUY' and vol_conf.get('buy_confidence', 0) > 0:
                vol_boost = vol_conf['buy_confidence'] * 1.75  # Increase weight by 75%
                additional_confidence += vol_boost
                logger.info(f"Added volatility buy confidence: {vol_boost:.4f} (weighted)")
                # Check if price is near lower Bollinger Band
                if 'bb_percent_b' in vol_df.columns and vol_df['bb_percent_b'].iloc[-1] < 0.2:
                    confluence_factors.append("Oversold Condition")
                    # Add extra confidence for oversold condition
                    additional_confidence += 0.05
                    logger.info(f"Added extra confidence for oversold condition: 0.0500")
                else:
                    confluence_factors.append("Favorable Volatility")
            elif action == 'SELL' and vol_conf.get('sell_confidence', 0) > 0:
                vol_boost = vol_conf['sell_confidence'] * 1.75  # Increase weight by 75%
                additional_confidence += vol_boost
                logger.info(f"Added volatility sell confidence: {vol_boost:.4f} (weighted)")
                # Check if price is near upper Bollinger Band
                if 'bb_percent_b' in vol_df.columns and vol_df['bb_percent_b'].iloc[-1] > 0.8:
                    confluence_factors.append("Overbought Condition")
                    # Add extra confidence for overbought condition
                    additional_confidence += 0.05
                    logger.info(f"Added extra confidence for overbought condition: 0.0500")
                else:
                    confluence_factors.append("Favorable Volatility")
        except Exception as e:
            logger.warning(f"Error calculating volatility confidence: {e}")
        
        # Calculate final confidence (capped at 0.95)
        final_confidence = min(base_confidence + additional_confidence, 0.95)
        
        # Log the confidence calculation
        logger.info(f"===== CONFIDENCE BOOST =====")
        logger.info(f"Action: {action}")
        logger.info(f"Base confidence: {base_confidence:.4f}")
        logger.info(f"Additional confidence: {additional_confidence:.4f}")
        logger.info(f"Final confidence: {final_confidence:.4f}")
        logger.info(f"Confluence factors: {confluence_factors}")
        logger.info(f"Execution threshold: 0.25")
        logger.info(f"Would execute: {'Yes' if final_confidence >= 0.25 else 'No'}")
        logger.info(f"===========================")
        
        return final_confidence, confluence_factors, additional_confidence
        
    except Exception as e:
        logger.error(f"Error boosting confidence: {e}")
        return base_confidence, confluence_factors, 0.0
