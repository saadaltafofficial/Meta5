#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Volatility Measures Module

This module implements volatility measures for enhancing ICT trading strategies.
"""

import numpy as np
import pandas as pd

def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR) to measure volatility
    
    Args:
        df (pd.DataFrame): Price data with OHLC values
        period (int): Period for ATR calculation
        
    Returns:
        pd.DataFrame: DataFrame with ATR column added
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Calculate True Range (TR)
    data['tr0'] = abs(data['high'] - data['low'])
    data['tr1'] = abs(data['high'] - data['close'].shift(1))
    data['tr2'] = abs(data['low'] - data['close'].shift(1))
    data['tr'] = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    
    # Calculate ATR
    data['atr'] = data['tr'].rolling(window=period).mean()
    
    # Calculate normalized ATR (as percentage of price)
    data['atr_percent'] = data['atr'] / data['close'] * 100
    
    # Clean up temporary columns
    data.drop(columns=['tr0', 'tr1', 'tr2', 'tr'], inplace=True)
    
    return data

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """
    Calculate Bollinger Bands to measure volatility and potential reversals
    
    Args:
        df (pd.DataFrame): Price data with OHLC values
        period (int): Period for moving average
        std_dev (float): Number of standard deviations for bands
        
    Returns:
        pd.DataFrame: DataFrame with Bollinger Bands columns added
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Calculate middle band (simple moving average)
    data['bb_middle'] = data['close'].rolling(window=period).mean()
    
    # Calculate standard deviation
    data['bb_std'] = data['close'].rolling(window=period).std()
    
    # Calculate upper and lower bands
    data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * std_dev)
    data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * std_dev)
    
    # Calculate bandwidth (measure of volatility)
    data['bb_bandwidth'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
    
    # Calculate %B (position within bands)
    data['bb_percent_b'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
    
    return data

def calculate_volatility_confidence(df, lookback=20):
    """
    Calculate confidence based on volatility measures
    
    Args:
        df (pd.DataFrame): Price data with volatility measures
        lookback (int): Number of candles to look back for volatility analysis
        
    Returns:
        dict: Volatility confidence values for BUY and SELL signals
    """
    # Use recent data for volatility analysis
    recent_df = df.iloc[-lookback:].copy()
    
    # Initialize confidence values
    buy_confidence = 0.0
    sell_confidence = 0.0
    
    # Check ATR for favorable volatility conditions
    if 'atr_percent' in recent_df.columns:
        atr_percent = recent_df['atr_percent'].iloc[-1]
        if not pd.isna(atr_percent):
            # Check if volatility is in a good range (not too low, not too high)
            if 0.5 <= atr_percent <= 1.5:
                # Moderate volatility - good for both buy and sell
                buy_confidence += 0.05
                sell_confidence += 0.05
            elif atr_percent > 1.5:
                # High volatility - better for quick trades
                buy_confidence += 0.03
                sell_confidence += 0.03
    
    # Check Bollinger Bands for potential reversals
    if all(col in recent_df.columns for col in ['bb_percent_b', 'bb_bandwidth']):
        percent_b = recent_df['bb_percent_b'].iloc[-1]
        bandwidth = recent_df['bb_bandwidth'].iloc[-1]
        
        if not pd.isna(percent_b) and not pd.isna(bandwidth):
            # Check for oversold conditions (price near lower band)
            if percent_b < 0.1:
                # Strong oversold - good for BUY
                buy_confidence += 0.15
            elif percent_b < 0.2:
                # Moderate oversold - good for BUY
                buy_confidence += 0.1
                
            # Check for overbought conditions (price near upper band)
            if percent_b > 0.9:
                # Strong overbought - good for SELL
                sell_confidence += 0.15
            elif percent_b > 0.8:
                # Moderate overbought - good for SELL
                sell_confidence += 0.1
                
            # Check for volatility expansion/contraction
            avg_bandwidth = recent_df['bb_bandwidth'].rolling(window=10).mean().iloc[-1]
            if not pd.isna(avg_bandwidth):
                if bandwidth < avg_bandwidth * 0.7:
                    # Volatility contraction - potential breakout coming
                    buy_confidence += 0.05
                    sell_confidence += 0.05
    
    return {
        'buy_confidence': min(buy_confidence, 0.2),  # Cap at 0.2
        'sell_confidence': min(sell_confidence, 0.2)  # Cap at 0.2
    }
