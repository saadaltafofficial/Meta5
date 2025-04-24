#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trend Strength Indicators Module

This module implements trend strength indicators for enhancing ICT trading strategies.
"""

import numpy as np
import pandas as pd

def calculate_adx(df, period=14):
    """
    Calculate Average Directional Index (ADX) to measure trend strength
    
    Args:
        df (pd.DataFrame): Price data with OHLC values
        period (int): Period for ADX calculation
        
    Returns:
        pd.DataFrame: DataFrame with ADX, +DI, and -DI columns added
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Calculate True Range (TR)
    data['tr0'] = abs(data['high'] - data['low'])
    data['tr1'] = abs(data['high'] - data['close'].shift(1))
    data['tr2'] = abs(data['low'] - data['close'].shift(1))
    data['tr'] = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    
    # Calculate +DM and -DM
    data['up_move'] = data['high'] - data['high'].shift(1)
    data['down_move'] = data['low'].shift(1) - data['low']
    
    data['+dm'] = np.where((data['up_move'] > data['down_move']) & (data['up_move'] > 0), data['up_move'], 0)
    data['-dm'] = np.where((data['down_move'] > data['up_move']) & (data['down_move'] > 0), data['down_move'], 0)
    
    # Calculate smoothed TR, +DM, and -DM
    data['smoothed_tr'] = data['tr'].rolling(window=period).sum()
    data['smoothed_+dm'] = data['+dm'].rolling(window=period).sum()
    data['smoothed_-dm'] = data['-dm'].rolling(window=period).sum()
    
    # Calculate +DI and -DI
    data['+di'] = 100 * (data['smoothed_+dm'] / data['smoothed_tr'])
    data['-di'] = 100 * (data['smoothed_-dm'] / data['smoothed_tr'])
    
    # Calculate DX and ADX
    data['dx'] = 100 * abs(data['+di'] - data['-di']) / (data['+di'] + data['-di'])
    data['adx'] = data['dx'].rolling(window=period).mean()
    
    # Clean up temporary columns
    cols_to_drop = ['tr0', 'tr1', 'tr2', 'tr', 'up_move', 'down_move', '+dm', '-dm',
                   'smoothed_tr', 'smoothed_+dm', 'smoothed_-dm', 'dx']
    data.drop(columns=cols_to_drop, inplace=True)
    
    return data

def calculate_ma_trend(df, fast_period=9, slow_period=21, trend_period=50):
    """
    Calculate moving average trend indicators
    
    Args:
        df (pd.DataFrame): Price data with OHLC values
        fast_period (int): Fast MA period
        slow_period (int): Slow MA period
        trend_period (int): Trend MA period
        
    Returns:
        pd.DataFrame: DataFrame with MA trend indicators added
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Calculate moving averages
    data['ma_fast'] = data['close'].rolling(window=fast_period).mean()
    data['ma_slow'] = data['close'].rolling(window=slow_period).mean()
    data['ma_trend'] = data['close'].rolling(window=trend_period).mean()
    
    # Calculate MA crossovers
    data['ma_cross'] = np.where(data['ma_fast'] > data['ma_slow'], 1, -1)
    data['ma_cross_change'] = data['ma_cross'].diff()
    
    # Calculate price relative to trend
    data['price_above_trend'] = np.where(data['close'] > data['ma_trend'], 1, -1)
    
    # Calculate trend strength
    data['trend_strength'] = abs(data['ma_fast'] - data['ma_slow']) / data['ma_slow'] * 100
    
    return data

def calculate_trend_confidence(df, lookback=20):
    """
    Calculate confidence based on trend strength indicators
    
    Args:
        df (pd.DataFrame): Price data with trend indicators
        lookback (int): Number of candles to look back for trend analysis
        
    Returns:
        dict: Trend confidence values for BUY and SELL signals
    """
    # Use recent data for trend analysis
    recent_df = df.iloc[-lookback:].copy()
    
    # Initialize confidence values
    buy_confidence = 0.0
    sell_confidence = 0.0
    
    # Check if ADX indicates a strong trend (ADX > 25)
    if 'adx' in recent_df.columns:
        adx_value = recent_df['adx'].iloc[-1]
        if not pd.isna(adx_value):
            if adx_value > 25:
                # Strong trend detected
                trend_factor = min((adx_value - 25) / 25, 1.0)  # Scale 25-50 to 0-1
                
                # Check trend direction using DI lines
                if recent_df['+di'].iloc[-1] > recent_df['-di'].iloc[-1]:
                    # Bullish trend
                    buy_confidence += trend_factor * 0.15
                else:
                    # Bearish trend
                    sell_confidence += trend_factor * 0.15
    
    # Check moving average trends
    if all(col in recent_df.columns for col in ['ma_fast', 'ma_slow', 'ma_trend']):
        # Check if price is above/below trend MA
        if recent_df['close'].iloc[-1] > recent_df['ma_trend'].iloc[-1]:
            # Price above trend - bullish
            buy_confidence += 0.05
        else:
            # Price below trend - bearish
            sell_confidence += 0.05
        
        # Check MA crossovers
        if 'ma_cross_change' in recent_df.columns:
            # Recent bullish crossover (fast MA crosses above slow MA)
            if recent_df['ma_cross_change'].iloc[-3:].eq(2).any():
                buy_confidence += 0.1
            # Recent bearish crossover (fast MA crosses below slow MA)
            elif recent_df['ma_cross_change'].iloc[-3:].eq(-2).any():
                sell_confidence += 0.1
        
        # Check if fast MA is above/below slow MA
        if recent_df['ma_fast'].iloc[-1] > recent_df['ma_slow'].iloc[-1]:
            # Bullish alignment
            buy_confidence += 0.05
        else:
            # Bearish alignment
            sell_confidence += 0.05
    
    return {
        'buy_confidence': min(buy_confidence, 0.25),  # Cap at 0.25
        'sell_confidence': min(sell_confidence, 0.25)  # Cap at 0.25
    }
