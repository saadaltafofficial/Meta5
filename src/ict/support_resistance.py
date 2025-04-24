#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Support and Resistance Module

This module implements support and resistance level detection algorithms
for enhancing ICT trading strategies.
"""

import numpy as np
import pandas as pd

def detect_support_resistance(df, lookback=100, strength_threshold=3, proximity_percent=0.001):
    """
    Detect key support and resistance levels using swing highs/lows
    
    Args:
        df (pd.DataFrame): Price data with OHLC values
        lookback (int): Number of candles to look back for identifying levels
        strength_threshold (int): Minimum number of touches required for a valid level
        proximity_percent (float): Percentage proximity for considering price touches
        
    Returns:
        dict: Support and resistance levels with their strengths
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Use a subset of data based on lookback
    if len(data) > lookback:
        data = data.iloc[-lookback:]
    
    # Initialize results
    levels = {
        'support': [],
        'resistance': [],
    }
    
    # Find swing highs and lows
    highs = []
    lows = []
    
    # Use rolling max/min to identify potential swing points
    for i in range(2, len(data) - 2):
        # Swing high: current high is greater than the 2 candles before and after
        if (data['high'].iloc[i] > data['high'].iloc[i-1] and 
            data['high'].iloc[i] > data['high'].iloc[i-2] and
            data['high'].iloc[i] > data['high'].iloc[i+1] and 
            data['high'].iloc[i] > data['high'].iloc[i+2]):
            highs.append((i, data['high'].iloc[i]))
            
        # Swing low: current low is less than the 2 candles before and after
        if (data['low'].iloc[i] < data['low'].iloc[i-1] and 
            data['low'].iloc[i] < data['low'].iloc[i-2] and
            data['low'].iloc[i] < data['low'].iloc[i+1] and 
            data['low'].iloc[i] < data['low'].iloc[i+2]):
            lows.append((i, data['low'].iloc[i]))
    
    # Group similar levels (within proximity)
    grouped_highs = group_levels(highs, proximity_percent)
    grouped_lows = group_levels(lows, proximity_percent)
    
    # Calculate level strengths (number of touches)
    for level, touches in grouped_highs.items():
        if len(touches) >= strength_threshold:
            levels['resistance'].append({
                'level': level,
                'strength': len(touches),
                'touches': touches
            })
    
    for level, touches in grouped_lows.items():
        if len(touches) >= strength_threshold:
            levels['support'].append({
                'level': level,
                'strength': len(touches),
                'touches': touches
            })
    
    # Sort levels by strength (descending)
    levels['resistance'] = sorted(levels['resistance'], key=lambda x: x['strength'], reverse=True)
    levels['support'] = sorted(levels['support'], key=lambda x: x['strength'], reverse=True)
    
    return levels

def group_levels(points, proximity_percent):
    """
    Group similar price levels together
    
    Args:
        points (list): List of (index, price) tuples
        proximity_percent (float): Percentage proximity for grouping
        
    Returns:
        dict: Grouped levels with their touches
    """
    if not points:
        return {}
    
    # Sort points by price
    sorted_points = sorted(points, key=lambda x: x[1])
    
    grouped = {}
    current_group = [sorted_points[0]]
    current_level = sorted_points[0][1]
    
    for i in range(1, len(sorted_points)):
        point_idx, point_price = sorted_points[i]
        
        # Check if this point is within proximity of the current group
        if abs(point_price - current_level) / current_level <= proximity_percent:
            # Add to current group
            current_group.append((point_idx, point_price))
            # Update the level (average of all points in group)
            current_level = sum(p[1] for p in current_group) / len(current_group)
        else:
            # Store the current group and start a new one
            grouped[current_level] = current_group
            current_group = [(point_idx, point_price)]
            current_level = point_price
    
    # Add the last group
    if current_group:
        grouped[current_level] = current_group
    
    return grouped

def is_near_level(price, levels, proximity_percent=0.001):
    """
    Check if price is near any support or resistance level
    
    Args:
        price (float): Current price to check
        levels (dict): Support and resistance levels
        proximity_percent (float): Percentage proximity for considering price near level
        
    Returns:
        tuple: (bool, str, float) - Is near level, level type, and the level price
    """
    # Check resistance levels
    for res in levels['resistance']:
        level = res['level']
        if abs(price - level) / level <= proximity_percent:
            return True, 'resistance', level
    
    # Check support levels
    for sup in levels['support']:
        level = sup['level']
        if abs(price - level) / level <= proximity_percent:
            return True, 'support', level
    
    return False, None, None

def calculate_sr_confidence(price, levels, proximity_percent=0.003):
    """
    Calculate confidence based on proximity to support/resistance levels
    
    Args:
        price (float): Current price
        levels (dict): Support and resistance levels
        proximity_percent (float): Percentage proximity for confidence calculation
        
    Returns:
        float: Confidence value (0.0 to 0.2)
    """
    confidence = 0.0
    
    # Find closest levels
    closest_res = None
    closest_res_dist = float('inf')
    for res in levels['resistance']:
        level = res['level']
        dist = abs(price - level) / level
        if dist < closest_res_dist:
            closest_res_dist = dist
            closest_res = res
    
    closest_sup = None
    closest_sup_dist = float('inf')
    for sup in levels['support']:
        level = sup['level']
        dist = abs(price - level) / level
        if dist < closest_sup_dist:
            closest_sup_dist = dist
            closest_sup = sup
    
    # Calculate confidence based on proximity and strength
    if closest_res and closest_res_dist <= proximity_percent:
        # Price is near resistance - good for SELL signals
        confidence_factor = (1 - (closest_res_dist / proximity_percent)) * min(closest_res['strength'] / 5, 1)
        confidence += confidence_factor * 0.2  # Max 0.2 confidence from S/R
    
    if closest_sup and closest_sup_dist <= proximity_percent:
        # Price is near support - good for BUY signals
        confidence_factor = (1 - (closest_sup_dist / proximity_percent)) * min(closest_sup['strength'] / 5, 1)
        confidence += confidence_factor * 0.2  # Max 0.2 confidence from S/R
    
    return confidence
